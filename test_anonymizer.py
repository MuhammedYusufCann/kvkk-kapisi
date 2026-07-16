"""python -m unittest -v

Testler bilerek NER modelini yuklemez: kural katmani, kasa ve BIO birlestirmesi
model agirligi olmadan sinanabilsin diye ner_bulgulari disaridan enjekte ediliyor.
"""

import unittest
from unittest import mock

from kvkk.anonymizer import (
    Finding,
    Vault,
    _iban_gecerli,
    maskele,
    sizinti_denetle,
    tckn_gecerli,
)
from kvkk.ner import _bio_birlestir, _kelime_sinirina_yasla, _unvan_ayikla

# Uretilmis, saglamasi tutan sahte veriler.
GECERLI_TCKN = "20433218148"
GECERLI_IBAN = "TR97 6155 9407 8161 8495 9310 34"


class TCKNTest(unittest.TestCase):
    def test_gecerli_tckn_kabul_edilir(self):
        self.assertTrue(tckn_gecerli(GECERLI_TCKN))

    def test_saglamasi_tutmayan_reddedilir(self):
        self.assertFalse(tckn_gecerli("12345678901"))

    def test_sifirla_baslayamaz(self):
        self.assertFalse(tckn_gecerli("01234567890"))

    def test_yanlis_uzunluk_reddedilir(self):
        self.assertFalse(tckn_gecerli("1234567895"))

    def test_rastgele_11_hane_maskelenmez(self):
        # Saglama olmasa fatura/dosya numaralari TCKN sanilirdi.
        maskeli, _, _ = maskele("Dosya numarasi 20433218149 olarak kaydedildi.")
        self.assertIn("20433218149", maskeli)


class IBANTest(unittest.TestCase):
    def test_gecerli_iban_maskelenir(self):
        maskeli, kasa, _ = maskele(f"Hesap: {GECERLI_IBAN}")
        self.assertEqual(maskeli, "Hesap: [IBAN_1]")
        self.assertEqual(kasa.esleme["[IBAN_1]"], GECERLI_IBAN)

    def test_bitisik_yazim_da_yakalanir(self):
        bitisik = GECERLI_IBAN.replace(" ", "")
        maskeli, _, _ = maskele(f"Hesap: {bitisik}")
        self.assertEqual(maskeli, "Hesap: [IBAN_1]")

    def test_kontrol_hanesi_bozuk_iban_reddedilir(self):
        self.assertFalse(_iban_gecerli("TR33 6155 9407 8161 8495 9310 34"))


class MaskelemeTest(unittest.TestCase):
    def test_ayni_kisi_ayni_etiketi_alir(self):
        metin = "Ayse Demir geldi. Sonra Ayse Demir gitti."
        bulgular = [
            Finding("KISI", "Ayse Demir", 0, 10, "model"),
            Finding("KISI", "Ayse Demir", 24, 34, "model"),
        ]
        maskeli, kasa, _ = maskele(metin, bulgular)
        self.assertEqual(maskeli, "[KISI_1] geldi. Sonra [KISI_1] gitti.")
        self.assertEqual(len(kasa), 1)

    def test_buyuk_kucuk_harf_farki_ayni_kisi_sayilir(self):
        kasa = Vault()
        self.assertEqual(
            kasa.etiket_ver("KISI", "Ayse Demir"),
            kasa.etiket_ver("KISI", "AYSE DEMIR"),
        )

    def test_farkli_kisiler_farkli_etiket_alir(self):
        metin = "Ayse Demir ve Kemal Dogan"
        bulgular = [
            Finding("KISI", "Ayse Demir", 0, 10, "model"),
            Finding("KISI", "Kemal Dogan", 14, 25, "model"),
        ]
        maskeli, _, _ = maskele(metin, bulgular)
        self.assertEqual(maskeli, "[KISI_1] ve [KISI_2]")

    def test_kural_katmani_modeli_yener(self):
        # Model TCKN'yi yanlislikla kisi adi sanarsa kural katmani kazanmali.
        metin = f"Kimlik {GECERLI_TCKN} numarali kisi"
        yanlis = Finding("KISI", GECERLI_TCKN, 7, 18, "model")
        _, _, bulgular = maskele(metin, [yanlis])
        self.assertEqual([b.tur for b in bulgular], ["TCKN"])

    def test_kapatilan_tur_maskelenmez(self):
        metin = f"Hesap {GECERLI_IBAN} ve tel 0532 111 22 33"
        maskeli, _, _ = maskele(metin, turler=("TELEFON",))
        self.assertIn(GECERLI_IBAN, maskeli)
        self.assertIn("[TELEFON_1]", maskeli)

    def test_bulunmayan_veri_metni_bozmaz(self):
        metin = "Bu belgede kisisel veri yok."
        maskeli, kasa, bulgular = maskele(metin)
        self.assertEqual(maskeli, metin)
        self.assertEqual(len(kasa), 0)
        self.assertEqual(bulgular, [])


class CozmeTest(unittest.TestCase):
    def test_gidis_donus_metni_aynen_geri_verir(self):
        metin = f"Ahmet Yilmaz, {GECERLI_TCKN}, hesap {GECERLI_IBAN}"
        bulgular = [Finding("KISI", "Ahmet Yilmaz", 0, 12, "model")]
        maskeli, kasa, _ = maskele(metin, bulgular)
        self.assertEqual(kasa.coz(maskeli), metin)

    def test_llm_cevabindaki_etiket_geri_doldurulur(self):
        kasa = Vault()
        kasa.etiket_ver("KISI", "Ahmet Yilmaz")
        self.assertEqual(
            kasa.coz("Hastanin adi [KISI_1] ve durumu iyi."),
            "Hastanin adi Ahmet Yilmaz ve durumu iyi.",
        )

    def test_iki_haneli_etiket_tek_haneliyle_karistirilmaz(self):
        # Naif string replace'te [KISI_1] deseni [KISI_11] icinde eslesir.
        kasa = Vault()
        for i in range(1, 12):
            kasa.etiket_ver("KISI", f"Kisi Numara {i}")
        self.assertEqual(kasa.coz("[KISI_11]"), "Kisi Numara 11")


class BIOBirlestirmeTest(unittest.TestCase):
    """transformers'in kendi aggregation'i bu vakalarda span kirpiyordu."""

    def test_b_ve_i_tek_varlikta_birlesir(self):
        metin = "Hasta Ahmet Yilmaz geldi."
        tokenlar = [
            {"entity": "O", "start": 0, "end": 5, "score": 0.99},
            {"entity": "B-PER", "start": 6, "end": 11, "score": 0.99},
            {"entity": "I-PER", "start": 12, "end": 18, "score": 0.99},
            {"entity": "O", "start": 19, "end": 24, "score": 0.99},
        ]
        (bulgu,) = _bio_birlestir(metin, tokenlar)
        self.assertEqual(bulgu.metin, "Ahmet Yilmaz")
        self.assertEqual((bulgu.baslangic, bulgu.bitis), (6, 18))

    def test_o_etiketli_alt_kelime_parcasi_maskeye_dahil_olur(self):
        # 'Yilmaz' -> 'Yilm' + '##az' bolunup ##az O etiketlenirse span yarim
        # kalir ve maskeli metinde '[KISI_1]az' sizintisi olusur.
        metin = "Hasta Yilmaz geldi."
        tokenlar = [
            {"entity": "O", "start": 0, "end": 5, "score": 0.99},
            {"entity": "B-PER", "start": 6, "end": 10, "score": 0.95},
            {"entity": "O", "start": 10, "end": 12, "score": 0.60},
        ]
        (bulgu,) = _bio_birlestir(metin, tokenlar)
        self.assertEqual(bulgu.metin, "Yilmaz")
        maskeli, _, _ = maskele(metin, [bulgu])
        self.assertEqual(maskeli, "Hasta [KISI_1] geldi.")

    def test_ardisik_iki_kisi_ayrilir(self):
        metin = "Ayse Demir Kemal Dogan"
        tokenlar = [
            {"entity": "B-PER", "start": 0, "end": 4, "score": 0.99},
            {"entity": "I-PER", "start": 5, "end": 10, "score": 0.99},
            {"entity": "B-PER", "start": 11, "end": 16, "score": 0.99},
            {"entity": "I-PER", "start": 17, "end": 22, "score": 0.99},
        ]
        bulgular = _bio_birlestir(metin, tokenlar)
        self.assertEqual([b.metin for b in bulgular], ["Ayse Demir", "Kemal Dogan"])

    def test_dusuk_skorlu_bulgu_elenir(self):
        metin = "Belki Zeynep belki degil"
        tokenlar = [{"entity": "B-PER", "start": 6, "end": 12, "score": 0.20}]
        self.assertEqual(_bio_birlestir(metin, tokenlar), [])

    def test_turkce_cekim_eki_maskenin_disinda_kalir(self):
        metin = "Ankara'da yasiyor"
        self.assertEqual(_kelime_sinirina_yasla(metin, 0, 6), (0, 6))

    def test_yaslama_kelimenin_tamamini_kapsar(self):
        metin = "Hasta Yilmaz geldi."
        self.assertEqual(_kelime_sinirina_yasla(metin, 6, 10), (6, 12))


class SizintiDenetimiTest(unittest.TestCase):
    def test_temiz_maskeli_metin_sizinti_vermez(self):
        metin = f"Ahmet Yilmaz, {GECERLI_TCKN}"
        bulgular = [Finding("KISI", "Ahmet Yilmaz", 0, 12, "model")]
        maskeli, kasa, _ = maskele(metin, bulgular)
        self.assertEqual(sizinti_denetle(maskeli, kasa), [])

    def test_maskelenmemis_tckn_kritik_sizinti_sayilir(self):
        kasa = Vault()
        kasa.etiket_ver("TCKN", GECERLI_TCKN)
        (sizinti,) = sizinti_denetle(f"Kimlik: {GECERLI_TCKN}", kasa)
        self.assertTrue(sizinti.kritik)
        self.assertEqual(sizinti.deger, GECERLI_TCKN)

    def test_iban_ayraclari_degisse_de_yakalanir(self):
        kasa = Vault()
        kasa.etiket_ver("IBAN", GECERLI_IBAN)
        bitisik = GECERLI_IBAN.replace(" ", "")
        (sizinti,) = sizinti_denetle(f"Hesap {bitisik}", kasa)
        self.assertTrue(sizinti.kritik)

    def test_yer_adi_kurum_icinde_gecerse_kritik_degil(self):
        # 'Ankara' maskelenmisken 'Ankara Numune Hastanesi' maskelenmemisse
        # kelime istekte gecer; bu bir kimlik ifsasi degil.
        kasa = Vault()
        kasa.etiket_ver("YER", "Ankara")
        (sizinti,) = sizinti_denetle("Kurum: Ankara Numune Hastanesi", kasa)
        self.assertFalse(sizinti.kritik)

    def test_alt_dizi_eslesmesi_yanlis_alarm_uretmez(self):
        # Kelime siniri olmadan 'Ali' her 'Kalite' kelimesinde eslesirdi.
        kasa = Vault()
        kasa.etiket_ver("KISI", "Ali")
        self.assertEqual(sizinti_denetle("Kalite kontrol raporu", kasa), [])


class DusenOturumTest(unittest.TestCase):
    """Kasa kaybolunca uygulama sessizce yanlis cevap vermemeli.

    Sayfa yenilenir ya da sunucu yeniden baslarsa gr.State sifirlanir; maskeli
    metin kutuda kalir ama kasa gider. Bu sessizce gecilirse etiketli cevap
    'acilmis' diye gosterilir ve denetim '0 deger tarandi' deyip temiz raporlar.
    """

    def test_kasasiz_etiketli_metin_gorunur_hata_verir(self):
        from app import soru_sor

        acik, ham, istek, denetim = soru_sor("Hasta [KISI_1] geldi.", "kim geldi", None)
        self.assertIn("Oturum sifirlanmis", denetim)
        self.assertEqual(acik, "")
        self.assertEqual(ham, "")

    def test_bos_kasa_da_ayni_sekilde_yakalanir(self):
        from app import soru_sor

        _, _, _, denetim = soru_sor("Hasta [KISI_1] geldi.", "kim geldi", Vault())
        self.assertIn("Oturum sifirlanmis", denetim)

    def test_etiketsiz_metin_kasasiz_da_calisabilir(self):
        # Icinde kisisel veri bulunmayan belge icin kasa bos olmasi normal, bu
        # durum 'oturum dustu' sayilmamali. LLM cagrisi taklit ediliyor: testler
        # aga cikmamali.
        import app
        from kvkk.llm import LLMCevap

        sahte = LLMCevap(metin="ozet", giden_istek="istek", saglayici="sahte")
        with mock.patch.object(app, "sor", return_value=sahte) as cagri:
            _, _, _, denetim = app.soru_sor(
                "Bu belgede kisisel veri yok.", "ozetle", Vault()
            )
        self.assertNotIn("Oturum sifirlanmis", denetim)
        cagri.assert_called_once()

    def test_guard_llme_hic_gitmeden_doner(self):
        # Oturum dustuyse bulut cagrisi hic yapilmamali.
        import app

        with mock.patch.object(app, "sor") as cagri:
            app.soru_sor("Hasta [KISI_1] geldi.", "kim geldi", None)
        cagri.assert_not_called()


class UnvanTest(unittest.TestCase):
    """Model 'Dr.'yi tek basina kisi adi sayiyordu; unvan ismin icine karisinca
    ayni kisi unvanli/unvansiz gecislerinde farkli etiket aliyordu."""

    def test_tek_basina_unvan_kisi_sayilmaz(self):
        metin = "Konsultan Hekim: Dr. Ayse Kandemir"
        self.assertIsNone(_unvan_ayikla(metin, 17, 20))  # "Dr."

    def test_unvan_isim_basindan_kirpilir(self):
        metin = "Dr. Mehmet Ozturk raporu yazdi."
        bas, bit = _unvan_ayikla(metin, 0, 17)
        self.assertEqual(metin[bas:bit], "Mehmet Ozturk")

    def test_ust_uste_unvanlar_kirpilir(self):
        metin = "Prof. Dr. Ayse Kandemir"
        bas, bit = _unvan_ayikla(metin, 0, len(metin))
        self.assertEqual(metin[bas:bit], "Ayse Kandemir")

    def test_unvanli_ve_unvansiz_gecis_ayni_etiketi_alir(self):
        metin = "Dr. Ayse Kandemir geldi. Ayse Kandemir imzaladi."
        tokenlar = [
            {"entity": "B-PER", "start": 0, "end": 3, "score": 0.90},
            {"entity": "I-PER", "start": 4, "end": 8, "score": 0.99},
            {"entity": "I-PER", "start": 9, "end": 17, "score": 0.99},
            {"entity": "B-PER", "start": 25, "end": 29, "score": 0.99},
            {"entity": "I-PER", "start": 30, "end": 38, "score": 0.99},
        ]
        bulgular = _bio_birlestir(metin, tokenlar)
        maskeli, kasa, _ = maskele(metin, bulgular)
        self.assertEqual(maskeli, "Dr. [KISI_1] geldi. [KISI_1] imzaladi.")
        self.assertEqual(len(kasa), 1)

    def test_unvan_olmayan_isim_bozulmaz(self):
        metin = "Ahmet Yilmaz geldi."
        bas, bit = _unvan_ayikla(metin, 0, 12)
        self.assertEqual(metin[bas:bit], "Ahmet Yilmaz")


if __name__ == "__main__":
    unittest.main()
