"""Belge buluttaki LLM'e gitmeden once kisisel veriyi maskeleyen katman.

Iki kaynagi birlestirir:
  - kural katmani: TCKN, IBAN, telefon, e-posta, plaka gibi bicimi belli veriler
  - model katmani: HuggingFace Turkce NER modeli ile kisi / yer / kurum adlari

Kural katmani modelin ustundedir: cakisma halinde kural kazanir. Hazir NER
modeli TCKN diye bir kavram bilmedigi icin bu iki katman birbirinin acigini
kapatir.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# NER etiketlerinden bizim tur adlarimiza. Etiketler bilerek ASCII: maskeli
# metin LLM'e gidip geri donerken Turkce karakterlerin bozulma riski olmasin.
NER_TUR_ESLESME = {
    "PER": "KISI",
    "PERSON": "KISI",
    "LOC": "YER",
    "LOCATION": "YER",
    "ORG": "KURUM",
    "ORGANIZATION": "KURUM",
}

# KURUM varsayilan olarak kapali, iki sebeple:
#   1. Kurum adi tek basina kisisel veri degil; maskelenince LLM'in isine
#      yarayan baglam bosuna kayboluyor.
#   2. Olculdu: model tibbi metinde KURUM etiketini guvenilmez veriyor —
#      'hipertansiyon' (0.83) ve 'Kardiyoloji' (0.88) kurum sanildi. Acilirsa
#      hastanin tanisi maskelenir ve asistan ise yaramaz hale gelir.
VARSAYILAN_TURLER = ("KISI", "YER", "TCKN", "IBAN", "TELEFON", "EPOSTA", "PLAKA")

MODEL_ADI = "savasy/bert-base-turkish-ner-cased"


@dataclass(frozen=True)
class Finding:
    """Metinde tespit edilmis tek bir kisisel veri parcasi."""

    tur: str
    metin: str
    baslangic: int
    bitis: int
    kaynak: str  # "kural" | "model"
    skor: float = 1.0

    @property
    def uzunluk(self) -> int:
        return self.bitis - self.baslangic


def tckn_gecerli(deger: str) -> bool:
    """TCKN'nin resmi saglama algoritmasi.

    Ham '11 haneli sayi' regex'i fatura tutarindan dosya numarasina kadar her
    seyi yakalar. Saglama olmadan bu katman kullanilamaz kadar cok yanlis pozitif
    uretiyor.
    """
    if len(deger) != 11 or not deger.isdigit() or deger[0] == "0":
        return False

    h = [int(c) for c in deger]
    tek_toplam = h[0] + h[2] + h[4] + h[6] + h[8]
    cift_toplam = h[1] + h[3] + h[5] + h[7]

    if (tek_toplam * 7 - cift_toplam) % 10 != h[9]:
        return False
    return sum(h[:10]) % 10 == h[10]


def _luhn_gecerli(rakamlar: str) -> bool:
    toplam = 0
    for i, c in enumerate(reversed(rakamlar)):
        d = int(c)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        toplam += d
    return toplam % 10 == 0


def _iban_gecerli(ham: str) -> bool:
    """TR IBAN: 26 karakter ve mod-97 saglamasi."""
    sade = re.sub(r"\s", "", ham).upper()
    if len(sade) != 26 or not sade.startswith("TR") or not sade[2:].isdigit():
        return False
    # IBAN mod-97: ilk 4 karakteri sona al, harfleri sayiya cevir (A=10...)
    tasinmis = sade[4:] + sade[:4]
    sayisal = "".join(
        str(ord(c) - 55) if c.isalpha() else c for c in tasinmis
    )
    return int(sayisal) % 97 == 1


# Her kural: (tur, regex, dogrulayici). Dogrulayici None ise regex yeterli.
KURALLAR: list[tuple[str, re.Pattern[str], object]] = [
    ("TCKN", re.compile(r"\b[1-9][0-9]{10}\b"), tckn_gecerli),
    # Desen bilerek IBAN'in resmi gruplamasina birebir sabit. Daha gevsek bir
    # desen ('TR' + rakam/bosluk yigini) tembel eslesmeyle IBAN'in kisa bir
    # onekini yakaliyor, dogrulayici onu reddediyor ve finditer reddedilen
    # eslesmenin ardindan devam ederek gercek IBAN'i hic denemiyordu.
    # Tek desen hem bosluklu hem bitisik yazimi karsilar (\s? istege bagli).
    (
        "IBAN",
        re.compile(r"\bTR\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE),
        _iban_gecerli,
    ),
    (
        "TELEFON",
        re.compile(r"(?:\+90|0)?\s?\(?5\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b"),
        None,
    ),
    (
        "EPOSTA",
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        None,
    ),
    (
        "KREDIKARTI",
        re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
        lambda s: _luhn_gecerli(re.sub(r"[\s-]", "", s)),
    ),
    (
        "PLAKA",
        re.compile(r"\b(0[1-9]|[1-7][0-9]|8[01])\s?[A-Z]{1,3}\s?\d{2,4}\b"),
        None,
    ),
]


def _anahtar(tur: str, metin: str) -> str:
    """Ayni varligin her gecisi ayni etiketi almali.

    'Ahmet Yilmaz', 'AHMET YILMAZ' ve 'Ahmet  Yilmaz' tek bir kisidir; TCKN'de
    ise araya girmis bosluk/tire onemsizdir.
    """
    if tur in {"TCKN", "IBAN", "TELEFON", "KREDIKARTI", "PLAKA"}:
        return tur + ":" + re.sub(r"[\s.\-()]", "", metin).upper()
    sade = unicodedata.normalize("NFKC", metin).casefold()
    return tur + ":" + re.sub(r"\s+", " ", sade).strip()


class Vault:
    """Etiket <-> gercek deger eslemesi.

    Bu esleme yalnizca bellekte durur ve hicbir zaman LLM'e gonderilmez;
    projenin tum iddiasi bu sinifin disari sizmamasina dayanir.
    """

    def __init__(self) -> None:
        self._etiketten_degere: dict[str, str] = {}
        self._anahtardan_etikete: dict[str, str] = {}
        self._sayaclar: dict[str, int] = {}

    def etiket_ver(self, tur: str, metin: str) -> str:
        anahtar = _anahtar(tur, metin)
        mevcut = self._anahtardan_etikete.get(anahtar)
        if mevcut is not None:
            return mevcut

        self._sayaclar[tur] = self._sayaclar.get(tur, 0) + 1
        etiket = f"[{tur}_{self._sayaclar[tur]}]"
        self._anahtardan_etikete[anahtar] = etiket
        self._etiketten_degere[etiket] = metin
        return etiket

    def coz(self, metin: str) -> str:
        """LLM cevabindaki etiketleri gercek degerlerle geri doldurur."""
        if not self._etiketten_degere:
            return metin
        # Uzun etiketler once: [KISI_11] varken [KISI_1] ile eslesmeyelim.
        desen = "|".join(
            re.escape(e)
            for e in sorted(self._etiketten_degere, key=len, reverse=True)
        )
        return re.sub(desen, lambda m: self._etiketten_degere[m.group(0)], metin)

    @property
    def esleme(self) -> dict[str, str]:
        return dict(self._etiketten_degere)

    def __len__(self) -> int:
        return len(self._etiketten_degere)


# Bicimi sabit, ayraci onemsiz veriler: 'TR97 6155...' ile bitisik yazimi ayni
# sey. Bunlarda ayraclari atip alt-dizi aramak dogru sonuc verir.
YAPISAL_TURLER = frozenset({"TCKN", "IBAN", "TELEFON", "KREDIKARTI", "PLAKA"})

# Sizmasi dogrudan kimlik ifsasi olan turler. YER/KURUM bunun disinda: 'Ankara'
# gibi bir sehir adi, maskelenmemis bir kurum adinin ('Ankara Numune Hastanesi')
# icinde de gecer ve bu bir ihlal degil, bilgilendirme konusudur.
KRITIK_TURLER = frozenset({"TCKN", "IBAN", "TELEFON", "KREDIKARTI", "PLAKA", "EPOSTA", "KISI"})


@dataclass(frozen=True)
class Sizinti:
    etiket: str
    deger: str
    kritik: bool


def _tur_of(etiket: str) -> str:
    return etiket.strip("[]").rsplit("_", 1)[0]


def sizinti_denetle(giden_metin: str, kasa: Vault) -> list[Sizinti]:
    """Buluta gidecek metni, kasadaki her gercek deger icin tarar.

    Maskelemenin dogrulugunu iddia etmek yerine olcer: kritik bir sizinti varsa
    o istek gonderilmemeli.

    Serbest metin turlerinde (KISI/YER/KURUM) arama kelime siniri ile yapilir;
    ayraclari atilmis ham alt-dizi aramasi 'Ankara'yi 'ankaranumunehastanesi'
    icinde bulup yanlis alarm uretiyordu.
    """
    sizanlar: list[Sizinti] = []
    yapisal_hedef = re.sub(r"[\s.\-()]", "", giden_metin).casefold()

    for etiket, deger in kasa.esleme.items():
        tur = _tur_of(etiket)
        if tur in YAPISAL_TURLER:
            sade = re.sub(r"[\s.\-()]", "", deger).casefold()
            bulundu = bool(sade) and sade in yapisal_hedef
        else:
            bulundu = bool(
                re.search(rf"(?<!\w){re.escape(deger)}(?!\w)", giden_metin, re.IGNORECASE)
            )
        if bulundu:
            sizanlar.append(Sizinti(etiket, deger, kritik=tur in KRITIK_TURLER))

    return sizanlar


def _kural_bulgulari(metin: str, turler: frozenset[str]) -> list[Finding]:
    bulgular: list[Finding] = []
    for tur, desen, dogrulayici in KURALLAR:
        if tur not in turler:
            continue
        for m in desen.finditer(metin):
            ham = m.group(0)
            if dogrulayici is not None and not dogrulayici(ham):
                continue
            bulgular.append(
                Finding(
                    tur=tur,
                    metin=ham,
                    baslangic=m.start(),
                    bitis=m.end(),
                    kaynak="kural",
                )
            )
    return bulgular


def _cakismalari_coz(bulgular: list[Finding]) -> list[Finding]:
    """Ust uste binen bulgulardan birini secer.

    Oncelik: once kural katmani, sonra daha uzun eslesme, sonra yuksek skor.
    """
    sirali = sorted(
        bulgular,
        key=lambda b: (0 if b.kaynak == "kural" else 1, -b.uzunluk, -b.skor),
    )
    secilenler: list[Finding] = []
    for aday in sirali:
        cakisiyor = any(
            aday.baslangic < s.bitis and s.baslangic < aday.bitis
            for s in secilenler
        )
        if not cakisiyor:
            secilenler.append(aday)
    return sorted(secilenler, key=lambda b: b.baslangic)


def maskele(
    metin: str,
    ner_bulgulari: list[Finding] | None = None,
    turler: tuple[str, ...] = VARSAYILAN_TURLER,
) -> tuple[str, Vault, list[Finding]]:
    """Metni maskeler.

    NER bulgularini disaridan alir; boylece model yuklemeden (ve testlerde
    agirlik indirmeden) yalnizca kural katmani sinanabilir.

    Donen: (maskeli metin, kasa, kullanilan bulgular)
    """
    secili = frozenset(turler)
    bulgular = _kural_bulgulari(metin, secili)
    bulgular += [b for b in (ner_bulgulari or []) if b.tur in secili]
    bulgular = _cakismalari_coz(bulgular)

    kasa = Vault()
    parcalar: list[str] = []
    imlec = 0
    for b in bulgular:
        parcalar.append(metin[imlec : b.baslangic])
        parcalar.append(kasa.etiket_ver(b.tur, b.metin))
        imlec = b.bitis
    parcalar.append(metin[imlec:])

    return "".join(parcalar), kasa, bulgular
