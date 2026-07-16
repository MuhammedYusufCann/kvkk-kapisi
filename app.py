"""KVKK Kapisi - kisisel veriyi buluta gondermeden calisan belge asistani.

Calistirma:  python app.py
"""

from __future__ import annotations

import pathlib
import re

import gradio as gr
from dotenv import load_dotenv

from kvkk.anonymizer import VARSAYILAN_TURLER, maskele, sizinti_denetle
from kvkk.llm import sor
from kvkk.ner import ner_bulgulari

load_dotenv()

ORNEK_DIZIN = pathlib.Path(__file__).parent / "ornek_belgeler"

TUM_TURLER = ["KISI", "YER", "KURUM", "TCKN", "IBAN", "TELEFON", "EPOSTA", "PLAKA", "KREDIKARTI"]

RENKLER = {
    "KISI": "red",
    "YER": "green",
    "KURUM": "blue",
    "TCKN": "purple",
    "IBAN": "orange",
    "TELEFON": "cyan",
    "EPOSTA": "yellow",
    "PLAKA": "pink",
    "KREDIKARTI": "brown",
}


def ornek_yukle(ad: str) -> str:
    return (ORNEK_DIZIN / ad).read_text(encoding="utf-8")


def _vurgu_parcalari(metin: str, bulgular) -> list[tuple[str, str | None]]:
    """HighlightedText'in bekledigi (parca, etiket) listesi."""
    parcalar: list[tuple[str, str | None]] = []
    imlec = 0
    for b in bulgular:
        if b.baslangic > imlec:
            parcalar.append((metin[imlec : b.baslangic], None))
        parcalar.append((b.metin, b.tur))
        imlec = b.bitis
    if imlec < len(metin):
        parcalar.append((metin[imlec:], None))
    return parcalar


def analiz_et(belge: str, secili_turler: list[str]):
    if not belge.strip():
        return [], "", [], None, "Once bir belge girin ya da ornek yukleyin."

    bulgular = ner_bulgulari(belge)
    maskeli, kasa, kullanilan = maskele(belge, bulgular, tuple(secili_turler))

    tablo = [
        [b.tur, b.metin, kasa.etiket_ver(b.tur, b.metin), b.kaynak, f"{b.skor:.2f}"]
        for b in kullanilan
    ]
    ozet = (
        f"{len(kullanilan)} kisisel veri bulundu, {len(kasa)} farkli degere "
        f"maskelendi. Kural katmani: {sum(1 for b in kullanilan if b.kaynak == 'kural')}, "
        f"model katmani: {sum(1 for b in kullanilan if b.kaynak == 'model')}."
    )
    return _vurgu_parcalari(belge, kullanilan), maskeli, tablo, kasa, ozet


ETIKET_DESENI = re.compile(r"\[[A-Z]+_\d+\]")


def soru_sor(maskeli: str, soru: str, kasa):
    if not maskeli.strip():
        return "", "", "", "Once belgeyi analiz edin."
    if not soru.strip():
        return "", "", "", "Bir soru yazin."

    # Maskeli metin duruyor ama kasa yoksa oturum dusmustur (sayfa yenilendi ya
    # da sunucu yeniden basladi). Bu sessizce gecilirse etiketli cevap 'acilmis'
    # diye gosterilir ve denetim '0 deger tarandi' deyip temiz raporlar - yani
    # uygulama yanlis oldugu halde dogru gorunur. Gorunur hataya ceviriyoruz.
    if not kasa and ETIKET_DESENI.search(maskeli):
        return (
            "",
            "",
            "",
            "### Oturum sifirlanmis\n"
            "Maskeli belge duruyor ama etiket-deger eslesmesini tutan kasa kayip; "
            "etiketler geri acilamaz. **'Analiz Et ve Maskele'ye tekrar basin.**",
        )

    cevap = sor(maskeli, soru)

    # Buluta giden istegi, kasadaki gercek degerlerin her biri icin tara.
    sizanlar = sizinti_denetle(cevap.giden_istek, kasa) if kasa else []
    kritikler = [s for s in sizanlar if s.kritik]
    uyarilar = [s for s in sizanlar if not s.kritik]

    if kritikler:
        rozet = "### KIRMIZI ALARM\n" + "\n".join(
            f"- `{s.etiket}` -> **{s.deger}** buluta giden istekte duruyor." for s in kritikler
        )
    else:
        rozet = (
            f"### Denetim temiz\n"
            f"Kasadaki **{len(kasa) if kasa else 0}** gercek degerin tamami tarandi; "
            f"hicbir kimlik verisi buluta giden istekte yok."
        )

    if uyarilar:
        rozet += "\n\nBilgi: " + ", ".join(f"`{s.deger}`" for s in uyarilar) + (
            " kelimesi istekte geciyor, ancak maskelenmemis bir kurum adinin parcasi "
            "oldugu icin kimlik ifsasi sayilmaz."
        )

    if cevap.saglayici == "cevrimdisi":
        rozet += "\n\n*CEVRIMDISI MOD*" + (f" — {cevap.hata}" if cevap.hata else "")

    acik_cevap = kasa.coz(cevap.metin) if kasa else cevap.metin
    return acik_cevap, cevap.metin, cevap.giden_istek, rozet


with gr.Blocks(title="KVKK Kapisi") as demo:
    gr.Markdown(
        """# KVKK Kapisi
### Kisisel veriyi buluta gondermeden calisan belge asistani

Belge LLM'e gitmeden once yerel bir HuggingFace Turkce NER modeli + kural katmani
kisisel veriyi maskeler. Bulut yalnizca `[KISI_1]`, `[TCKN_1]` gibi etiketleri gorur;
gercek degerler bu makineden hic cikmaz.
"""
    )

    kasa_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Belge")
            belge_kutusu = gr.Textbox(
                label="Belge metni",
                lines=18,
                placeholder="Belgeyi buraya yapistirin...",
            )
            with gr.Row():
                btn_hasta = gr.Button("Ornek: Hasta raporu", size="sm")
                btn_icra = gr.Button("Ornek: Icra dilekcesi", size="sm")

            tur_secimi = gr.CheckboxGroup(
                choices=TUM_TURLER,
                value=list(VARSAYILAN_TURLER),
                label="Maskelenecek veri turleri",
                info="KURUM varsayilan kapali: kurum adi tek basina kisisel veri degil, ayrica "
                "model tibbi metinde 'hipertansiyon' ve 'Kardiyoloji'yi de kurum saniyor - "
                "acilirsa hastanin tanisi maskelenir. (Sunumda dokunmayin.)",
            )
            btn_analiz = gr.Button("Analiz Et ve Maskele", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 2. Tespit")
            vurgulu = gr.HighlightedText(
                label="Bulunan kisisel veriler",
                color_map=RENKLER,
                show_legend=True,
            )
            ozet_kutusu = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 3. Buluta giden hali")
            maskeli_kutusu = gr.Textbox(
                label="Maskelenmis belge (LLM bunu goruyor)", lines=14
            )
        with gr.Column():
            gr.Markdown("### Bulgu dokumu")
            bulgu_tablosu = gr.Dataframe(
                headers=["Tur", "Gercek deger", "Etiket", "Kaynak", "Skor"],
                label="Ne, nasil yakalandi",
                wrap=True,
            )

    gr.Markdown("### 4. Soru sor")
    with gr.Row():
        soru_kutusu = gr.Textbox(
            label="Soru",
            placeholder="Hastaya hangi tani konmus ve hangi ilac baslanmis?",
            scale=4,
        )
        btn_sor = gr.Button("Sor", variant="primary", scale=1)

    denetim_kutusu = gr.Markdown()

    with gr.Row():
        cevap_acik = gr.Textbox(
            label="Size gosterilen cevap (etiketler geri acildi)", lines=6
        )
        cevap_maskeli = gr.Textbox(label="LLM'in urettigi ham cevap", lines=6)

    with gr.Accordion("Buluta giden ham istek - kanit", open=False):
        gr.Markdown(
            "Gemini'ye tam olarak asagidaki metin gitti. Icinde tek bir gercek "
            "isim, TCKN ya da IBAN yok."
        )
        istek_kutusu = gr.Textbox(label="HTTP govdesi", lines=20)

    btn_hasta.click(lambda: ornek_yukle("hasta_raporu.txt"), outputs=belge_kutusu)
    btn_icra.click(lambda: ornek_yukle("icra_dilekce.txt"), outputs=belge_kutusu)

    btn_analiz.click(
        analiz_et,
        inputs=[belge_kutusu, tur_secimi],
        outputs=[vurgulu, maskeli_kutusu, bulgu_tablosu, kasa_state, ozet_kutusu],
    )
    btn_sor.click(
        soru_sor,
        inputs=[maskeli_kutusu, soru_kutusu, kasa_state],
        outputs=[cevap_acik, cevap_maskeli, istek_kutusu, denetim_kutusu],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
