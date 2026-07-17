"""KVKK Kapisi - kisisel veriyi buluta gondermeden calisan belge asistani.

Calistirma:  python app.py
"""

from __future__ import annotations

import html
import os
import pathlib
import re

import gradio as gr
from dotenv import load_dotenv
from gradio.themes.utils import fonts

from kvkk.anonymizer import VARSAYILAN_TURLER, maskele, sizinti_denetle
from kvkk.llm import MODEL as LLM_MODEL
from kvkk.llm import sor
from kvkk.ner import ner_bulgulari

load_dotenv()

ORNEK_DIZIN = pathlib.Path(__file__).parent / "ornek_belgeler"

# Tur kodlari ASCII kalir: maskeli metin LLM'e gidip geri donerken Turkce
# karakterin bozulma riski olmasin (bkz. anonymizer.NER_TUR_ESLESME). Ekranda
# gorunen ad ayri tutuluyor; kod ile etiket arasindaki bag bozulmadan arayuz
# duzgun Turkce olabiliyor.
TUR_ADI = {
    "KISI": "Kişi",
    "YER": "Yer",
    "KURUM": "Kurum",
    "TCKN": "T.C. Kimlik No",
    "IBAN": "IBAN",
    "TELEFON": "Telefon",
    "EPOSTA": "E-posta",
    "PLAKA": "Plaka",
    "KREDIKARTI": "Kredi kartı",
}

KAYNAK_ADI = {"kural": "Kural katmanı", "model": "NER modeli"}

# Renkler anonymizer.KRITIK_TURLER ayrimini izler: sizmasi dogrudan kimlik
# ifsasi olan turler sicak/doygun, yalnizca baglam tasiyan YER/KURUM soguk ve
# soluk. Boylece vurgulu metne bakan biri neyin neden onemli oldugunu
# lejanti okumadan goruyor.
RENKLER = {
    "Kişi": "#fda4af",
    "T.C. Kimlik No": "#fca5a5",
    "IBAN": "#fdba74",
    "Telefon": "#fcd34d",
    "E-posta": "#fde68a",
    "Kredi kartı": "#f9a8d4",
    "Plaka": "#d8b4fe",
    "Yer": "#bae6fd",
    "Kurum": "#c7d2fe",
}

GEMINI_ANAHTARI_VAR = bool(os.environ.get("GEMINI_API_KEY", "").strip())

# IBM Plex, gradio'nun kendi paketinden yerel olarak servis ediliyor
# (stylesheet url'i None). GoogleFont sunum sirasinda aga cikardi; bu projenin
# tasarim kisiti ag riski almamak (bkz. cevrimdisi yedek), yazi tipi de buna
# dahil. Yedekler de Font nesnesi: gradio yerlesik temalarla karsilastirirken
# listeyi eleman eleman geziyor ve duz string gorunce .name arayip patliyor.
TEMA = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=[fonts.LocalFont("IBMPlexSans"), fonts.Font("Segoe UI"), fonts.Font("system-ui")],
    font_mono=[fonts.LocalFont("IBMPlexMono"), fonts.Font("Consolas"), fonts.Font("monospace")],
)

CSS = """
:root {
  --kvkk-yerel: #059669;
  --kvkk-yerel-bg: #ecfdf5;
  --kvkk-bulut: #0284c7;
  --kvkk-bulut-bg: #eff6ff;
  --kvkk-ink: #0f172a;
}
.gradio-container { max-width: 1500px !important; }
footer { display: none !important; }

/* ---------- Hero ---------- */
.kvkk-hero {
  background: linear-gradient(135deg, #0f172a 0%, #134e4a 55%, #0c4a6e 100%);
  border-radius: 18px; padding: 1.6rem 1.8rem; margin-bottom: 1.1rem;
  color: #e2e8f0;
}
.kvkk-hero__ust { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; flex-wrap: wrap; }
.kvkk-hero__baslik { font-size: 1.9rem; font-weight: 800; color: #fff; letter-spacing: -.02em; line-height: 1.15; }
.kvkk-hero__alt { font-size: .95rem; color: #94a3b8; margin-top: .25rem; max-width: 60ch; }
.kvkk-hero__alt b { color: #5eead4; font-weight: 600; }

/* ---------- Akis seridi ---------- */
.kvkk-flow { display: flex; align-items: stretch; gap: .5rem; margin-top: 1.3rem; flex-wrap: wrap; }
.kvkk-flow__adim {
  flex: 1 1 0; min-width: 150px;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12);
  border-radius: 11px; padding: .6rem .75rem;
}
.kvkk-flow__adim--bulut { background: rgba(2,132,199,.16); border-color: rgba(56,189,248,.45); }
.kvkk-flow__et { font-size: .62rem; font-weight: 700; letter-spacing: .09em; opacity: .75; }
.kvkk-flow__adim--yerel .kvkk-flow__et { color: #5eead4; }
.kvkk-flow__adim--bulut .kvkk-flow__et { color: #7dd3fc; }
.kvkk-flow__ad { font-size: .84rem; font-weight: 600; color: #f1f5f9; margin-top: .15rem; }
.kvkk-flow__ok { display: flex; align-items: center; color: #475569; font-size: 1.1rem; }
/* Serit sardiginda ok ikinci satirin basina dusuyor ve akisi yanlis okutuyor;
   dar ekranda oklari birakip adimlari izgaraya seriyoruz. */
@media (max-width: 900px) {
  .kvkk-flow__ok { display: none; }
  .kvkk-flow__adim { flex: 1 1 44%; }
}

/* ---------- Durum rozeti ---------- */
.kvkk-rozet {
  display: inline-flex; align-items: center; gap: .5rem; white-space: nowrap;
  padding: .4rem .8rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); color: #e2e8f0;
}
.kvkk-rozet__nokta { width: .5rem; height: .5rem; border-radius: 50%; }
.kvkk-rozet--acik .kvkk-rozet__nokta { background: #34d399; box-shadow: 0 0 0 3px rgba(52,211,153,.22); }
.kvkk-rozet--kapali .kvkk-rozet__nokta { background: #fbbf24; box-shadow: 0 0 0 3px rgba(251,191,36,.22); }

/* ---------- Adim basligi + bolge rozeti ---------- */
.kvkk-adim { display: flex; align-items: center; gap: .6rem; margin: .1rem 0 .55rem; flex-wrap: wrap; }
.kvkk-adim__no {
  width: 1.6rem; height: 1.6rem; border-radius: 8px; flex: none;
  background: var(--kvkk-ink); color: #fff;
  font-size: .8rem; font-weight: 700; display: flex; align-items: center; justify-content: center;
}
.kvkk-adim__ad { font-size: 1.02rem; font-weight: 700; color: var(--kvkk-ink); }
.kvkk-bolge {
  margin-left: auto; display: inline-flex; align-items: center; gap: .3rem;
  font-size: .63rem; font-weight: 800; letter-spacing: .07em;
  padding: .22rem .55rem; border-radius: 999px; border: 1px solid;
}
.kvkk-bolge--yerel { color: var(--kvkk-yerel); background: var(--kvkk-yerel-bg); border-color: #a7f3d0; }
.kvkk-bolge--bulut { color: var(--kvkk-bulut); background: var(--kvkk-bulut-bg); border-color: #bae6fd; }

/* Panelin hangi bolgeye ait oldugunu govdeden de belli et: sol kenar serit. */
.kvkk-panel-yerel .block { border-left: 3px solid #6ee7b7 !important; }
.kvkk-panel-bulut .block { border-left: 3px solid #7dd3fc !important; }

/* ---------- Stat tile ---------- */
.kvkk-tiles { display: flex; gap: .5rem; flex-wrap: wrap; margin: 0 0 .6rem; }
.kvkk-tile {
  flex: 1 1 0; min-width: 92px; border-radius: 12px; padding: .6rem .7rem;
  background: #fff; border: 1px solid #e2e8f0;
}
.kvkk-tile__num { display: block; font-size: 1.55rem; font-weight: 800; line-height: 1.05; color: var(--kvkk-ink); }
.kvkk-tile__lbl { display: block; font-size: .68rem; opacity: .62; margin-top: .2rem; line-height: 1.25; }
.kvkk-tile--vurgu { background: var(--kvkk-yerel-bg); border-color: #6ee7b7; }
.kvkk-tile--vurgu .kvkk-tile__num { color: #047857; }

.kvkk-hint { font-size: .85rem; opacity: .6; padding: .6rem .1rem; }

/* ---------- Denetim banner'i ---------- */
.kvkk-verdict {
  display: flex; gap: .9rem; align-items: flex-start;
  border: 1px solid; border-left-width: 5px; border-radius: 13px;
  padding: .95rem 1.1rem; margin: .35rem 0;
}
.kvkk-verdict__icon {
  width: 2rem; height: 2rem; flex: none; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.05rem; font-weight: 800; color: #fff;
}
.kvkk-verdict__title { font-weight: 800; font-size: 1.1rem; letter-spacing: .01em; }
.kvkk-verdict__sub { font-size: .87rem; margin-top: .2rem; }
.kvkk-verdict__note { font-size: .78rem; margin-top: .5rem; opacity: .85; }
.kvkk-verdict__list { margin: .35rem 0 0; padding-left: 1.1rem; font-size: .87rem; }
.kvkk-verdict code { background: rgba(0,0,0,.07); padding: .05rem .3rem; border-radius: 4px; }

.kvkk-verdict--ok    { background: #ecfdf5; border-color: #6ee7b7; border-left-color: #10b981; color: #065f46; }
.kvkk-verdict--ok    .kvkk-verdict__icon { background: #10b981; }
.kvkk-verdict--alarm { background: #fef2f2; border-color: #fca5a5; border-left-color: #ef4444; color: #7f1d1d; }
.kvkk-verdict--alarm .kvkk-verdict__icon { background: #ef4444; }
.kvkk-verdict--uyari { background: #fffbeb; border-color: #fcd34d; border-left-color: #f59e0b; color: #78350f; }
.kvkk-verdict--uyari .kvkk-verdict__icon { background: #f59e0b; }

/* ---------- Gradio duzeltmeleri ---------- */
/* Vurgulu metin belgenin tamamini tek parca cizdigi icin sinirsiz uzuyor ve
   altindaki kolona yuzlerce piksel olu bosluk biniyordu. */
#vurgulu .textfield { max-height: 500px; overflow-y: auto; line-height: 1.75; }
#vurgulu .textfield, #maskeli textarea { font-size: .82rem; }
/* Maskeli belge ve ham istek birer veri kaniti; mono yazi hem hizalar hem
   etiketleri metinden ayirir. */
#maskeli textarea, #istek textarea { font-family: var(--font-mono); }
.kvkk-sorular button { font-size: .78rem !important; }
"""


def ornek_yukle(ad: str) -> str:
    return (ORNEK_DIZIN / ad).read_text(encoding="utf-8")


def _vurgu_parcalari(metin: str, bulgular) -> list[tuple[str, str | None]]:
    """HighlightedText'in bekledigi (parca, etiket) listesi."""
    parcalar: list[tuple[str, str | None]] = []
    imlec = 0
    for b in bulgular:
        if b.baslangic > imlec:
            parcalar.append((metin[imlec : b.baslangic], None))
        parcalar.append((b.metin, TUR_ADI.get(b.tur, b.tur)))
        imlec = b.bitis
    if imlec < len(metin):
        parcalar.append((metin[imlec:], None))
    return parcalar


def _ipucu(mesaj: str) -> str:
    return f'<div class="kvkk-hint">{html.escape(mesaj)}</div>'


def _tile(sayi: int, etiket: str, vurgu: bool = False) -> str:
    sinif = "kvkk-tile kvkk-tile--vurgu" if vurgu else "kvkk-tile"
    return (
        f'<div class="{sinif}"><span class="kvkk-tile__num">{sayi}</span>'
        f'<span class="kvkk-tile__lbl">{html.escape(etiket)}</span></div>'
    )


def _adim(no: str, ad: str, bolge: str | None = None) -> str:
    """Adim basligi. `no` bossa numara kutusu hic cizilmez (yan panellerin
    kendi adim numarasi yok); bolge verilirse 'yerel/bulut' rozeti eklenir.
    """
    rozet = ""
    if bolge == "yerel":
        rozet = '<span class="kvkk-bolge kvkk-bolge--yerel">● BU MAKİNEDE</span>'
    elif bolge == "bulut":
        rozet = '<span class="kvkk-bolge kvkk-bolge--bulut">▲ BULUTA GİDİYOR</span>'
    kutu = f'<span class="kvkk-adim__no">{html.escape(no)}</span>' if no else ""
    return (
        f'<div class="kvkk-adim">{kutu}'
        f'<span class="kvkk-adim__ad">{html.escape(ad)}</span>{rozet}</div>'
    )


def analiz_et(belge: str, secili_turler: list[str]):
    if not belge.strip():
        return [], "", [], None, _ipucu("Önce bir belge girin ya da örnek yükleyin.")

    bulgular = ner_bulgulari(belge)
    maskeli, kasa, kullanilan = maskele(belge, bulgular, tuple(secili_turler))

    tablo = [
        [
            TUR_ADI.get(b.tur, b.tur),
            b.metin,
            kasa.etiket_ver(b.tur, b.metin),
            KAYNAK_ADI.get(b.kaynak, b.kaynak),
            f"{b.skor:.2f}",
        ]
        for b in kullanilan
    ]
    kural = sum(1 for b in kullanilan if b.kaynak == "kural")
    model = sum(1 for b in kullanilan if b.kaynak == "model")
    ozet = (
        '<div class="kvkk-tiles">'
        + _tile(len(kullanilan), "kişisel veri bulundu", vurgu=True)
        + _tile(len(kasa), "farklı değere maskelendi")
        + _tile(kural, "kural katmanı")
        + _tile(model, "NER modeli")
        + "</div>"
    )
    return _vurgu_parcalari(belge, kullanilan), maskeli, tablo, kasa, ozet


ETIKET_DESENI = re.compile(r"\[[A-Z]+_\d+\]")

# Dusen oturum uyarisinin basligi sabit: testler bu guard'in calistigini buradan
# dogruluyor, metnin kendisi degisince test kirilmasin.
OTURUM_HATASI_BASLIGI = "OTURUM SIFIRLANMIŞ"


def soru_sor(maskeli: str, soru: str, kasa):
    if not maskeli.strip():
        return "", "", "", _ipucu("Önce belgeyi analiz edin.")
    if not soru.strip():
        return "", "", "", _ipucu("Bir soru yazın.")

    # Maskeli metin duruyor ama kasa yoksa oturum dusmustur (sayfa yenilendi ya
    # da sunucu yeniden basladi). Bu sessizce gecilirse etiketli cevap 'acilmis'
    # diye gosterilir ve denetim '0 deger tarandi' deyip temiz raporlar - yani
    # uygulama yanlis oldugu halde dogru gorunur. Gorunur hataya ceviriyoruz.
    if not kasa and ETIKET_DESENI.search(maskeli):
        return (
            "",
            "",
            "",
            '<div class="kvkk-verdict kvkk-verdict--uyari">'
            '<div class="kvkk-verdict__icon">↻</div><div>'
            f'<div class="kvkk-verdict__title">{OTURUM_HATASI_BASLIGI}</div>'
            '<div class="kvkk-verdict__sub">Maskeli belge duruyor ama etiket–değer '
            "eşleşmesini tutan kasa kayıp; etiketler geri açılamaz. "
            "<b>“Analiz Et ve Maskele”ye tekrar basın.</b></div></div></div>",
        )

    cevap = sor(maskeli, soru)

    # Buluta giden istegi, kasadaki gercek degerlerin her biri icin tara.
    sizanlar = sizinti_denetle(cevap.giden_istek, kasa) if kasa else []
    kritikler = [s for s in sizanlar if s.kritik]
    uyarilar = [s for s in sizanlar if not s.kritik]

    if kritikler:
        maddeler = "".join(
            f"<li><code>{html.escape(s.etiket)}</code> → "
            f"<b>{html.escape(s.deger)}</b> buluta giden istekte duruyor.</li>"
            for s in kritikler
        )
        rozet = (
            '<div class="kvkk-verdict kvkk-verdict--alarm">'
            '<div class="kvkk-verdict__icon">!</div><div>'
            '<div class="kvkk-verdict__title">KIRMIZI ALARM</div>'
            f'<ul class="kvkk-verdict__list">{maddeler}</ul>'
        )
    else:
        rozet = (
            '<div class="kvkk-verdict kvkk-verdict--ok">'
            '<div class="kvkk-verdict__icon">✓</div><div>'
            '<div class="kvkk-verdict__title">DENETİM TEMİZ</div>'
            '<div class="kvkk-verdict__sub">Kasadaki '
            f"<b>{len(kasa) if kasa else 0}</b> gerçek değerin tamamı buluta giden "
            "istekte tek tek arandı; hiçbiri bulunamadı.</div>"
        )

    if uyarilar:
        kelimeler = ", ".join(f"<code>{html.escape(s.deger)}</code>" for s in uyarilar)
        rozet += (
            f'<div class="kvkk-verdict__note">Bilgi: {kelimeler} istekte geçiyor, '
            "ancak maskelenmemiş bir kurum adının parçası olduğu için kimlik "
            "ifşası sayılmaz.</div>"
        )

    if cevap.saglayici == "cevrimdisi":
        ek = f" — {html.escape(cevap.hata)}" if cevap.hata else ""
        rozet += f'<div class="kvkk-verdict__note"><b>ÇEVRİMDIŞI MOD</b>{ek}</div>'

    rozet += "</div></div>"

    acik_cevap = kasa.coz(cevap.metin) if kasa else cevap.metin
    return acik_cevap, cevap.metin, cevap.giden_istek, rozet


if GEMINI_ANAHTARI_VAR:
    DURUM_ROZETI = (
        '<div class="kvkk-rozet kvkk-rozet--acik"><span class="kvkk-rozet__nokta">'
        f"</span>{html.escape(LLM_MODEL)} bağlı</div>"
    )
else:
    DURUM_ROZETI = (
        '<div class="kvkk-rozet kvkk-rozet--kapali"><span class="kvkk-rozet__nokta">'
        "</span>Çevrimdışı mod — anahtar yok</div>"
    )

HERO = f"""
<div class="kvkk-hero">
  <div class="kvkk-hero__ust">
    <div>
      <div class="kvkk-hero__baslik">KVKK Kapısı</div>
      <div class="kvkk-hero__alt">Kişisel veriyi buluta göndermeden çalışan belge asistanı.
      Belge LLM'e gitmeden önce yerel NER modeli + kural katmanı kişisel veriyi maskeler;
      bulut yalnızca etiketleri görür, <b>gerçek değerler bu makineden hiç çıkmaz</b>.</div>
    </div>
    {DURUM_ROZETI}
  </div>
  <div class="kvkk-flow">
    <div class="kvkk-flow__adim kvkk-flow__adim--yerel">
      <div class="kvkk-flow__et">YEREL</div><div class="kvkk-flow__ad">Belge</div></div>
    <div class="kvkk-flow__ok">→</div>
    <div class="kvkk-flow__adim kvkk-flow__adim--yerel">
      <div class="kvkk-flow__et">YEREL · CPU</div><div class="kvkk-flow__ad">NER modeli + kural katmanı</div></div>
    <div class="kvkk-flow__ok">→</div>
    <div class="kvkk-flow__adim kvkk-flow__adim--bulut">
      <div class="kvkk-flow__et">BULUT</div><div class="kvkk-flow__ad">Gemini — sadece [KISI_1]</div></div>
    <div class="kvkk-flow__ok">→</div>
    <div class="kvkk-flow__adim kvkk-flow__adim--yerel">
      <div class="kvkk-flow__et">YEREL</div><div class="kvkk-flow__ad">Etiketleri geri aç</div></div>
  </div>
</div>
"""

HAZIR_SORULAR = [
    "Hastaya hangi tanı konmuş ve hangi ilaç başlanmış?",
    "Alacak tutarı ne kadar ve kimden talep ediliyor?",
    "Belgedeki kişilerin iletişim bilgilerini listele.",
]

with gr.Blocks(title="KVKK Kapısı") as demo:
    gr.HTML(HERO)

    kasa_state = gr.State(None)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML(_adim("1", "Belge", "yerel"))
            with gr.Column(elem_classes="kvkk-panel-yerel"):
                belge_kutusu = gr.Textbox(
                    label="Belge metni",
                    lines=16,
                    placeholder="Belgeyi buraya yapıştırın...",
                )
            with gr.Row():
                btn_hasta = gr.Button("Örnek: Hasta raporu", size="sm")
                btn_icra = gr.Button("Örnek: İcra dilekçesi", size="sm")

            tur_secimi = gr.CheckboxGroup(
                choices=[(TUR_ADI[t], t) for t in TUR_ADI],
                value=list(VARSAYILAN_TURLER),
                label="Maskelenecek veri türleri",
                info="Kurum varsayılan kapalı: kurum adı tek başına kişisel veri değil, "
                "ayrıca model tıbbi metinde 'hipertansiyon' ve 'Kardiyoloji'yi de kurum "
                "sanıyor — açılırsa hastanın tanısı maskelenir. (Sunumda dokunmayın.)",
            )
            btn_analiz = gr.Button("Analiz Et ve Maskele", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.HTML(_adim("2", "Tespit", "yerel"))
            ozet_kutusu = gr.HTML()
            with gr.Column(elem_classes="kvkk-panel-yerel"):
                vurgulu = gr.HighlightedText(
                    label="Bulunan kişisel veriler",
                    color_map=RENKLER,
                    show_legend=True,
                    elem_id="vurgulu",
                )

    with gr.Row(equal_height=False):
        with gr.Column():
            gr.HTML(_adim("3", "Buluta giden hali", "bulut"))
            with gr.Column(elem_classes="kvkk-panel-bulut"):
                maskeli_kutusu = gr.Textbox(
                    label="Maskelenmiş belge — LLM yalnızca bunu görüyor",
                    lines=14,
                    elem_id="maskeli",
                )
        with gr.Column():
            gr.HTML(_adim("", "Bulgu dökümü", "yerel"))
            with gr.Column(elem_classes="kvkk-panel-yerel"):
                bulgu_tablosu = gr.Dataframe(
                    headers=["Tür", "Gerçek değer", "Etiket", "Kaynak", "Skor"],
                    label="Ne, nasıl yakalandı",
                    wrap=True,
                )

    gr.HTML(_adim("4", "Soru sor"))
    with gr.Row():
        soru_kutusu = gr.Textbox(
            label="Soru",
            placeholder="Hastaya hangi tanı konmuş ve hangi ilaç başlanmış?",
            scale=4,
        )
        btn_sor = gr.Button("Sor", variant="primary", scale=1, size="lg")

    with gr.Row(elem_classes="kvkk-sorular"):
        hazir_butonlar = [gr.Button(s, size="sm") for s in HAZIR_SORULAR]

    denetim_kutusu = gr.HTML()

    with gr.Row(equal_height=False):
        with gr.Column(elem_classes="kvkk-panel-yerel"):
            cevap_acik = gr.Textbox(
                label="Size gösterilen cevap — etiketler yerelde geri açıldı", lines=7
            )
        with gr.Column(elem_classes="kvkk-panel-bulut"):
            cevap_maskeli = gr.Textbox(
                label="LLM'in ürettiği ham cevap — etiketlerle", lines=7
            )

    with gr.Accordion("Buluta giden ham istek — kanıt", open=False):
        gr.Markdown(
            "Gemini'ye tam olarak aşağıdaki metin gitti. İçinde tek bir gerçek "
            "isim, TCKN ya da IBAN yok."
        )
        istek_kutusu = gr.Textbox(label="HTTP gövdesi", lines=20, elem_id="istek")

    btn_hasta.click(lambda: ornek_yukle("hasta_raporu.txt"), outputs=belge_kutusu)
    btn_icra.click(lambda: ornek_yukle("icra_dilekce.txt"), outputs=belge_kutusu)

    for btn, metin in zip(hazir_butonlar, HAZIR_SORULAR):
        btn.click(lambda m=metin: m, outputs=soru_kutusu)

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
    demo.launch(theme=TEMA, css=CSS)
