"""Maskeli metin uzerinden Gemini'ye soru soran katman.

Bu dosya buluta cikan tek yerdir. Buraya yalnizca maskelenmis metin girer;
gercek degerleri tutan Vault hicbir zaman bu katmana gecmez. `LLMCevap.giden_istek`
alani, sunumda 'buluta tam olarak ne gitti' sorusunu kanitlamak icin duruyor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

MODEL = "gemini-2.5-flash"

SISTEM_TALIMATI = """Sen bir Turkce kurumsal belge asistanisin.

Sana verilen belgede [KISI_1], [TCKN_1], [IBAN_1] gibi koseli parantezli
etiketler gorursun. Bunlar KVKK geregi maskelenmis kisisel verilerdir.

Kurallar:
- Bu etiketlerin ardindaki gercek degerleri BILMIYORSUN. Tahmin etme, uydurma.
- Bir kisiye ya da veriye atif yapman gerekiyorsa etiketi AYNEN yaz: [KISI_1]
- Cevabini yalnizca belgedeki bilgiye dayandir. Belgede olmayan bir sey
  soruluyorsa "Belgede bu bilgi yok." de.
- Kisa ve net yaz."""

ISTEK_SABLONU = """Belge:
\"\"\"
{belge}
\"\"\"

Soru: {soru}"""


@dataclass
class LLMCevap:
    metin: str
    giden_istek: str
    saglayici: str  # "gemini" | "cevrimdisi"
    hata: str | None = None


def _istemci():
    anahtar = os.environ.get("GEMINI_API_KEY", "").strip()
    if not anahtar:
        return None
    from google import genai

    return genai.Client(api_key=anahtar)


def _cevrimdisi_cevap(maskeli_belge: str, soru: str) -> str:
    """API anahtari yoksa / internet giderse sunumu ayakta tutan yedek.

    Kasitli olarak 'akilli' degil: LLM'in yerine gecmeye calismaz, yalnizca
    maskeleme zincirinin geri kalaninin calistigini gosterir.
    """
    etiketler = sorted({e for e in _etiketleri_bul(maskeli_belge)})
    return (
        "[CEVRIMDISI MOD - GEMINI_API_KEY tanimli degil]\n\n"
        f"Soru alindi: {soru}\n"
        f"Belgede maskelenmis {len(etiketler)} farkli kisisel veri var: "
        f"{', '.join(etiketler) if etiketler else '-'}\n\n"
        "Gercek cevap icin .env dosyasina GEMINI_API_KEY ekleyin."
    )


def _etiketleri_bul(metin: str) -> list[str]:
    import re

    return re.findall(r"\[[A-Z]+_\d+\]", metin)


def sor(maskeli_belge: str, soru: str) -> LLMCevap:
    """Maskeli belge uzerinden soruyu yanitlar.

    Girdi olarak yalnizca maskeli metni kabul eder; cagiran taraf buraya ham
    belgeyi gecirmemekle yukumludur.
    """
    istek = ISTEK_SABLONU.format(belge=maskeli_belge, soru=soru)
    tam_istek = f"--- SISTEM TALIMATI ---\n{SISTEM_TALIMATI}\n\n--- ISTEK ---\n{istek}"

    istemci = _istemci()
    if istemci is None:
        return LLMCevap(
            metin=_cevrimdisi_cevap(maskeli_belge, soru),
            giden_istek=tam_istek,
            saglayici="cevrimdisi",
        )

    try:
        from google.genai import types

        yanit = istemci.models.generate_content(
            model=MODEL,
            contents=istek,
            config=types.GenerateContentConfig(
                system_instruction=SISTEM_TALIMATI,
                temperature=0.2,
            ),
        )
        return LLMCevap(
            metin=(yanit.text or "").strip(),
            giden_istek=tam_istek,
            saglayici="gemini",
        )
    except Exception as e:  # sunum sirasinda ag hatasi demoyu dusurmesin
        return LLMCevap(
            metin=_cevrimdisi_cevap(maskeli_belge, soru),
            giden_istek=tam_istek,
            saglayici="cevrimdisi",
            hata=f"{type(e).__name__}: {e}",
        )
