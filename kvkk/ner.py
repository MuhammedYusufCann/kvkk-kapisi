"""HuggingFace Turkce NER modelini yukleyip Finding uretir.

Model tamamen yerelde, CPU'da calisir. Bu dosya disari hicbir istek atmaz;
projenin ana iddiasi belgenin makineden hic cikmamasi. (Model agirliklari ilk
calistirmada bir kez indirilir, sonrasinda yerel cache'ten okunur.)

NOT: transformers'in kendi `aggregation_strategy` katmani kullanilmiyor.
Olculdu: ham model 'Hasta Ahmet Yilmaz' icin B-PER/I-PER'i 0.99 skorla dogru
etiketliyor, ama pipeline birlestirmesi span'i 'Yilmaz'a kirpip 'Ahmet'i
dusuruyor. Maskeleyicide kirpilmis span, sizmis kisisel veri demek; bu yuzden
BIO birlestirmesini asagida kendimiz yapiyoruz.
"""

from __future__ import annotations

import re

from .anonymizer import MODEL_ADI, NER_TUR_ESLESME, Finding

_boru_hatti = None

# Esik bilerek dusuk: bu bir gizlilik araci, kacan bir isim yanlis maskelenmis
# bir kelimeden cok daha pahali. Yanlis pozitifleri esigi yukselterek degil,
# asagidaki unvan filtresi gibi hedefli kurallarla eliyoruz.
_SKOR_ESIGI = 0.5

# Model 'Dr.'yi tek basina kisi adi sayabiliyor (olculdu: skor 0.50) ve
# 'Dr. Mehmet Ozturk'te unvani ismin icine katiyor. Ikisi de maskeyi bozuyor:
# biri '[KISI_3] [KISI_4]' gibi sacma bir cikti, digeri ayni kisinin unvanli ve
# unvansiz gecislerinin farkli etiket almasi demek.
UNVANLAR = frozenset(
    {
        "dr", "doc", "doç", "prof", "op", "uzm", "av", "sayin", "sayın",
        "bay", "bayan", "muh", "müh", "hak", "hakim", "hâkim", "savci", "savcı",
    }
)

_ONEK_DESENI = re.compile(r"^([^\W\d_]+)\.?\s+", re.UNICODE)


def _unvan_ayikla(metin: str, bas: int, bit: int) -> tuple[int, int] | None:
    """Kisi span'inin basindaki unvanlari kirpar; span sadece unvansa eler."""
    while True:
        parca = metin[bas:bit]
        m = _ONEK_DESENI.match(parca)
        if m is None or m.group(1).casefold() not in UNVANLAR:
            break
        bas += m.end()

    kalan = metin[bas:bit].strip()
    if not kalan or kalan.rstrip(".").casefold() in UNVANLAR:
        return None
    return bas, bas + len(kalan)


def modeli_yukle():
    """Modeli tembel yukler; ilk cagri agirliklari indirebilir."""
    global _boru_hatti
    if _boru_hatti is None:
        from transformers import pipeline

        _boru_hatti = pipeline(
            "token-classification",
            model=MODEL_ADI,
            aggregation_strategy="none",  # birlestirmeyi kendimiz yapiyoruz
        )
    return _boru_hatti


def _kelime_sinirina_yasla(metin: str, bas: int, bit: int) -> tuple[int, int]:
    """Span'i icinde bulundugu kelimenin tamamini kapsayacak sekilde genisletir.

    Tokenizer 'Yilmaz'i 'Yilm' + '##az' diye bolup ikinci parcayi O etiketleyince
    span yarim kaliyor ve maskeli metinde '[KISI_1]az' gibi bir sizinti olusuyor.
    Yalnizca harf/rakam uzerinden genisliyoruz: 'Ankara'da' -> 'Ankara' kalir,
    kesme isaretiyle gelen Turkce cekim eki maskenin disinda birakilir.
    """
    while bas > 0 and metin[bas - 1].isalnum():
        bas -= 1
    while bit < len(metin) and metin[bit].isalnum():
        bit += 1
    return bas, bit


def _bio_birlestir(metin: str, tokenlar: list[dict]) -> list[Finding]:
    bulgular: list[Finding] = []
    acik: dict | None = None

    def kapat() -> None:
        nonlocal acik
        if acik is None:
            return
        bas, bit = _kelime_sinirina_yasla(metin, acik["bas"], acik["bit"])
        if acik["tur"] == "KISI":
            kirpilmis = _unvan_ayikla(metin, bas, bit)
            if kirpilmis is None:
                acik = None
                return
            bas, bit = kirpilmis

        parca = metin[bas:bit].strip()
        if parca:
            bulgular.append(
                Finding(
                    tur=acik["tur"],
                    metin=parca,
                    baslangic=bas,
                    bitis=bas + len(parca),
                    kaynak="model",
                    skor=acik["skor"],
                )
            )
        acik = None

    for t in tokenlar:
        etiket = t["entity"]
        if etiket == "O":
            kapat()
            continue

        onek, _, ham_tur = etiket.partition("-")
        tur = NER_TUR_ESLESME.get(ham_tur or onek)
        if tur is None:
            kapat()
            continue

        bas, bit, skor = int(t["start"]), int(t["end"]), float(t["score"])

        # Ayni varligin devami mi? I- oneki ayni turu suruyorsa, ya da token
        # bir onceki token'a bosluksuz yapisiksa (alt-kelime parcasi).
        devam = (
            acik is not None
            and acik["tur"] == tur
            and (onek == "I" or bas == acik["bit"])
        )
        if devam:
            acik["bit"] = bit
            acik["skor"] = min(acik["skor"], skor)  # en zayif halka belirleyici
        else:
            kapat()
            acik = {"tur": tur, "bas": bas, "bit": bit, "skor": skor}

    kapat()
    return [b for b in bulgular if b.skor >= _SKOR_ESIGI]


def ner_bulgulari(metin: str) -> list[Finding]:
    if not metin.strip():
        return []
    return _bio_birlestir(metin, list(modeli_yukle()(metin)))
