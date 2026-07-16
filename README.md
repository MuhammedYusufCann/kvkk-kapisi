# KVKK Kapısı

**Kişisel veriyi buluta göndermeden çalışan belge asistanı.**

Hastane, hukuk bürosu gibi kurumlar belgeleri bir LLM'e özetletmek ister — ancak belgelerde TC kimlik numarası, isim, IBAN gibi KVKK kapsamındaki veriler bulunur. Bu proje araya bir gizlilik katmanı koyar: belge LLM'e gitmeden önce **yerelde çalışan** bir NER modeli ve kural katmanı kişisel veriyi maskeler. Bulut yalnızca `[KISI_1]`, `[TCKN_1]` gibi etiketleri görür. Cevap dönünce etiketler yerelde geri açılır. Gerçek değerler makineden hiç çıkmaz.

## Mimari

```
belge ──► [ NER modeli + kural katmanı ]  ──► maskeli belge ──► Gemini API
             (yerel, CPU, çevrimdışı)                              │
             kasa: etiket ↔ gerçek değer                           │
                    │                                              ▼
             cevap ◄─┴───────── etiketleri geri aç ◄──────── maskeli cevap
```

### Bileşenler

| Dosya | Görev |
|---|---|
| `kvkk/anonymizer.py` | Kural katmanı, maskeleme, kasa (Vault), sızıntı denetimi |
| `kvkk/ner.py` | HuggingFace NER modeli, BIO birleştirme, unvan ayıklama |
| `kvkk/llm.py` | Gemini API çağrısı — buluta çıkan **tek** yer |
| `app.py` | Gradio arayüzü |

### İki katmanlı tespit

**Kural katmanı** biçimi belli verileri yakalar: TCKN (resmî sağlama algoritması), IBAN (mod-97), kredi kartı (Luhn), telefon, e-posta, plaka. Doğrulama algoritmaları olmadan yanlış pozitif oranı kullanılamaz düzeyde olduğu için her veri türüne özel doğrulayıcı uygulanır.

**Model katmanı** (`savasy/bert-base-turkish-ner-cased`) kişi ve yer adlarını bağlamsal olarak tespit eder. Bu veriler regex ile yakalanamaz: isimlerin sabit bir biçimi yoktur ve aynı kelime farklı bağlamlarda farklı anlam taşır.

```
"Kurum: Ankara Numune Eğitim ve Araştırma Hastanesi"
   → KURUM  'Ankara Numune Eğitim ve Araştırma Hastanesi'  (0.95)

"Adres: Kızılay Mahallesi, Çankaya / Ankara"
   → YER    'Ankara'                                        (0.97)
```

Çakışma durumunda **kural katmanı kazanır**: kesin olan sonuç, olasılıksal olanın önüne geçer.

### Model ve veri seti

| Özellik | Değer |
|---|---|
| Model | `savasy/bert-base-turkish-ner-cased` |
| Temel model | `dbmdz/bert-base-turkish-cased` (BERTurk) |
| Eğitim veri seti | WikiANN (Vikipedi tabanlı, IOB2 etiketli, çok dilli NER) |
| Varlık tipleri | PER (Kişi), LOC (Yer), ORG (Kurum) |
| Çalışma ortamı | Tamamen yerel, CPU üzerinde |

### Sızıntı denetimi

Uygulama "veri gitmiyor" iddiasında bulunmak yerine **ölçer**: buluta giden isteği kasadaki her gerçek değer için tarayıp sonucu ekranda gösterir. Kimlik verileri (TCKN, IBAN, telefon, e-posta, isim) için eşleşme kritik sızıntı sayılır; yer/kurum adları bilgilendirme düzeyinde uyarı olarak raporlanır.

### Çözülen teknik sorunlar

1. **Span kırpması:** `transformers` kütüphanesinin `aggregation_strategy` katmanı, çok kelimeli isimlerde span'i kırparak kişisel veri sızıntısına yol açıyordu. BIO birleştirmesi özel olarak implemente edildi.

2. **Alt-kelime parçalanması:** Tokenizer'ın alt-kelime bölmesi (`Yılm` + `##az`) maskenin dışında kalan parçalar oluşturuyordu. Span'ler kelime sınırına yaslanarak bu sorun giderildi.

3. **Unvan karışıklığı:** Model `Dr.` gibi unvanları tek başına kişi adı olarak algılıyordu. Unvan ayıklama filtresi eklenerek isimler unvanlardan ayrıştırıldı.

## Kurulum

```bash
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
```

`.env` dosyasına [Google AI Studio](https://aistudio.google.com/apikey)'dan alınan API anahtarını ekleyin:

```
GEMINI_API_KEY=...
```

> API anahtarı olmadan da uygulama **çevrimdışı modda** açılır ve maskeleme zinciri çalışır.

## Çalıştırma

```bash
python app.py          # http://127.0.0.1:7860
python -m unittest     # 33 test
```

İlk çalıştırmada NER modeli (~440 MB) bir kez indirilir, sonrasında yerel cache'ten okunur.

## Örnek belgeler

`ornek_belgeler/` altındaki hasta raporu ve icra dilekçesi tamamen kurgudur. İçlerindeki TCKN, IBAN ve kart numaraları **üretilmiş sahte verilerdir** ancak sağlamaları geçerlidir — aksi halde kural katmanı onları doğru şekilde maskeleyemezdi.

## Lisans

MIT
