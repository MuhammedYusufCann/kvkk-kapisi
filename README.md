# KVKK Kapısı

**Kişisel veriyi buluta göndermeden çalışan belge asistanı.**

Bir hastane ya da hukuk bürosu, elindeki belgeleri bir LLM'e özetletmek ister.
Ama belgede TC kimlik numarası, isim, IBAN var — KVKK gereği buluta gönderilemez.

Bu proje araya bir kapı koyar: belge LLM'e gitmeden önce **yerelde çalışan** bir
HuggingFace Türkçe NER modeli + kural katmanı kişisel veriyi maskeler. Bulut
yalnızca `[KISI_1]`, `[TCKN_1]` gibi etiketleri görür. Cevap dönünce etiketler
yerelde geri açılır. Gerçek değerler makineden hiç çıkmaz.

```
belge ──► [ NER modeli + kural katmanı ]  ──► maskeli belge ──► Gemini API
             (yerel, CPU, offline)                                  │
             kasa: etiket ↔ gerçek değer                            │
                    │                                               ▼
             cevap ◄─┴───────── etiketleri geri aç ◄──────── maskeli cevap
```

## Kurulum

```bash
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
```

`.env` içine [Google AI Studio](https://aistudio.google.com/apikey)'dan aldığınız
ücretsiz anahtarı yazın:

```
GEMINI_API_KEY=...
```

Anahtar olmasa da uygulama **çevrimdışı modda** açılır ve maskeleme zinciri
çalışır — sunumda internet giderse demo ayakta kalır.

## Çalıştırma

```bash
python app.py          # http://127.0.0.1:7860
python -m unittest     # 33 test
```

İlk çalıştırmada NER modeli (~440 MB) bir kez indirilir, sonrasında yerel
cache'ten okunur.

## Mimari

| Dosya | İş |
|---|---|
| `kvkk/anonymizer.py` | Kural katmanı, maskeleme, kasa, sızıntı denetimi |
| `kvkk/ner.py` | HF modeli, BIO birleştirme, unvan ayıklama |
| `kvkk/llm.py` | Gemini çağrısı — buluta çıkan **tek** yer |
| `app.py` | Gradio arayüzü |

### İki katman, çünkü tek katman yetmiyor

**Kural katmanı** TCKN, IBAN, telefon, e-posta, plaka, kredi kartı yakalar.
TCKN ve IBAN'da resmî sağlama (checksum / mod-97), kredi kartında Luhn
doğrulaması yapılır — ham "11 haneli sayı" regex'i fatura numarasından dosya
numarasına kadar her şeyi TCKN sanıyordu.

**Model katmanı** (`savasy/bert-base-turkish-ner-cased`) kişi ve yer adlarını
bulur. Bunlar regex'le yakalanamaz: "Ahmet Yılmaz"ın sabit bir biçimi yoktur.
Tüm Türkçe isimlerin listesi tutulamaz, "büyük harfle başlayan kelime" kuralı
ise belgedeki `Kardiyoloji`, `Troponin`, `SGK`, `HASTA ÇIKIŞ RAPORU` gibi her
şeyi isim sanardı.

Modelin asıl işi **bağlam**. Ölçülen örnek — aynı kelime, iki farklı sınıf:

```
"Kurum: Ankara Numune Eğitim ve Araştırma Hastanesi"
   → KURUM  'Ankara Numune Eğitim ve Araştırma Hastanesi'  (0.95)

"Adres: Kızılay Mahallesi, Çankaya / Ankara"
   → YER    'Ankara'                                        (0.97)
```

`Ankara` birinde kurum adının parçası, diğerinde kişinin adresi. Regex kelimeye
bakar, model cümleye bakar. Bu ayrımı yapan başka bir yöntem yok.

Çakışmada **kural kazanır**: model bir TCKN'yi yanlışlıkla isim sanarsa kural
katmanı onu düzeltir.

### Neden fine-tune yok?

Hazır model TCKN/IBAN diye bir kavram bilmiyor — ama bunlar biçimi tamamen belli
veriler. Bir modeli bunları öğrenmesi için eğitmek, sağlama algoritmasının
kesin olarak çözdüğü bir problemi olasılıksal hale getirmek olurdu. Kural
katmanı burada hem daha doğru hem daha ucuz. Model, yalnızca kuralın
yazılamayacağı yerde (isimler) kullanılıyor.

### Ölçülen üç sorun ve çözümleri

Bunlar tahmin değil; geliştirme sırasında karşılaşılıp testle sabitlendi.

1. **`transformers` kendi span birleştirmesi ismi kırpıyordu.** Ham model
   `Hasta Ahmet Yılmaz` için `B-PER 'Ahmet' 0.996` + `I-PER 'Yılmaz' 0.998`
   veriyor, ama pipeline'ın `aggregation_strategy` katmanı span'i sadece
   `Yılmaz`a indirip `Ahmet`i düşürüyordu. Maskeleyicide kırpılmış span =
   sızmış kişisel veri. Çözüm: BIO birleştirmesini `kvkk/ner.py` içinde
   kendimiz yapıyoruz.

2. **Alt-kelime parçaları maskenin dışında kalıyordu.** Tokenizer `Yılmaz`ı
   `Yılm` + `##az` diye bölüp ikinciyi `O` etiketleyince ekranda `[KISI_1]az`
   çıkıyordu. Çözüm: span'i kelime sınırına yaslıyoruz. Yalnızca harf/rakam
   üzerinden genişlediği için `Ankara'da` → `[YER_1]'da` olarak kalıyor, Türkçe
   çekim eki korunuyor.

3. **Model `Dr.`yi tek başına kişi adı sanıyordu** (skor tam 0.50) ve
   `Dr. Mehmet Öztürk`te unvanı ismin içine katıyordu. İkisi birden, aynı
   kişinin unvanlı ve unvansız geçişlerinin farklı etiket alması demekti.
   Çözüm: unvan ayıklama filtresi. Skor eşiği bilerek 0.5'te bırakıldı —
   bu bir gizlilik aracı, kaçan bir isim yanlış maskelenmiş bir kelimeden
   çok daha pahalı.

### Sızıntı denetimi

Uygulama "veri gitmiyor" demiyor, **ölçüyor**: buluta giden isteği kasadaki her
gerçek değer için tarayıp sonucu ekranda gösteriyor. Kimlik verileri (TCKN,
IBAN, telefon, e-posta, isim) için herhangi bir eşleşme kritik sayılır.

Yer/kurum adları ayrı tutulur: `Ankara` maskeliyken `Ankara Numune Hastanesi`
maskelenmemişse kelime istekte geçer — bu kimlik ifşası değil, denetim bunu
uyarı olarak bildirir.

### KURUM neden kapalı — ve sunumda neden açmamalısın

İki sebep var, ikincisi ölçüldü:

1. Kurum adı tek başına kişisel veri değil; maskelenince LLM'in işine yarayan
   bağlam boşuna kayboluyor.
2. Model tıbbi metinde KURUM etiketini güvenilmez veriyor:
   `hipertansiyon` → KURUM (0.83), `Kardiyoloji` → KURUM (0.88). Açarsanız
   **hastanın tanısı maskelenir** ve asistan işe yaramaz hale gelir.

Yani `Ankara` uyarısını kapatmak için KURUM'u açmak çözüm değil — demoyu bozar.
Uyarı zaten kritik değil; olduğu gibi bırakın ve sorulursa dürüst cevap verin.

## 5 dakikalık sunum planı

> **Sahne notu:** Demoyu açar açmaz bir kez boşa çalıştır — model ilk yüklemede
> ~5 sn alıyor, sahnede bekleme. Ve analiz ettikten sonra **sayfayı yenileme**:
> etiket–değer eşlemesini tutan kasa oturumda durur, yenilersen gider. (Uygulama
> bu durumda sessizce yanlış davranmaz, uyarı verir — ama sahnede o uyarıyı da
> görmemek en iyisi. Görürsen "Analiz Et"e tekrar bas, düzelir.)

**0:00 – 0:45 — Problem.** Ekranda hasta raporu. "Bu belgeyi ChatGPT'ye atıp
özetletmek istiyorsunuz. Atamazsınız: içinde TC kimlik no, isim, IBAN var. KVKK.
Türkiye'de her hastane, her hukuk bürosu bu duvara çarpıyor."

**0:45 – 1:30 — Çözüm fikri.** Mimari şeması. "Modeli cevabı üretmek için değil,
buluta giden yolu kontrol etmek için kullanıyoruz. HuggingFace'ten çektiğimiz
Türkçe NER modeli tamamen bu makinede, CPU'da çalışıyor."

**1:30 – 2:45 — Canlı demo.** `Analiz Et`e bas. Kişisel veriler renkli
işaretlensin. Bulgu tablosunu göster: **"Kural katmanı 6, model katmanı 8"** —
iki katmanın da çalıştığı burada görünüyor. Sağdaki maskeli belgeyi göster:
"Buluta giden hali bu."

**2:45 – 3:45 — Asıl an.** Soruyu sor, doğru cevabı al. Sonra denetim rozetini
göster: *"Kasadaki 13 gerçek değerin tamamı tarandı; hiçbir kimlik verisi
buluta giden istekte yok."* Akordiyonu aç, Gemini'ye giden ham isteği göster.
**"İddia etmiyoruz, ölçüyoruz."**

**3:45 – 4:30 — Mühendislik.** Yukarıdaki üç ölçülen sorundan birini anlat —
en çarpıcısı `transformers`ın span kırpması. "Kütüphanenin hazır fonksiyonuna
güvenseydik isimlerin yarısı sızacaktı. Ham skorlara bakıp birleştirmeyi
kendimiz yazdık, 33 testle sabitledik."

**4:30 – 5:00 — Kapanış.** "Fine-tune yapmadık çünkü gerekmiyordu: TCKN'yi
sağlama algoritması kesin çözüyor, modeli buna eğitmek kesin olanı olasılıksal
yapardı. Model sadece kuralın yazılamayacağı yerde — isimlerde — çalışıyor."

### Hocanın sorabileceği sorular

**"Neden eğitmediniz?"** → Yukarıdaki kapanış. Eğitmek burada teknik olarak
yanlış tercih olurdu; kural katmanı hem daha doğru hem ölçülebilir.

**"Model bir ismi kaçırırsa?"** → Kaçırabilir, recall %100 değil. Bu yüzden
sızıntı denetimi var: gönderimden önce ölçüyoruz. Üretimde kritik sızıntıda
istek bloklanır. Eşiği de bu yüzden düşük tuttuk.

**"Etiketler LLM'i şaşırtmıyor mu?"** → Sistem talimatı açıkça "bu etiketlerin
arkasındaki değerleri bilmiyorsun, tahmin etme, aynen kullan" diyor. Demoda
cevaptaki etiketler doğru yere oturuyor.

**"Neden Ankara uyarıda çıkıyor?"** → Dürüst cevap: kendi tasarım kararımızın
yan etkisi. KURUM maskelemesini kapalı bıraktık, o yüzden şehir adı kurum
adının içinde geçiyor. Denetim bunu gizlemiyor, sınıflandırıyor. Açsak kalkardı
ama açamayız: model `hipertansiyon`u da kurum sanıyor, tanı maskelenirdi.

**"Model burada ne işe yarıyor, regex yetmez mi?"** → Yetmez, çünkü isimlerin
biçimi yok. Ölçülen cevap yukarıdaki `Ankara` örneği: aynı kelime bir cümlede
kurum, diğerinde adres. Model cümleye bakıyor, regex kelimeye. Ayrıca son
çalıştırmada 14 bulgunun 8'i model katmanından geldi.

## Örnek belgeler

`ornek_belgeler/` altındaki hasta raporu ve icra dilekçesi tamamen kurgudur.
İçlerindeki TCKN, IBAN ve kart numaraları **üretilmiş sahte verilerdir** —
ama sağlamaları geçerlidir, aksi halde kural katmanı onları (doğru şekilde)
maskelemezdi.
