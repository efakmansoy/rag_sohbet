# 🏆 TÜBİTAK 2204-A Yarışma Asistanı

TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması için öğrenci ve danışmanlara yardımcı olmak üzere geliştirilmiş yapay zeka destekli sohbet asistanı. Bu uygulama, RAG (Retrieval-Augmented Generation) teknolojisi kullanarak yarışma şartnameleri, başvuru süreçleri ve rapor hazırlama konularında rehberlik sağlar.

## 📋 Özellikler

- **Akıllı Soru-Cevap Sistemi**: TÜBİTAK 2204-A yarışması hakkında doğru ve güvenilir bilgiler sunar
- **Belge Yükleme ve İşleme**: PDF dosyalarınızı yükleyerek içeriklerini analiz edebilir
- **Örnek Proje Önerileri**: Geçmiş yarışma projelerinden örnekler ve öneriler sunar
- **Konuşma Geçmişi**: Önceki sorularınızı ve cevaplarınızı hatırlayarak bağlamsal yanıtlar verir
- **Türkçe Dil Desteği**: Tamamen Türkçe dil modelleri ile optimize edilmiş

## 🚀 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- Google Gemini API anahtarı

### Sistem Gereksinimleri

Aşağıdaki sistem paketlerinin kurulu olması gerekir:
```bash
poppler-utils
libgl1-mesa-glx
tesseract-ocr
```

### Python Kütüphanelerinin Kurulumu

```bash
# Repository'yi klonlayın
git clone https://github.com/efakmansoy/RAG-Sohbet-Asistan--Streamlit-Arayuzu.git
cd RAG-Sohbet-Asistan--Streamlit-Arayuzu

# Gerekli Python paketlerini yükleyin
pip install -r requirements.txt
```

### Yapılandırma

1. Google Gemini API anahtarınızı edinin
2. `app.py` dosyasındaki `GOOGLE_API_KEY` değerini kendi API anahtarınızla değiştirin veya ortam değişkeni olarak ayarlayın

```python
Config.GOOGLE_API_KEY = "your-api-key-here"
```

## 💻 Kullanım

### Streamlit Uygulamasını Başlatma

```bash
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde çalışacaktır.

### Temel Kullanım Adımları

1. **Belge Yükleme**: Sidebar'dan PDF dosyalarınızı yükleyin
2. **Veritabanı Oluşturma**: "Vektör Veritabanını Oluştur" butonuna tıklayarak belgelerinizi işleyin
3. **Soru Sorma**: Ana ekranda sohbet kutusuna sorularınızı yazın
4. **Örnek Proje İnceleme**: Dataset'ten örnek projeler hakkında bilgi alın

## 🏗️ Teknik Mimari

### Kullanılan Teknolojiler

- **Streamlit**: Web arayüzü
- **LangChain**: RAG sistemi ve doğal dil işleme zinciri
- **ChromaDB**: Vektör veritabanı
- **Hugging Face Transformers**: Türkçe dil modelleri
- **Google Gemini**: Büyük dil modeli (LLM)

### Kullanılan Modeller

- **Embedding Modeli**: `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr`
- **Sınıflandırma Modeli**: `efakmansoy/bert-base-turkish-fine-tuned`
- **Dataset**: `Q-bert/Custom-2204` (Hugging Face)

### Bileşenler

#### 1. **VectorStoreService**
Vektör veritabanı oluşturma ve yönetimi:
- PDF belgelerini işleme
- ChromaDB ile vektör saklama
- Benzerlik araması

#### 2. **RAGSystem**
Ana RAG sistemi:
- Belge geri getirme (retrieval)
- Konuşma geçmişi yönetimi
- Yanıt oluşturma

#### 3. **DocumentClassifier**
Belge sınıflandırma:
- Soruların kategorize edilmesi
- İlgili belgelerin filtrelenmesi

#### 4. **StreamlitChatInterface**
Kullanıcı arayüzü:
- Sohbet arayüzü
- Dosya yükleme
- Veri görselleştirme

## ⚙️ Yapılandırma Parametreleri

| Parametre | Açıklama | Varsayılan Değer |
|-----------|----------|------------------|
| `CHUNK_SIZE` | Belge parçalama boyutu | 1000 |
| `CHUNK_OVERLAP` | Parçalar arası örtüşme | 100 |
| `RETRIEVAL_K` | Geri getirilen belge sayısı | 13 |
| `EXAMPLE_PROJECTS_K` | Örnek proje sayısı | 2 |

## 📁 Dizin Yapısı

```
.
├── app.py                  # Ana uygulama dosyası
├── requirements.txt        # Python bağımlılıkları
├── packages.txt           # Sistem bağımlılıkları
├── LICENSE                # Apache 2.0 Lisans
├── files/                 # Yüklenen PDF dosyaları
├── chroma_db/            # Ana vektör veritabanı
└── dataset_chroma_db/    # Dataset vektör veritabanı
```

## 🔒 Güvenlik Notları

- API anahtarlarınızı asla kod içinde saklamayın
- Üretim ortamında ortam değişkenleri kullanın
- `.gitignore` dosyasında hassas bilgileri dışlayın

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları takip edin:

1. Bu repository'yi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje Apache License 2.0 altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## 🔗 Bağlantılar

- [TÜBİTAK 2204-A Yarışması](https://tubitak.gov.tr/tr/yarismalar/2204-lise-ogrencileri-arastirma-projeleri-yarismasi)
- [Hugging Face Dataset](https://huggingface.co/datasets/Q-bert/Custom-2204)

## 📧 İletişim

Sorularınız veya önerileriniz için lütfen bir issue açın.

## 🙏 Teşekkürler

Bu projeyi geliştirirken kullanılan açık kaynak kütüphanelerin ve toplulukların katkılarına teşekkür ederiz.

---

**Not**: Bu asistan, TÜBİTAK 2204-A yarışması için bilgi verme amacıyla geliştirilmiştir. Resmi bilgiler için mutlaka [TÜBİTAK'ın resmi web sitesini](https://tubitak.gov.tr) ziyaret ediniz.
