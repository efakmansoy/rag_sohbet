"""
TÜBİTAK 2204-A Yarışma Asistanı - Ana RAG Sistemi
Streamlit'ten bağımsız ana model ve işleme kodları
"""

# Standart Kütüphane İmportları
import os
import glob
from typing import Optional, List, Tuple, Dict, Any

# ChromaDB telemetry'yi devre dışı bırak (deploy ortamları için)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
# os.environ["CHROMA_SERVER_NOFILE"] = "2048"  # Gerekirse yorum satırından çıkarın

# Üçüncü Parti İmportları
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# LangChain İmportları
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document


# Konfigürasyon Sabitleri
class Config:
    """Uygulama konfigürasyon sabitleri"""
    EMBEDDING_MODEL = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    CLASSIFIER_MODEL = "efakmansoy/bert-base-turkish-fine-tuned"
    DATASET_NAME = "Q-bert/Custom-2204"
    
    # Dosya Yolları
    MAIN_DB_PATH = "./chroma_db"
    DATASET_DB_PATH = "./dataset_chroma_db"
    FILES_DIR = "./files"
    
    # Model Parametreleri
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    DATASET_CHUNK_SIZE = 500  # Küçültüldü - hız için
    DATASET_CHUNK_OVERLAP = 50  # Küçültüldü - hız için
    RETRIEVAL_K = 13
    EXAMPLE_PROJECTS_K = 2
    
    # Web Adresi
    TUBITAK_URL = "https://tubitak.gov.tr/tr/yarismalar/2204-lise-ogrencileri-arastirma-projeleri-yarismasi"
    
    # Google API Anahtarı (Environment Variable'dan alınmalı)
    GOOGLE_API_KEY = "AIzaSyBUs9xbTmXp2-En0tAF0kks0wWkwxUdgIs"
    
    # Prompt Şablonu
    PROMPT_TEMPLATE = """
Sen, TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması hakkında öğrenci ve danışmanlara yardımcı olan bir asistansın. Görevin, onlara yarışmanın şartnameleri, başvuru ve rapor süreçleri gibi konularda, **sadece verilen belgelerden edindiğin bilgilere dayanarak** rehberlik etmektir.
Eğer verilen bağlamda sorunun cevabı yoksa, elindeki bilgilere göre en mantıklı yanıtı üretmeye çalış.  Kesinlikle uydurma bilgi verme. Yanıtların profesyonel, anlaşılır ve yarışma konusuna odaklı olsun.
Öneriler ve tavsiyeler verirken, TÜBİTAK'ın resmi politikalarına ve yönergelerine uygun olmasına dikkat et. Öneri verirken yaratıcı ol ve verilen belgeleri kullanmak zorunda değilsin sadece öneri verirken.

Konuşma Geçmişi:
{chat_history}

Bağlam:
{context}

{example_projects}

Soru:
{question}

Yardımcı Asistanın Cevabı:
"""


# Yardımcı Sınıflar
class Logger:
    """Debug loglama yardımcısı"""
    
    @staticmethod
    def debug(message: str) -> None:
        """Debug mesajı yazdır"""
        print(f"[DEBUG] {message}")
    
    @staticmethod
    def info(message: str) -> None:
        """Info mesajı yazdır"""
        print(f"[INFO] {message}")
    
    @staticmethod
    def error(message: str) -> None:
        """Error mesajı yazdır"""
        print(f"[ERROR] {message}")


class DatasetProcessor:
    """Dataset işleme yardımcı sınıfı"""
    
    @staticmethod
    def extract_project_fields(item) -> Dict[str, str]:
        """Dataset item'ından proje alanlarını çıkar."""
        try:
            # DEBUG: Item tipini kontrol et
            Logger.debug(f"extract_project_fields: item type = {type(item)}")
            if isinstance(item, dict):
                Logger.debug(f"extract_project_fields: item keys = {list(item.keys())}")
            
            # Item'ın içeriğini al
            if isinstance(item, dict) and 'Proje' in item:
                content = item['Proje']
                Logger.debug(f"extract_project_fields: content length = {len(content)}")
                Logger.debug(f"extract_project_fields: content preview = {content[:200]}...")
            elif isinstance(item, str):
                content = item
                Logger.debug(f"extract_project_fields: string content length = {len(content)}")
                Logger.debug(f"extract_project_fields: string content = '{content}'")
            else:
                content = str(item)
                Logger.debug(f"extract_project_fields: converted to string = {content[:200]}...")
            
            # Temel alanlar
            fields = {
                'kategori': 'Bilinmiyor',
                'proje_alani': 'Bilinmiyor', 
                'proje_ismi': 'Bilinmiyor',
                'ozet': 'Bilinmiyor'
            }
            
            lines = content.split('\n')
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Kategori İsmi
                if 'Kategori İsmi:' in line or 'Kategori Ismi:' in line:
                    if i + 1 < len(lines):
                        fields['kategori'] = lines[i + 1].strip()
                
                # Proje Alanı
                elif 'Proje Alanı:' in line or 'Proje Alani:' in line:
                    if i + 1 < len(lines):
                        fields['proje_alani'] = lines[i + 1].strip()
                
                # Proje İsmi
                elif 'Proje İsmi:' in line or 'Proje Ismi:' in line:
                    if i + 1 < len(lines):
                        fields['proje_ismi'] = lines[i + 1].strip()
                
                # Özet
                elif 'Özet:' in line or 'Ozet:' in line:
                    if i + 1 < len(lines):
                        # Özet birden fazla satır olabilir
                        ozet_lines = []
                        j = i + 1
                        while j < len(lines) and lines[j].strip():
                            ozet_lines.append(lines[j].strip())
                            j += 1
                        if ozet_lines:
                            fields['ozet'] = ' '.join(ozet_lines)
                
                i += 1
            
            return fields
            
        except Exception as e:
            Logger.error(f"extract_project_fields hatası: {e}")
            return {
                'kategori': 'Bilinmiyor',
                'proje_alani': 'Bilinmiyor',
                'proje_ismi': 'Bilinmiyor', 
                'ozet': 'Bilinmiyor'
            }
    
    @staticmethod
    def create_project_content(fields: dict) -> str:
        """Proje alanlarından içerik oluştur"""
        return f"""
Kategori İsmi: {fields['kategori']}
Proje Alanı: {fields['proje_alani']}
Proje İsmi: {fields['proje_ismi']}
Özet: {fields['ozet']}
""".strip()
    
    @staticmethod
    def create_document(fields: dict) -> Document:
        """Document objesi oluştur"""
        content = DatasetProcessor.create_project_content(fields)
        return Document(
            page_content=content,
            metadata={
                "kategori": fields['kategori'],
                "proje_alani": fields['proje_alani'],
                "proje_ismi": fields['proje_ismi'],
                "source": "dataset"
            }
        )


class DocumentManager:
    """Doküman yönetimi için yardımcı sınıf"""
    
    @staticmethod
    def load_pdf_documents(files_dir: str) -> List[Document]:
        """PDF dosyalarını yükle"""
        pdf_files = glob.glob(os.path.join(files_dir, "*.pdf"))
        documents = []
        
        if pdf_files:
            Logger.debug(f"{len(pdf_files)} PDF dosyası bulundu")
            for file_path in pdf_files:
                try:
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    Logger.debug(f"PDF yüklendi: {os.path.basename(file_path)}")
                except Exception as e:
                    Logger.error(f"PDF yüklenirken hata {file_path}: {e}")
        
        return documents
    
    @staticmethod
    def load_web_content(url: str) -> List[Document]:
        """Web içeriğini yükle"""
        try:
            Logger.debug(f"Web içeriği yükleniyor: {url}")
            web_loader = WebBaseLoader(url)
            return web_loader.load()
        except Exception as e:
            Logger.error(f"Web içeriği yüklenirken hata: {e}")
            return []


class ClassifierService:
    """Soru sınıflandırma servisi"""
    
    def __init__(self):
        """ClassifierService başlatıcı"""
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Fikir klasifikasyon modelini yükle"""
        try:
            Logger.debug(f"Classifier model yükleniyor: {Config.CLASSIFIER_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(Config.CLASSIFIER_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(Config.CLASSIFIER_MODEL)
            Logger.debug("Classifier model başarıyla yüklendi!")
        except Exception as e:
            Logger.error(f"Classifier model yüklenirken hata: {e}")
            self.tokenizer = None
            self.model = None
    
    def classify_question(self, question: str) -> int:
        """Sorunun fikir içerip içermediğini sınıflandır"""
        if self.tokenizer is None or self.model is None:
            return 0
        
        try:
            inputs = self.tokenizer(question, return_tensors="pt", padding=True, 
                                  truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                
            return predicted_class
        except Exception as e:
            Logger.error(f"Sınıflandırma sırasında hata: {e}")
            return 0


class VectorStoreService:
    """Vektör veritabanı servisi"""
    
    def __init__(self):
        """VectorStoreService başlatıcı"""
        self.embeddings = self._get_embeddings()
    
    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """Embedding modeli döndür"""
        Logger.debug(f"Embedding model yükleniyor: {Config.EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    
    def create_main_vectorstore(self) -> Optional[Chroma]:
        """Ana vektör veritabanını oluştur"""
        Logger.debug("Ana vektör veritabanı oluşturuluyor")
        
        # PDF dokümanları yükle
        pdf_documents = DocumentManager.load_pdf_documents(Config.FILES_DIR)
        
        # Web içeriğini yükle
        web_documents = DocumentManager.load_web_content(Config.TUBITAK_URL)
        
        # Tüm dokümanları birleştir
        all_documents = pdf_documents + web_documents
        
        if not all_documents:
            Logger.error("Hiçbir belge yüklenemedi.")
            return None
        
        Logger.debug(f"Toplam {len(all_documents)} doküman yüklendi")
        
        # Text splitter uygula
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        split_documents = text_splitter.split_documents(all_documents)
        Logger.debug(f"{len(split_documents)} chunk oluşturuldu")
        
        # Vektör veritabanını oluştur
        vectorstore = Chroma.from_documents(
            documents=split_documents,
            embedding=self.embeddings,
            collection_name="parent_child_collection",
            persist_directory=Config.MAIN_DB_PATH
        )
        
        return vectorstore
    
    def load_or_create_main_vectorstore(self) -> Optional[Chroma]:
        """Ana vektör veritabanını yükle veya oluştur"""
        if os.path.exists(Config.MAIN_DB_PATH) and os.path.isdir(Config.MAIN_DB_PATH):
            try:
                Logger.debug("Mevcut ana vektör veritabanı yükleniyor")
                vectorstore = Chroma(
                    collection_name="parent_child_collection",
                    embedding_function=self.embeddings,
                    persist_directory=Config.MAIN_DB_PATH
                )
                return vectorstore
            except Exception as e:
                Logger.error(f"Ana veritabanı yüklenirken hata: {e}")
                Logger.info("Ana veritabanı yeniden oluşturuluyor...")
        
        return self.create_main_vectorstore()
    
    def setup_dataset_vectorstore(self) -> Optional[Chroma]:
        """Dataset vektör veritabanını yükle/oluştur"""
        Logger.debug(f"Dataset veritabanı kontrol ediliyor: {Config.DATASET_DB_PATH}")
        
        # Mevcut veritabanını yükle
        if os.path.exists(Config.DATASET_DB_PATH) and os.path.isdir(Config.DATASET_DB_PATH):
            try:
                Logger.debug("Mevcut dataset veritabanı yükleniyor")
                dataset_vectorstore = Chroma(
                    collection_name="dataset_collection",
                    embedding_function=self.embeddings,
                    persist_directory=Config.DATASET_DB_PATH
                )
                Logger.info("Dataset vektör veritabanı başarıyla yüklendi.")
                return dataset_vectorstore
            except Exception as e:
                Logger.error(f"Dataset veritabanı yüklenirken hata: {e}")
                Logger.info("Dataset veritabanı yeniden oluşturuluyor...")
        
        # Yeni veritabanı oluştur
        return self.create_dataset_vectorstore()
    
    def create_dataset_vectorstore(self) -> Optional[Chroma]:
        """Dataset vektör veritabanını oluştur"""
        Logger.debug("Dataset veritabanı oluşturuluyor")
        
        try:
            # Mevcut vektör veritabanını temizle
            if os.path.exists(Config.DATASET_DB_PATH):
                import shutil
                shutil.rmtree(Config.DATASET_DB_PATH)
                Logger.debug(f"Mevcut dataset veritabanı temizlendi: {Config.DATASET_DB_PATH}")
            
            Logger.debug(f"HuggingFace'den dataset yükleniyor: {Config.DATASET_NAME}")
            
            dataset = load_dataset(Config.DATASET_NAME)
            Logger.debug(f"Dataset yüklendi! Train boyutu: {len(dataset)}")
            
            # Dataset yapısını kontrol et
            documents = []
            Logger.debug("Dataset Document formatına dönüştürülüyor (İlk 1000 proje - splitting yok)")
            
            # Performans için ilk 1000 projeyi al (splitting yok, daha hızlı)
            processed_count = 0
            num = len(dataset["train"]["Proje"])
            for i, item in enumerate(dataset["train"]["Proje"]):
                if i % 50 == 0:  # Her 50 projede bir rapor
                    Logger.debug(f"İşlenen proje sayısı: {i}/{num}")

                try:
                    full_text = item
                    if i < 3:
                        Logger.debug(f"DEBUG Proje {i}: Dict item, Proje key = {len(full_text)} kar")
                    from langchain.schema import Document
                    doc = Document(
                        page_content=full_text,
                        metadata={"source": "dataset", "project_id": i}
                    )
                    documents.append(doc)
                    processed_count += 1
                    
                    if i < 3:
                        Logger.debug(f"DEBUG Proje {i}: EKLENDI - {len(full_text)} karakter")
                    else:
                        if i < 3:
                            Logger.debug(f"DEBUG Proje {i}: ATLANDI - çok kısa ({len(full_text)} kar)")
                except Exception as item_error:
                    Logger.error(f"Proje işlenirken hata (item {i}): {item_error}")
                    continue
            
            Logger.debug(f"Toplam {processed_count} geçerli proje Document formatına dönüştürüldü")
            
            if not documents:
                Logger.error("Hiç geçerli proje bulunamadı!")
                return None
            
            # Text splitting yapmıyoruz - her proje direkt bir chunk
            Logger.debug("Projeler direkt chunk olarak kullanılıyor (splitting yok)")
            
            # Vektör veritabanını oluştur - direkt documents ile
            Logger.debug("ChromaDB vektör veritabanı oluşturuluyor")
            dataset_vectorstore = Chroma.from_documents(
                documents=documents,  # split_documents değil, direkt documents
                embedding=self.embeddings,
                collection_name="dataset_collection",
                persist_directory=Config.DATASET_DB_PATH
            )
            
            Logger.debug(f"Dataset başarıyla işlendi! {processed_count} proje eklendi")
            
            # Test similarity search
            test_results = dataset_vectorstore.similarity_search("proje", k=1)
            if test_results:
                test_kategori = test_results[0]
                Logger.debug(f"Test search sonucu: {test_kategori}")
            else:
                Logger.error("Test search başarısız!")
            
            return dataset_vectorstore
            
        except Exception as e:
            Logger.error(f"Dataset yüklenirken ana hata: {e}")
            return None
    
    def get_example_projects(self, question: str, k: int = Config.EXAMPLE_PROJECTS_K) -> str:
        """Dataset'ten örnek projeler getir"""
        try:
            Logger.debug(f"Dataset'ten örnek projeler getiriliyor... Yol: {Config.DATASET_DB_PATH}")
            dataset_vectorstore = Chroma(
                collection_name="dataset_collection",
                embedding_function=self.embeddings,
                persist_directory=Config.DATASET_DB_PATH
            )
            
            Logger.debug(f"Similarity search yapılıyor, k={k*2}")  # Daha fazla getir
            docs = dataset_vectorstore.similarity_search(question, k=k*2)
            
            if docs:
                Logger.debug(f"{len(docs)} benzer proje bulundu")
                
                # Duplicate temizleme
                seen_hashes = set()
                unique_docs = []
                
                for i, doc in enumerate(docs):
                    project_id = doc.metadata.get('project_id', f'unknown_{i}')
                    content_hash = hash(doc.page_content[:200])  # İlk 200 karakter hash'i
                    
                    Logger.debug(f"Proje {i+1}: ID={project_id}, Hash={content_hash}, Uzunluk={len(doc.page_content)}")
                    
                    if content_hash not in seen_hashes:
                        seen_hashes.add(content_hash)
                        unique_docs.append(doc)
                        
                        # Yeterince unique proje topladık
                        if len(unique_docs) >= k:
                            break
                
                Logger.debug(f"Duplicate temizlendi: {len(unique_docs)} unique proje / {len(docs)} toplam")
                
                examples = "\n\nÖrnek Projeler:\n"
                for i, doc in enumerate(unique_docs, 1):
                    examples += f"{i}. {doc.page_content}\n---\n"
                Logger.debug(f"Örnek projeler hazırlandı, toplam uzunluk: {len(examples)} karakter")
                return examples
            else:
                Logger.debug("Hiç benzer proje bulunamadı")
                return ""
        except Exception as e:
            Logger.error(f"Örnek projeler getirilirken hata: {e}")
            return ""


class RAGSystem:
    """RAG sistem yönetimi"""
    
    def __init__(self):
        """RAGSystem başlatıcı"""
        self.vector_service = VectorStoreService()
        self.classifier_service = ClassifierService()
        self.retriever = None
        self.qa_chain = None
        self.llm = None
        self.memory = None
        
        # Sistemi başlat
        self.setup()
    
    def setup(self) -> bool:
        """RAG sistemini kurulum"""
        try:
            # Ana RAG sistemi yükle/oluştur
            vectorstore = self.vector_service.load_or_create_main_vectorstore()
            if not vectorstore:
                Logger.error("Ana vektör veritabanı oluşturulamadı")
                return False
            
            # Dataset vektör veritabanını yükle/oluştur
            self.vector_service.setup_dataset_vectorstore()
            
            # Retriever oluştur
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
            
            # LLM chain'i kur
            self._setup_llm_chain()
            
            Logger.info("RAG sistemi başarıyla kuruldu")
            return True
            
        except Exception as e:
            Logger.error(f"RAG sistem kurulumunda hata: {e}")
            return False
    
    def _setup_llm_chain(self) -> None:
        """LLM chain'i kur"""
        try:
            # LLM'i başlat
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.5,
                google_api_key=Config.GOOGLE_API_KEY
            )
            
            # Memory'yi başlat
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history", 
                return_messages=True
            )
            
            # Multi-query retriever oluştur
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=self.retriever,
                llm=self.llm
            )
            
            # Basit conversational chain oluştur (custom prompt için)
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=multi_query_retriever,
                memory=self.memory,
                rephrase_question=False
            )
            
            Logger.debug("LLM chain başarıyla kuruldu")
            
        except Exception as e:
            Logger.error(f"LLM chain kurulumunda hata: {e}")
            raise
    
    def process_question(self, question: str) -> str:
        """Kullanıcı sorusunu işle ve cevap döndür"""
        if not self.qa_chain:
            return "RAG sistemi henüz hazır değil. Lütfen bekleyin."
        
        try:
            Logger.debug(f"Kullanıcı sorusu: {question}")
            
            # Soruyu sınıflandır
            Logger.debug("Soru sınıflandırılıyor")
            classification = self.classifier_service.classify_question(question)
            Logger.debug(f"Sınıflandırma sonucu: {classification}")
            
            # Eğer fikir içeren bir soru ise (1), dataset'ten örnekler ekle
            if classification == 1:
                Logger.debug("Fikir içeren soru tespit edildi, örnek projeler getiriliyor")
                example_projects = self.vector_service.get_example_projects(question)
                
                if example_projects:
                    # Örnek projeleri context olarak ana soruya ekle
                    enhanced_context = f"{question}\n\nİlgili Başarılı Proje Örnekleri:\n{example_projects}"
                    Logger.debug(enhanced_context)
                    Logger.debug(f"Context örnek projelerle genişletildi. Uzunluk: {len(enhanced_context)} karakter")
                    
                    # Custom prompt ile manual çağrı yap
                    result = self._process_with_examples(question, example_projects)
                    return result
                else:
                    Logger.debug("Örnek proje bulunamadı, normal işlem devam ediyor")
            else:
                Logger.debug("Normal soru, örnek proje eklenmeyecek")
            
            # Normal chain invoke
            result = self.qa_chain.invoke({"question": question})
            return result["answer"]
            
        except Exception as e:
            Logger.error(f"Soru işlenirken hata: {e}")
            return f"Üzgünüm, sorunuzu işlerken bir hata oluştu: {str(e)}"
    
    def _process_with_examples(self, question: str, example_projects: str) -> str:
        """Örnek projelerle birlikte soruyu işle"""
        try:
            # Retriever'dan context dokümanları al
            docs = self.retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Chat history al
            chat_history = ""
            if self.memory and hasattr(self.memory, 'chat_memory'):
                messages = self.memory.chat_memory.messages
                chat_history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in messages[-4:]])  # Son 4 mesaj
            
            # Custom prompt hazırla
            enhanced_prompt = Config.PROMPT_TEMPLATE.format(
                chat_history=chat_history,
                context=context,
                example_projects=f"\nİlgili Başarılı Proje Örnekleri:\n{example_projects}\n",
                question=question
            )
            
            # LLM'e gönder
            response = self.llm.invoke(enhanced_prompt)
            
            # Memory'yi güncelle
            if self.memory:
                self.memory.save_context({"input": question}, {"output": response.content})
            
            return response.content
            
        except Exception as e:
            Logger.error(f"Örnek projeli işlem sırasında hata: {e}")
            # Fallback: normal chain
            result = self.qa_chain.invoke({"question": question})
            return result["answer"]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Konuşma geçmişini döndür"""
        if self.memory and hasattr(self.memory, 'chat_memory'):
            messages = []
            for message in self.memory.chat_memory.messages:
                messages.append({
                    "role": "user" if hasattr(message, 'content') and message.__class__.__name__ == "HumanMessage" else "assistant",
                    "content": message.content
                })
            return messages
        return []
    
    def clear_conversation_history(self) -> None:
        """Konuşma geçmişini temizle"""
        if self.memory:
            self.memory.clear()
            Logger.debug("Konuşma geçmişi temizlendi")


# Kolay başlatma için fabrika fonksiyonu
def create_rag_system() -> RAGSystem:
    """RAG sistemi oluştur ve döndür"""
    return RAGSystem()


if __name__ == "__main__":
    rag = create_rag_system()
    if rag.qa_chain:
        print("RAG sistem test edilebilir.")
        # Test sorusu
        test_question = "Yarışma başvurusu nasıl yapılır?"
        response = rag.process_question(test_question)
        print(f"Test Sorusu: {test_question}")
        print(f"Cevap: {response}")
    else:
        print("RAG sistem başlatılamadı.")