"""
TÜBİTAK 2204-A Yarışma Asistanı - Ana RAG Sistemi
Streamlit'ten bağımsız ana model ve işleme kodları
"""

# Standart Kütüphane İmportları
import os
import glob
from typing import Optional, List, Tuple, Dict, Any

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
    DATASET_CHUNK_SIZE = 800
    DATASET_CHUNK_OVERLAP = 80
    RETRIEVAL_K = 10
    EXAMPLE_PROJECTS_K = 4
    
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
    def extract_project_fields(item: dict) -> dict:
        """Dataset item'ından proje alanlarını güvenli şekilde çıkar"""
        return {
            'kategori': item.get('Kategori İsmi', item.get('kategori', item.get('Kategori', 'Bilinmiyor'))),
            'proje_alani': item.get('Proje Alanı', item.get('proje_alani', item.get('Proje Alani', 'Bilinmiyor'))),
            'proje_ismi': item.get('Proje İsmi', item.get('proje_ismi', item.get('Proje Ismi', 'Bilinmiyor'))),
            'ozet': item.get('Özet', item.get('ozet', item.get('Ozet', 'Özet bulunamadı')))
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
            Logger.debug(f"HuggingFace'den dataset yükleniyor: {Config.DATASET_NAME}")
            
            dataset = load_dataset(Config.DATASET_NAME)
            Logger.debug(f"Dataset yüklendi! Train boyutu: {len(dataset['train'])}")
            
            # Dataset yapısını kontrol et
            if len(dataset['train']) > 0:
                first_item = dataset['train'][0]
                Logger.debug(f"Dataset sütunları: {list(first_item.keys())}")
            
            # Dataset'i Document formatına dönüştür
            documents = []
            Logger.debug("Dataset Document formatına dönüştürülüyor")
            
            for i, item in enumerate(dataset['train']):
                if i % 1000 == 0:  # Her 1000 projede bir rapor
                    Logger.debug(f"İşlenen proje sayısı: {i}/{len(dataset['train'])}")
                
                try:
                    fields = DatasetProcessor.extract_project_fields(item)
                    doc = DatasetProcessor.create_document(fields)
                    documents.append(doc)
                except Exception as item_error:
                    Logger.error(f"Proje işlenirken hata (item {i}): {item_error}")
                    continue
            
            Logger.debug(f"Toplam {len(documents)} proje Document formatına dönüştürüldü")
            
            # Text splitter uygula
            Logger.debug("Text splitter uygulanıyor")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.DATASET_CHUNK_SIZE, 
                chunk_overlap=Config.DATASET_CHUNK_OVERLAP
            )
            split_documents = text_splitter.split_documents(documents)
            Logger.debug(f"{len(split_documents)} chunk oluşturuldu")
            
            # Vektör veritabanını oluştur
            Logger.debug("ChromaDB vektör veritabanı oluşturuluyor")
            dataset_vectorstore = Chroma.from_documents(
                documents=split_documents,
                embedding=self.embeddings,
                collection_name="dataset_collection",
                persist_directory=Config.DATASET_DB_PATH
            )
            
            Logger.debug(f"Dataset başarıyla işlendi! {len(documents)} proje, {len(split_documents)} chunk eklendi")
            Logger.info(f"Dataset başarıyla işlendi! {len(documents)} proje eklendi.")
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
            
            Logger.debug(f"Similarity search yapılıyor, k={k}")
            docs = dataset_vectorstore.similarity_search(question, k=k)
            
            if docs:
                Logger.debug(f"{len(docs)} benzer proje bulundu")
                examples = "\n\nÖrnek Projeler:\n"
                for i, doc in enumerate(docs, 1):
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
            
            # Prompt template'i oluştur
            custom_prompt = PromptTemplate(
                template=Config.PROMPT_TEMPLATE,
                input_variables=["chat_history", "context", "question"]
            )
            
            # Multi-query retriever oluştur
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=self.retriever,
                llm=self.llm
            )
            
            # Conversational chain oluştur
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=multi_query_retriever,
                memory=self.memory,
                rephrase_question=False,
                combine_docs_chain_kwargs={"prompt": custom_prompt} 
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
                
                # Prompt'u örneklerle genişlet
                enhanced_question = question + example_projects
                Logger.debug(f"Soru örnek projelerle genişletildi. Yeni uzunluk: {len(enhanced_question)} karakter")
                result = self.qa_chain.invoke({"question": enhanced_question})
            else:
                Logger.debug("Normal soru, örnek proje eklenmeyecek")
                result = self.qa_chain.invoke({"question": question})
            
            return result["answer"]
            
        except Exception as e:
            Logger.error(f"Soru işlenirken hata: {e}")
            return f"Üzgünüm, sorunuzu işlerken bir hata oluştu: {str(e)}"
    
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