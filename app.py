import os
import glob
import streamlit as st
import sys
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document

@st.cache_resource
def load_classifier_model():
    """Fikir klasifikasyon modelini yükle"""
    model_path = "./model"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Klasifikasyon modeli yüklenirken hata oluştu: {e}")
        return None, None

def classify_question(question, tokenizer, model):
    """Sorunun fikir içerip içermediğini sınıflandır"""
    if tokenizer is None or model is None:
        return 0
    
    try:
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            
        return predicted_class
    except Exception as e:
        st.error(f"Sınıflandırma sırasında hata oluştu: {e}")
        return 0

def get_example_projects(dataset_db_path, embeddings, question, k=4):
    """Dataset'ten örnek projeler getir"""
    try:
        print(f"[DEBUG] Dataset'ten örnek projeler getiriliyor... Yol: {dataset_db_path}")
        dataset_vectorstore = Chroma(
            collection_name="dataset_collection",
            embedding_function=embeddings,
            persist_directory=dataset_db_path
        )
        
        print(f"[DEBUG] Similarity search yapılıyor, k={k}")
        # Soruya benzer projeleri bul
        docs = dataset_vectorstore.similarity_search(question, k=k)
        
        if docs:
            print(f"[DEBUG] {len(docs)} benzer proje bulundu")
            examples = "\n\nÖrnek Projeler:\n"
            for i, doc in enumerate(docs, 1):
                examples += f"{i}. {doc.page_content}\n---\n"
            print(f"[DEBUG] Örnek projeler hazırlandı, toplam uzunluk: {len(examples)} karakter")
            return examples
        else:
            print("[DEBUG] Hiç benzer proje bulunamadı")
            return ""
    except Exception as e:
        print(f"[DEBUG] Örnek projeler getirilirken hata: {e}")
        st.warning(f"Örnek projeler getirilirken hata oluştu: {e}")
        return ""

@st.cache_resource
def setup_rag_system():
    db_path = "./chroma_db"
    dataset_db_path = "./dataset_chroma_db"
    files_dir = "./files"
    
    embeddings = HuggingFaceEmbeddings(model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
    
    # Ana RAG sistemi (PDF + Web) yükleme
    if os.path.exists(db_path) and os.path.isdir(db_path):
        try:
            vectorstore = Chroma(
                collection_name="parent_child_collection",
                embedding_function=embeddings,
                persist_directory=db_path
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        except Exception as e:
            st.warning(f"Ana veritabanı yüklenirken bir hata oluştu: {e}. Yeniden oluşturuluyor...")
            retriever = create_main_vectorstore(embeddings, files_dir, db_path)
    else:
        retriever = create_main_vectorstore(embeddings, files_dir, db_path)
    
    # Dataset vektör veritabanını yükle/oluştur
    setup_dataset_vectorstore(embeddings, dataset_db_path)
    
    return retriever

def create_main_vectorstore(embeddings, files_dir, db_path):
            
    
    pdf_files = glob.glob(os.path.join(files_dir, "*.pdf"))
    all_documents = []
    if pdf_files:
        for file_path in pdf_files:
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())

    web_url = "https://tubitak.gov.tr/tr/yarismalar/2204-lise-ogrencileri-arastirma-projeleri-yarismasi"
    web_loader = WebBaseLoader(web_url)
    all_documents.extend(web_loader.load())

    if not all_documents:
        st.error("Hiçbir belge (PDF veya web sayfası) yüklenemedi. Lütfen dosyalarınızın doğru klasörde olduğundan ve URL'nin doğru olduğundan emin olun.")
        return None

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(all_documents)

    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        collection_name="parent_child_collection",
        persist_directory=db_path
    )
    # k değeri 15'e çıkarıldı
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    return retriever

def setup_dataset_vectorstore(embeddings, dataset_db_path):
    """Dataset vektör veritabanını yükle/oluştur"""
    print(f"[DEBUG] Dataset veritabanı kontrol ediliyor: {dataset_db_path}")
    
    if os.path.exists(dataset_db_path) and os.path.isdir(dataset_db_path):
        try:
            print("[DEBUG] Mevcut dataset veritabanı bulundu, yükleniyor...")
            # Mevcut dataset veritabanını yükle
            dataset_vectorstore = Chroma(
                collection_name="dataset_collection",
                embedding_function=embeddings,
                persist_directory=dataset_db_path
            )
            print("[DEBUG] Dataset vektör veritabanı başarıyla yüklendi!")
            st.info("Dataset vektör veritabanı başarıyla yüklendi.")
            return dataset_vectorstore
        except Exception as e:
            print(f"[DEBUG] Dataset veritabanı yüklenirken hata: {e}")
            st.warning(f"Dataset veritabanı yüklenirken hata oluştu: {e}. Yeniden oluşturuluyor...")
    
    # Dataset veritabanını oluştur
    print("[DEBUG] Dataset veritabanı oluşturuluyor...")
    try:
        print("[DEBUG] HuggingFace'den dataset yükleniyor...")
        st.info("Dataset yükleniyor...")
        dataset = load_dataset("Q-bert/Custom-2204")
        print(f"[DEBUG] Dataset yüklendi! Train boyutu: {len(dataset['train'])}")
        
        # Dataset yapısını kontrol et
        if len(dataset['train']) > 0:
            first_item = dataset['train'][0]
            print(f"[DEBUG] Dataset sütunları: {list(first_item.keys())}")
            st.info(f"Dataset sütunları: {list(first_item.keys())}")
        
        print("[DEBUG] Dataset'i Document formatına dönüştürülüyor...")
        # Dataset'i Document formatına dönüştür
        documents = []
        for i, item in enumerate(dataset['train']):
            if i % 1000 == 0:  # Her 1000 projede bir rapor
                print(f"[DEBUG] İşlenen proje sayısı: {i}/{len(dataset['train'])}")
            
            # Sütun adlarını kontrol et ve güvenli erişim
            try:
                # Farklı olası sütun adlarını dene
                kategori = item.get('Kategori İsmi', item.get('kategori', item.get('Kategori', 'Bilinmiyor')))
                proje_alani = item.get('Proje Alanı', item.get('proje_alani', item.get('Proje Alani', 'Bilinmiyor')))
                proje_ismi = item.get('Proje İsmi', item.get('proje_ismi', item.get('Proje Ismi', 'Bilinmiyor')))
                ozet = item.get('Özet', item.get('ozet', item.get('Ozet', 'Özet bulunamadı')))
                
                # Proje bilgilerini birleştir
                content = f"""
Kategori İsmi: {kategori}
Proje Alanı: {proje_alani}
Proje İsmi: {proje_ismi}
Özet: {ozet}
"""
                
                doc = Document(
                    page_content=content.strip(),
                    metadata={
                        "kategori": kategori,
                        "proje_alani": proje_alani,
                        "proje_ismi": proje_ismi,
                        "source": "dataset"
                    }
                )
                documents.append(doc)
            except Exception as item_error:
                print(f"[DEBUG] Proje işlenirken hata (item {i}): {item_error}")
                st.warning(f"Bir proje işlenirken hata oluştu: {item_error}")
                continue
        
        print(f"[DEBUG] Toplam {len(documents)} proje Document formatına dönüştürüldü")
        
        # Text splitter uygula
        print("[DEBUG] Text splitter uygulanıyor...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        split_documents = text_splitter.split_documents(documents)
        print(f"[DEBUG] {len(split_documents)} chunk oluşturuldu")
        
        # Vektör veritabanını oluştur
        print("[DEBUG] ChromaDB vektör veritabanı oluşturuluyor...")
        dataset_vectorstore = Chroma.from_documents(
            documents=split_documents,
            embedding=embeddings,
            collection_name="dataset_collection",
            persist_directory=dataset_db_path
        )
        
        print(f"[DEBUG] Dataset başarıyla işlendi! {len(documents)} proje, {len(split_documents)} chunk eklendi")
        st.success(f"Dataset başarıyla işlendi! {len(documents)} proje eklendi.")
        return dataset_vectorstore
        
    except Exception as e:
        print(f"[DEBUG] Dataset yüklenirken ana hata: {e}")
        st.error(f"Dataset yüklenirken hata oluştu: {e}")
        return None

st.set_page_config(page_title="Yarışma Asistanı", layout="wide")
st.title("🏆 Yarışma Asistanı")
st.write("Şartnameler ve raporlar hakkında sorularınızı sorun.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "memory" not in st.session_state:
    st.session_state.memory = None
if "classifier_tokenizer" not in st.session_state:
    st.session_state.classifier_tokenizer = None
if "classifier_model" not in st.session_state:
    st.session_state.classifier_model = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Classifier'ı yükle
if st.session_state.classifier_tokenizer is None:
    st.session_state.classifier_tokenizer, st.session_state.classifier_model = load_classifier_model()

retriever = setup_rag_system()
if retriever:
    if st.session_state.qa_chain is None:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.5,
            google_api_key="AIzaSyBUs9xbTmXp2-En0tAF0kks0wWkwxUdgIs"
        )
        st.session_state.memory = ConversationSummaryMemory(
            llm=st.session_state.llm,
            memory_key="chat_history", 
            return_messages=True
        )
        
        # Güncellenmiş Prompt
        custom_prompt_template = """
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
        CUSTOM_PROMPT = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["chat_history", "context", "question"]
        )
        
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=st.session_state.llm
        )
        
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            retriever=multi_query_retriever,
            memory=st.session_state.memory,
            rephrase_question=False,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT} 
        )

    if prompt := st.chat_input("Buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Cevap bekleniyor..."):
            print(f"[DEBUG] Kullanıcı sorusu: {prompt}")
            
            # Soruyu sınıflandır
            print("[DEBUG] Soru sınıflandırılıyor...")
            classification = classify_question(
                prompt, 
                st.session_state.classifier_tokenizer, 
                st.session_state.classifier_model
            )
            print(f"[DEBUG] Sınıflandırma sonucu: {classification}")
            
            # Eğer fikir içeren bir soru ise (1), dataset'ten örnekler ekle
            if classification == 1:
                print("[DEBUG] Fikir içeren soru tespit edildi, örnek projeler getiriliyor...")
                embeddings = HuggingFaceEmbeddings(model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
                example_projects = get_example_projects("./dataset_chroma_db", embeddings, prompt, k=3)
                
                # Prompt'u örneklerle genişlet
                enhanced_prompt = prompt + example_projects
                print(f"[DEBUG] Prompt örnek projelerle genişletildi. Yeni uzunluk: {len(enhanced_prompt)} karakter")
                result = st.session_state.qa_chain.invoke({"question": enhanced_prompt})
            else:
                print("[DEBUG] Normal soru, örnek proje eklenmeyecek")
                # Normal prompt ile devam et
                result = st.session_state.qa_chain.invoke({"question": prompt})
            
            response = result["answer"]
            
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("Proje başlatılamıyor. Lütfen gerekli dosyaların çalıştığından emin olun.")


