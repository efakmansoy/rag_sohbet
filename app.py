import os
import glob
import streamlit as st
import sys
import pysqlite3

# pysqlite3'ü sistemin varsayılan sqlite3'ü olarak ayarla
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

# Gerekli kütüphaneleri içe aktarın
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores.utils import filter_complex_metadata

# --- RAG Sisteminin Hazırlanması ---
@st.cache_resource
def setup_rag_system():
    # Tüm durum mesajlarını içeren kalıcı bir kapsayıcı oluşturun
    info_container = st.empty()
    
    with info_container.container():
        st.info("Sistem başlatılıyor...")

        db_path = "./chroma_db"
        files_dir = "./files"
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Mevcut veritabanını yüklemeyi dene
        if os.path.exists(db_path) and os.path.isdir(db_path):
            try:
                st.info("Mevcut veritabanı bulunuyor. Yükleniyor...")
                vectorstore = Chroma(
                    collection_name="parent_child_collection",
                    embedding_function=embeddings,
                    persist_directory=db_path
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
                st.success("Veritabanı başarıyla yüklendi.")
                return retriever
            except Exception as e:
                st.warning(f"Veritabanı yüklenirken bir hata oluştu: {e}. Yeniden oluşturuluyor...")
                
        st.info("Veritabanı bulunamadı veya yüklenemedi. Yeni bir veritabanı oluşturuluyor...")
        image_files = glob.glob(os.path.join(files_dir, "*.png")) + glob.glob(os.path.join(files_dir, "*.jpg"))
        # PDF ve resim dosyalarını yükle
        pdf_files = glob.glob(os.path.join(files_dir, "*.pdf"))
        
        all_documents = []
        if image_files:
            for file_path in image_files:
                st.info(f"'{os.path.basename(file_path)}' resim dosyası yükleniyor...")
                # Resim dosyalarından metin çıkarmak için UnstructuredImageLoader kullanın
                loader = UnstructuredImageLoader(file_path)
                all_documents.extend(loader.load())
        if pdf_files:
            for file_path in pdf_files:
                st.info(f"'{os.path.basename(file_path)}' dosyası yükleniyor...")
                # Metin PDF'leri için PyPDFLoader kullanın
                loader = PyPDFLoader(file_path)
                all_documents.extend(loader.load())

        

        # Web sayfasını yükle
        web_url = "https://tubitak.gov.tr/tr/yarismalar/2204-lise-ogrencileri-arastirma-projeleri-yarismasi"
        st.info(f"'{web_url}' adresindeki sayfa yükleniyor...")
        web_loader = WebBaseLoader(web_url)
        all_documents.extend(web_loader.load())

        if not all_documents:
            st.error("Hiçbir belge (PDF veya web sayfası) yüklenemedi. Lütfen dosyalarınızın doğru klasörde olduğundan ve URL'nin doğru olduğundan emin olun.")
            return None

        st.success(f"Toplam {len(all_documents)} sayfa yüklendi.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.split_documents(all_documents)
        
        # Karmaşık meta verileri filtreleyin
        filtered_documents = filter_complex_metadata(split_documents)

        vectorstore = Chroma.from_documents(
            documents=filtered_documents,
            embedding=embeddings,
            collection_name="parent_child_collection",
            persist_directory=db_path
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
        st.success("Veritabanı başarıyla oluşturuldu.")
        return retriever

# --- Streamlit Uygulamasının Ana Bölümü ---
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

retriever = setup_rag_system()
if retriever:
    if st.session_state.qa_chain is None:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        st.session_state.memory = ConversationSummaryMemory(
            llm=st.session_state.llm,
            memory_key="chat_history", 
            return_messages=True
        )
        
        custom_prompt_template = """
Sen, TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması hakkında öğrenci ve danışmanlara yardımcı olan bir asistansın. Görevin, onlara yarışmanın şartnameleri, başvuru ve rapor süreçleri gibi konularda, **sadece verilen belgelerden edindiğin bilgilere dayanarak** rehberlik etmektir.
Eğer verilen bağlamda sorunun cevabı yoksa, elindeki bilgilere göre en mantıklı yanıtı üretmeye çalış. Eğer hiçbir şekilde ilgili bilgi bulunamıyorsa, kibar bir şekilde **"Verilen belgelerde bu konuda spesifik bir bilgi bulunmamaktadır."** şeklinde yanıt ver. Kesinlikle uydurma bilgi verme. Yanıtların profesyonel, anlaşılır ve yarışma konusuna odaklı olsun.

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
            result = st.session_state.qa_chain.invoke({"question": prompt})
            response = result["answer"]
            
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("Proje başlatılamıyor. Lütfen gerekli dosyaların olduğundan emin olun.")


