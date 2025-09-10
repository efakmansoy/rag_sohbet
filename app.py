import os
import glob
import streamlit as st
import sys
import pysqlite3

# SQLite modülünü pysqlite3 ile değiştirme
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

# Gerekli LangChain kütüphanelerini ve transformers'ı içe aktarma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from transformers import pipeline

@st.cache_resource
def setup_rag_system():
    db_path = "./chroma_db"
    files_dir = "./files"
    
    print("Sistem başlatılıyor...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(db_path) and os.path.isdir(db_path):
        try:
            print("Mevcut veritabanı bulunuyor. Yükleniyor...")
            vectorstore = Chroma(
                collection_name="parent_child_collection",
                embedding_function=embeddings,
                persist_directory=db_path
            )
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
            print("Veritabanı başarıyla yüklendi.")
            return retriever
        except Exception as e:
            print(f"Veritabanı yüklenirken bir hata oluştu: {e}. Yeniden oluşturuluyor...")
            
    print("Veritabanı bulunamadı veya yüklenemedi. Yeni bir veritabanı oluşturuluyor...")
    
    pdf_files = glob.glob(os.path.join(files_dir, "*.pdf"))
    all_documents = []
    if pdf_files:
        for file_path in pdf_files:
            print(f"'{os.path.basename(file_path)}' dosyası yükleniyor...")
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())

    web_url = "https://tubitak.gov.tr/tr/yarismalar/2204-lise-ogrencileri-arastirma-projeleri-yarismasi"
    print(f"'{web_url}' adresindeki sayfa yükleniyor...")
    web_loader = WebBaseLoader(web_url)
    all_documents.extend(web_loader.load())

    if not all_documents:
        print("Hiçbir belge (PDF veya web sayfası) yüklenemedi. Lütfen dosyalarınızın doğru klasörde olduğundan ve URL'nin doğru olduğundan emin olun.")
        return None

    print(f"Toplam {len(all_documents)} sayfa yüklendi.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(all_documents)

    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        collection_name="parent_child_collection",
        persist_directory=db_path
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    st.success("Veritabanı başarıyla oluşturuldu.")
    return retriever

st.set_page_config(page_title="2204-A Yarışma Asistanı", layout="wide")
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
        try:
            # Model, transformers pipeline ile yükleniyor
            pipe = pipeline(
                "text-generation", 
                model="deepseek-ai/DeepSeek-V3.1",
                trust_remote_code=True,
                device_map="auto"
            )
            # LangChain için HuggingFacePipeline sarmalayıcısı kullanılıyor
            st.session_state.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            st.error(f"Model yüklenirken bir hata oluştu: {e}")
            st.warning("Lütfen yeterli VRAM ve RAM'e sahip olduğunuzdan emin olun. Bu modeller çok büyük olabilir.")
            st.stop()

        st.session_state.memory = ConversationSummaryMemory(
            llm=st.session_state.llm,
            memory_key="chat_history", 
            return_messages=True
        )
        
        custom_prompt_template = """
Sen, TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması hakkında öğrenci ve danışmanlara yardımcı olan bir asistansın. Görevin, onlara yarışmanın şartnameleri, başvuru ve rapor süreçleri gibi konularda, **sadece verilen belgelerden edindiğin bilgilere dayanarak** rehberlik etmektir.
Eğer verilen bağlamda sorunun cevabı yoksa, elindeki bilgilere göre en mantıklı yanıtı üretmeye çalış. Kesinlikle uydurma bilgi verme. Yanıtların profesyonel, anlaşılır ve yarışma konusuna odaklı olsun.
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
            result = st.session_state.qa_chain.invoke({"question": prompt})
            response = result["answer"]
            
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("Proje başlatılamıyor. Lütfen gerekli dosyaların ve modelin doğru yüklendiğinden emin olun.")

