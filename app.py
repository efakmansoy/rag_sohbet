import os
import glob
import streamlit as st
import sys
import pysqlite3 # <<< Bu satırı ekledik

sys.modules["sqlite3"] = sys.modules["pysqlite3"]

# Yeni ve güncellenmiş import'lar
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Ek import'lar
from chromadb.config import Settings

# Diğer gerekli import'lar
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# --- RAG Sisteminin Hazırlanması ---
# Streamlit Cloud'da her oturum için yeniden çalışır
@st.cache_resource
def setup_rag_system():
    """
    RAG sistemini hazırlar ve belleğe alır.
    Bulut dağıtımı için veritabanını bellekte tutar.
    """
    st.info("Bulut ortamı için veritabanı bellekte yeniden oluşturuluyor...")
    files_dir = "./files"
    
    if not os.path.exists(files_dir):
        st.error(f"'{files_dir}' klasörü bulunamadı. Lütfen bu klasörü oluşturun ve içine PDF dosyalarınızı yerleştirin.")
        return None
    
    pdf_files = glob.glob(os.path.join(files_dir, "*.pdf"))
    
    if not pdf_files:
        st.error(f"'{files_dir}' klasöründe hiçbir PDF dosyası bulunamadı. Lütfen PDF dosyalarınızı bu klasöre yerleştirin.")
        return None
    
    all_documents = []
    for file_path in pdf_files:
        st.info(f"'{os.path.basename(file_path)}' dosyası yükleniyor...")
        loader = PyPDFLoader(file_path)
        all_documents.extend(loader.load())

    st.success(f"Toplam {len(all_documents)} sayfa yüklendi.")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Bellekte çalışan Chroma veritabanı
    vectorstore = Chroma(
        collection_name="parent_child_collection",
        embedding_function=embeddings,
        client_settings=Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=None,
        )
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )
    retriever.add_documents(all_documents)
    st.success("Gelişmiş indeksleme ve bellek tabanlı geri alma sistemi hazır.")

    return retriever

# --- Streamlit Arayüzü ---
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
            temperature=0.4,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        st.session_state.memory = ConversationSummaryMemory(
            llm=st.session_state.llm,
            memory_key="chat_history", 
            return_messages=True
        )
        
        custom_prompt_template = """
Sen 2204-A yarışmasına hazırlanan öğrenci ve danışman öğretmenlere yardımcı olan bir asistansın. Öğrenci ve öğretmenlere neyi nasıl yapmaları gerektiği konusunda rehberlik ediyorsun.
Aşağıdaki konuşma geçmişini ve bağlamı kullanarak, en son kullanıcı sorusuna kısa ve net bir yanıt ver.
Cevabını doğrudan bağlamdaki bilgilerden al. Eğer bağlamda sorunun cevabı yoksa, "Bu konuda şartnamede net bir bilgi bulunmamaktadır." şeklinde yanıt ver.
Kesinlikle bağlamda olmayan bir bilgi uydurma.

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
            retrieved_docs = retriever.invoke(prompt)
            total_retrieved_length = sum(len(doc.page_content) for doc in retrieved_docs)

            if total_retrieved_length < 100:
                general_llm_response = st.session_state.llm.invoke(prompt)
                response = general_llm_response.content
                st.session_state.memory.save_context({"input": prompt}, {"output": response})
            else:
                result = st.session_state.qa_chain.invoke({"question": prompt})
                response = result["answer"]
            
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("Proje başlatılamıyor. Lütfen gerekli dosyaların ve Ollama'nın çalıştığından emin olun.")
