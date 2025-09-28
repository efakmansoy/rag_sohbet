"""
TÜBİTAK 2204-A Yarışma Asistanı - Streamlit Arayüzü
RAG Core sistemini kullanan web arayüzü
"""

# Third Party Imports
import streamlit as st

# Local Imports
from rag_core import RAGSystem, create_rag_system, Logger


class StreamlitChatInterface:
    """Streamlit tabanlı sohbet arayüzü"""
    
    def __init__(self):
        """StreamlitChatInterface başlatıcı"""
        self.rag_system = None
        self._initialize_session_state()
        self._setup_rag_system()
    
    def _initialize_session_state(self):
        """Session state değişkenlerini başlat"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "rag_system" not in st.session_state:
            st.session_state.rag_system = None
        if "system_ready" not in st.session_state:
            st.session_state.system_ready = False
    
    @st.cache_resource
    def _setup_rag_system(_self):
        """RAG sistemini yükle (Cached)"""
        try:
            with st.spinner("RAG sistemi başlatılıyor... Bu işlem birkaç dakika sürebilir."):
                rag_system = create_rag_system()
                if rag_system and rag_system.qa_chain:
                    st.session_state.rag_system = rag_system
                    st.session_state.system_ready = True
                    return rag_system
                else:
                    st.error("❌ RAG sistemi başlatılamadı!")
                    return None
        except Exception as e:
            st.error(f"❌ RAG sistemi başlatılırken hata oluştu: {str(e)}")
            Logger.error(f"RAG sistemi başlatılamadı: {e}")
            return None
    
    def display_chat_history(self):
        """Sohbet geçmişini göster"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def process_user_input(self, user_input: str) -> str:
        """Kullanıcı girdisini işle"""
        if not st.session_state.system_ready or not st.session_state.rag_system:
            return "❌ Sistem henüz hazır değil. Lütfen sistemin yüklenmesini bekleyin."
        
        try:
            response = st.session_state.rag_system.process_question(user_input)
            return response
        except Exception as e:
            Logger.error(f"Soru işlenirken hata: {e}")
            return f"❌ Üzgünüm, sorunuzu işlerken bir hata oluştu: {str(e)}"
    
    def show_system_status(self):
        """Sistem durumunu göster"""
        if st.session_state.system_ready:
            st.sidebar.success("🟢 Sistem Aktif")
        else:
            st.sidebar.warning("🟡 Sistem Yükleniyor...")
    
    def show_conversation_controls(self):
        """Konuşma kontrolleri"""
        st.sidebar.markdown("---")
        
        if st.sidebar.button("🗑️ Sohbeti Temizle"):
            st.session_state.messages = []
            if st.session_state.rag_system:
                st.session_state.rag_system.clear_conversation_history()
            st.rerun()
        
        # Konuşma sayısı
        if st.session_state.messages:
            msg_count = len(st.session_state.messages)
            st.sidebar.write(f"💬 Toplam Mesaj: {msg_count}")
    
    def run(self):
        """Ana sohbet döngüsü"""
        # Sistem durumunu göster
        self.show_system_status()
        
        # Konuşma kontrollerini göster
        self.show_conversation_controls()
        
        # Sohbet geçmişini göster
        self.display_chat_history()
        
        # Kullanıcı girdisi
        if user_input := st.chat_input("Buraya sorunuzu yazın..."):
            # Kullanıcı mesajını ekle
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Cevabı işle ve göster
            with st.chat_message("assistant"):
                with st.spinner("Cevap hazırlanıyor..."):
                    response = self.process_user_input(user_input)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


def setup_page():
    """Sayfa konfigürasyonu"""
    st.set_page_config(
        page_title="TÜBİTAK 2204-A Yarışma Asistanı",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Ana başlık
    st.title("🏆 TÜBİTAK 2204-A Yarışma Asistanı")


def main():
    """Ana uygulama fonksiyonu"""
    # Sayfa kurulumu
    setup_page()
    
    # Sohbet arayüzünü başlat ve çalıştır
    chat_interface = StreamlitChatInterface()
    chat_interface.run()


if __name__ == "__main__":
    main()
