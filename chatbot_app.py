import streamlit as st
import os
from datetime import datetime
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import RAW_CSV, PERSIST_DIR, COLLECTION, HF_TOKEN, NEWS_API_KEY
from src.data_loader import FinancialDataLoader
from src.embeddings import get_embedder
from src.vectorstore import init_vectorstore, load_vectorstore
from src.llm_client import HFChatModel
from src.advisor_chain import build_fin_advisor_chain
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("❌ Variable d'environnement OPENROUTER_API_KEY non trouvée. Veuillez la configurer.")
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="🤖 Chatbot Conseiller Financier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .stButton > button {
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=True)
def load_resources():
    """Charge les ressources nécessaires (vecteur store, modèle, etc.)"""
    try:
        # Configuration du token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
        
        # Initialisation du modèle
        llm = HFChatModel(
            repo_id="HuggingFaceH4/zephyr-7b-beta", 
            temperature=0.1, 
            token=HF_TOKEN
        )
        
        # Chargement des données historiques (pas utilisé dans notre cas, mais utile pour le contexte)
        loader = FinancialDataLoader(RAW_CSV)
        entries = loader.load()
        docs = []
        
        for e in entries:
            prompt = (f"I'm a {e['age']}-year-old {e['gender']} looking to invest in {e['Avenue']} "
                      f"for {e['Purpose']} over the next {e['Duration']}.")
            response = e.get('recommendation', 'No recommendation')
            docs.append(Document(page_content=f"Prompt: {prompt}\nResponse: {response}"))
        
        # Découpage en chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []
        for d in docs:
            for txt in splitter.split_text(d.page_content):
                chunks.append(Document(page_content=txt))
        
        # Initialisation du vecteur store
        embedder = get_embedder()
        if os.path.exists(PERSIST_DIR):
            store = load_vectorstore(embedder, PERSIST_DIR, COLLECTION)
        else:
            store = init_vectorstore(chunks, embedder, PERSIST_DIR, COLLECTION)
        
        retriever = store.as_retriever(search_kwargs={"k": 5})
        
        return llm, retriever
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des ressources: {str(e)}")
        return None, None

def display_chat_message(message, is_user=True):
    #Affiche un message dans le chat avec un style personnalisé
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 Vous:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Conseiller:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)

def ask_openrouter(prompt, model="mistralai/mistral-7b-instruct", api_key=None, max_tokens=512, temperature=0.1):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Erreur API OpenRouter: {response.status_code} - {response.text}"

def main():
    st.title("🤖 Chatbot Conseiller Financier")
    st.markdown("---")
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("ℹ️ À propos")
        st.markdown("""
        Ce chatbot utilise l'IA pour vous donner des conseils financiers personnalisés.
        
        **Fonctionnalités:**
        - 💡 Conseils d'investissement
        - 📊 Analyse financière
        - 🎯 Recommandations personnalisées
        - 📈 Stratégies financières
        """)
        
        st.markdown("---")
        st.header("⚙️ Paramètres")
        temperature = st.slider("Créativité du modèle", 0.0, 1.0, 0.1, 0.1)
        max_tokens = st.slider("Longueur max de réponse", 100, 1000, 512, 50)
      
        st.markdown("---")
        st.header("📊 Statistiques")
        if 'message_count' in st.session_state:
            st.metric("Messages échangés", st.session_state.message_count)
    
    # Initialisation de la session
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0
    if 'pending_user_input' not in st.session_state:
        st.session_state.pending_user_input = ""
    if 'reset_input' not in st.session_state:
        st.session_state.reset_input = False
    
    # Chargement des ressources
    with st.spinner("Chargement du modèle et des données..."):
        llm, retriever = load_resources()
    
    if llm is None or retriever is None:
        st.error("❌ Impossible de charger les ressources. Vérifiez votre token Hugging Face.")
        return
    
    # Zone de chat
    st.subheader("💬 Conversation")
    
    # Affichage de l'historique des messages
    for message in st.session_state.messages:
        display_chat_message(message["content"], message["is_user"])

    # Affiche le spinner si une question est en attente de réponse
    if st.session_state.pending_user_input:
        with st.spinner("🤔 Le conseiller réfléchit..."):
            pass  # Le spinner s'affiche

    # Zone d'input toujours visible sous le spinner
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            label="Votre question",
            placeholder="Poser une question Financière...",
            key="user_input",
            value="" if st.session_state.reset_input else st.session_state.get("user_input", "")
        )
    with col2:
        send_button = st.button("🚀 Envoyer", use_container_width=True)

    # Gestion de l'envoi
    if send_button and user_input.strip():
        st.session_state.messages.append({"content": user_input, "is_user": True})
        st.session_state.pending_user_input = user_input
        st.session_state.reset_input = True
        st.rerun()

    # Bouton pour effacer l'historique
    if st.session_state.messages:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.rerun()

    # Génération de la réponse
    if st.session_state.pending_user_input:
        try:
            context_prompt = f"""
            Tu es un conseiller financier expert. Réponds à la question suivante de manière claire, 
            professionnelle et utile. Donne des conseils pratiques et des explications détaillées.
            Reponds uniquement dans le contexte de la question posée et dans la langue de la question posée.

            Question: {st.session_state.pending_user_input}

            Réponse:
            """
            assistant_response = ask_openrouter(
                context_prompt,
                model="mistralai/mistral-7b-instruct",
                api_key=OPENROUTER_API_KEY,
                max_tokens=max_tokens,
                temperature=temperature
            )
            st.session_state.messages.append({"content": assistant_response, "is_user": False})
        except Exception as e:
            error_msg = f"❌ Erreur lors de la génération de la réponse: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"content": error_msg, "is_user": False})
        st.session_state.pending_user_input = ""
        st.session_state.reset_input = False
        st.rerun()

if __name__ == "__main__":
    main() 