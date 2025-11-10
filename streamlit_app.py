import streamlit as st
from datetime import datetime
import json
import os
import base64
from io import BytesIO

# Import RAG pipeline
from rag_pipeline import RAGPipeline

# ------------------- CONFIG -------------------
PDF_FOLDER = "C://iwmi-remote-work//CBE-Chatbot//New folder//cbe//agri and waste water"
INDEX_FILE = "pdf_index_enhanced.pkl"

AZURE_OPENAI_KEY = st.secrets["azure_api_key"]
AZURE_OPENAI_ENDPOINT = "https://iwmi-chat-demo.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"

HF_TOKEN = st.secrets["hf_token"]
DEEPSEEK_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3.1:novita"

CONVERSATION_HISTORY_FILE = "conversation_history.json"

# ------------------- CUSTOM CSS -------------------
def load_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f9fbfc;
        padding: 2rem 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #e6f4ff 0%, #f0f2f5 100%);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #4b5563;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px 16px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stChatMessage.user {
        background-color: #e6f3ff;
        border-left: 4px solid #0a7aff;
    }
    .stChatMessage.assistant {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
    }
    .reference-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px;
        margin: 6px 0;
        font-size: 0.85rem;
    }
    .reference-header {
        font-weight: 600;
        color: #111827;
        margin-bottom: 4px;
    }
    .reference-badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-left: 6px;
    }
    .badge-table { background: #fff7ed; color: #ea580c; }
    .badge-heading { background: #f0fdf4; color: #16a34a; }
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .export-btn {
        background: #3b82f6;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 500;
        width: 100%;
        margin-top: 8px;
    }
    .export-btn:hover {
        background: #2563eb;
    }
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    #MainMenu, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ------------------- SESSION STATE -------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "model" not in st.session_state:
        st.session_state.model = "DeepSeek"
    if "rag_loaded" not in st.session_state:
        st.session_state.rag_loaded = False
    if "is_switching" not in st.session_state:
        st.session_state.is_switching = False

# ------------------- RAG PIPELINE -------------------
@st.cache_resource(show_spinner=False)
def get_rag_pipeline(selected_model: str):
    params = {}
    if selected_model == "GPT-4o-mini":
        params = {
            "llm_type": "azure",
            "azure_key": AZURE_OPENAI_KEY,
            "azure_endpoint": AZURE_OPENAI_ENDPOINT,
            "azure_deployment": AZURE_OPENAI_DEPLOYMENT,
        }
    else:  # DeepSeek
        params = {
            "llm_type": "deepseek",
            "hf_token": HF_TOKEN,
            "deepseek_url": DEEPSEEK_API_URL,
            "deepseek_model": DEEPSEEK_MODEL_NAME,
        }
    return RAGPipeline(
        pdf_folder=PDF_FOLDER,
        index_file=INDEX_FILE,
        model_params=params,
    )

# ------------------- HELPER FUNCTIONS -------------------
def save_conversation():
    try:
        conversation_data = {
            "id": st.session_state.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "total_queries": st.session_state.total_queries
        }
        
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            with open(CONVERSATION_HISTORY_FILE, "r") as f:
                all_conversations = json.load(f)
        else:
            all_conversations = []
        
        all_conversations = [c for c in all_conversations if c["id"] != st.session_state.conversation_id]
        all_conversations.append(conversation_data)
        
        with open(CONVERSATION_HISTORY_FILE, "w") as f:
            json.dump(all_conversations, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")
        return False

def export_conversation_markdown():
    md_content = f"# Conversation Export\n\n"
    md_content += f"**ID:** {st.session_state.conversation_id}\n\n"
    md_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += "---\n\n"
    
    for msg in st.session_state.messages:
        role = "🧑 **User**" if msg["role"] == "user" else "🤖 **Assistant**"
        md_content += f"{role}:\n{msg['content']}\n\n"
    
    return md_content

def export_conversation_pdf():
    md_content = export_conversation_markdown()
    buffer = BytesIO()
    # Simple markdown to PDF conversion placeholder; in production, use markdown2 + fpdf or similar
    buffer.write(md_content.encode())
    return buffer.getvalue()

# ------------------- MAIN APP -------------------
def main():
    st.set_page_config(
        page_title="CBE Chatbot",
        layout="wide",
        page_icon="📚",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    init_session_state()
    
    # Professional loading screen
    if not st.session_state.rag_loaded:
        spinner_text = "Switching Model... Please wait." if st.session_state.is_switching else "Initializing Assistant... Please wait."
        with st.spinner(spinner_text):
            rag = get_rag_pipeline(st.session_state.model)
            if not rag.load_index():
                st.warning("Index not found. Contact admin to build the index.")
                return
            st.session_state.rag_loaded = True
            st.session_state.rag = rag
            st.session_state.is_switching = False
            st.rerun()
    
    rag = st.session_state.rag
    
    with st.sidebar:
        st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("🤖 Model Selection")
        selected_model = st.selectbox(
            "Choose model",
            ["DeepSeek", "GPT-4o-mini"],
            index=0 if st.session_state.model == "DeepSeek" else 1
        )
        if selected_model != st.session_state.model:
            st.toast(f"You have switched the model to {selected_model}")
            st.session_state.model = selected_model
            st.session_state.messages = []  # Clear chat
            st.session_state.total_queries = 0  # Reset queries
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # New session ID
            st.session_state.rag_loaded = False  # Force reload
            st.session_state.is_switching = True  # Set switching flag
            st.rerun()
        
        st.divider()
        
        st.subheader("💾 Export Chat")
        if st.button("Export as MD", key="export_md", help="Download conversation as Markdown"):
            md_content = export_conversation_markdown()
            st.download_button(
                label="Download MD",
                data=md_content,
                file_name=f"conversation_{st.session_state.conversation_id}.md",
                mime="text/markdown",
            )
        
        if st.button("Export as PDF", key="export_pdf", help="Download conversation as PDF"):
            pdf_content = export_conversation_pdf()
            b64 = base64.b64encode(pdf_content).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="conversation_{st.session_state.conversation_id}.pdf">Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        st.divider()
        
        st.caption(f"📅 Session: {st.session_state.conversation_id}")
    
    st.markdown('<div class="header-title">📚 PDF Knowledge Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Ask questions about agricultural and wastewater management</div>', unsafe_allow_html=True)
    
    if not rag.documents:
        st.warning("⚠️ Sources not loaded. Please contact support.")
        return
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if "references" in message and message["references"]:
                    with st.expander(f"📖 View {len(message['references'])} sources"):
                        for i, doc in enumerate(message["references"], 1):
                            src = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "?")
                            doc_type = doc.metadata.get("type", "text")
                            
                            type_badge = ""
                            badge_class = ""
                            if doc_type == "table":
                                type_badge = "🔢 TABLE"
                                badge_class = "badge-table"
                            elif doc_type == "heading":
                                type_badge = "📌 HEADING"
                                badge_class = "badge-heading"
                            
                            st.markdown(f"""
                            <div class="reference-card">
                                <div class="reference-header">{i}. {src} (Page {page}) <span class="reference-badge {badge_class}">{type_badge}</span></div>
                                <span style="color:#4b5563;">{doc.page_content[:300]}...</span>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing sources..."):
                try:
                    answer, references = rag.query(prompt)
                    
                    st.markdown(answer)
                    
                    if references:
                        with st.expander(f"📖 View {len(references)} sources"):
                            for i, doc in enumerate(references, 1):
                                src = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", "?")
                                doc_type = doc.metadata.get("type", "text")
                                
                                type_badge = ""
                                badge_class = ""
                                if doc_type == "table":
                                    type_badge = "🔢 TABLE"
                                    badge_class = "badge-table"
                                elif doc_type == "heading":
                                    type_badge = "📌 HEADING"
                                    badge_class = "badge-heading"
                                
                                st.markdown(f"""
                                <div class="reference-card">
                                    <div class="reference-header">{i}. {src} (Page {page}) <span class="reference-badge {badge_class}">{type_badge}</span></div>
                                    <span style="color:#4b5563;">{doc.page_content[:300]}...</span>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "references": references
                    })
                    st.session_state.total_queries += 1
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "references": []
                    })

if __name__ == "__main__":
    main()