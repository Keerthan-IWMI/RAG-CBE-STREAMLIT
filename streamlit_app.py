import streamlit as st
from datetime import datetime
import json
import os
from io import BytesIO
from fpdf import FPDF
import hashlib
import requests #for Whisper
import base64
from streamlit_chat_widget import chat_input_widget
from streamlit_float import float_init


# Import RAG pipeline
from rag_pipeline import RAGPipeline
from google_auth import check_google_auth  # Import the auth function

# ------------------- CONFIG -------------------
PDF_FOLDER = "C://iwmi-remote-work//CBE-Chatbot//New folder//cbe//agri and waste water"
INDEX_FILE = "pdf_index_enhanced.pkl"

AZURE_OPENAI_KEY = st.secrets["azure_api_key"]
AZURE_OPENAI_ENDPOINT = "https://iwmi-chat-demo.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"

HF_TOKEN = st.secrets["hf_token"]
DEEPSEEK_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3.1:novita"

CHAT_HISTORY_DIR = "chat_histories"

# Professional Color Palette
PRIMARY_COLOR = "#0F766E"  # Teal
SECONDARY_COLOR = "#06B6D4"  # Cyan
ACCENT_COLOR = "#10B981"  # Emerald
BACKGROUND_LIGHT = "#F0FDFA"  # Light teal
TEXT_PRIMARY = "#0F172A"  # Slate
TEXT_SECONDARY = "#475569"  # Slate gray

# =============== CHAT HISTORY MANAGEMENT ===============

def get_chat_history_file(email: str) -> str:
    """Generate chat history file path based on user email"""
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)
    
    # Create a safe filename from email
    safe_email = hashlib.md5(email.encode()).hexdigest()
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_email}_chat.json")

def save_chat_history(email: str, messages: list, total_queries: int, model: str):
    """Save chat history to JSON file for specific user"""
    try:
        file_path = get_chat_history_file(email)
        
        # Convert messages to JSON-serializable format
        serializable_messages = []
        for msg in messages:
            msg_copy = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            # Convert Document-like objects or dicts to dictionaries
            refs = []
            if "references" in msg and msg["references"]:
                for doc in msg["references"]:
                    try:
                        if isinstance(doc, dict):
                            # Already a dict; copy safe keys
                            page_content = doc.get("page_content", "")
                            metadata = doc.get("metadata", {})
                        else:
                            # Assume object with page_content and metadata attributes
                            page_content = getattr(doc, "page_content", "")
                            metadata = getattr(doc, "metadata", {})
                        refs.append({"page_content": page_content, "metadata": metadata})
                    except Exception:
                        # Fallback: stringify the doc to avoid failing the entire save
                        try:
                            refs.append({"page_content": str(doc), "metadata": {}})
                        except Exception:
                            refs.append({"page_content": "", "metadata": {}})
            msg_copy["references"] = refs
            
            serializable_messages.append(msg_copy)
        
        chat_data = {
            "user_email": email,
            "timestamp": datetime.now().isoformat(),
            "messages": serializable_messages,
            "total_queries": total_queries,
            "model": model
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to save chat history: {e}")
        return False

def load_chat_history(email: str) -> dict:
    """Load chat history from JSON file for specific user"""
    try:
        file_path = get_chat_history_file(email)
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            chat_data = json.load(f)
        
        return chat_data
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return None

def delete_chat_history(email: str) -> bool:
    """Delete chat history for specific user"""
    try:
        file_path = get_chat_history_file(email)
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting chat history: {e}")
        return False

# ------------------- CUSTOM CSS -------------------

def load_custom_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .main {{
        background: linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #ECFDF5 50%, #F0F9FF 100%);
        padding: 0;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #ECFDF5 50%, #F0F9FF 100%);
    }}
    
    /* Header Styling */
    .header-container {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 8px 32px rgba(15, 118, 110, 0.15);
        margin-bottom: 2rem;
    }}
    
    .brand-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }}
    
    .brand-icon {{
        font-size: 3rem;
        filter: drop-shadow(0 4px 8px rgba(255,255,255,0.2));
    }}
    
    .header-title {{
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    .header-subtitle {{
        color: rgba(255, 255, 255, 0.95);
        text-align: center;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
        line-height: 1.6;
    }}
    
    .iwmi-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.4rem 1rem;
        border-radius: 20px;
        color: white;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }}
    
    /* Chat Messages */
    .stChatMessage {{
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }}
    
    .stChatMessage:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }}
    
    [data-testid="stChatMessageContent"] {{
        background-color: transparent !important;
    }}
    
    /* User Messages */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 4px solid {SECONDARY_COLOR};
    }}
    
    /* Assistant Messages */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        background: linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #ECFDF5 100%);
        border-left: 4px solid {ACCENT_COLOR};
    }}
    
    /* Reference Cards */
    .reference-card {{
        background: white;
        border: 1.5px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    
    .reference-card:hover {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.1);
        transform: translateY(-2px);
    }}
    
    .reference-header {{
        font-weight: 600;
        color: {TEXT_PRIMARY};
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .reference-content {{
        color: {TEXT_SECONDARY};
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }}
    
    .reference-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .badge-table {{ 
        background: #FEF3C7; 
        color: #D97706;
        border: 1px solid #FCD34D;
    }}
    
    .badge-heading {{ 
        background: #D1FAE5; 
        color: #059669;
        border: 1px solid #6EE7B7;
    }}
    
    .badge-text {{
        background: #E0E7FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #F8FAFC 0%, white 100%);
        border-right: 2px solid #E2E8F0;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 0rem;
    }}
    
    .sidebar-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {PRIMARY_COLOR};
        margin-bottom: 0rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .sidebar-section {{
        background: white;
        padding: 0rem;
        border-radius: 10px;
        margin-bottom: 0rem;
        border: 1px solid #E2E8F0;
    }}
    
    /* Buttons */
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        color: white;
        border: none;
        padding: 0rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.2);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(15, 118, 110, 0.3);
    }}
    
    .stDownloadButton > button {{
        width: 100%;
        background: white;
        color: {PRIMARY_COLOR};
        border: 2px solid {PRIMARY_COLOR};
        padding: 0.6rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.3s ease;
    }}
    
    .stDownloadButton > button:hover {{
        background: {PRIMARY_COLOR};
        color: white;
    }}
    
    /* Input Styling */
    .stChatInput {{
        border-radius: 12px;
    }}
    
    .stChatInput > div {{
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        transition: all 0.2s ease;
    }}
    
    .stChatInput > div:focus-within {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.1);
    }}
    
    /* Expander Styling */
    .streamlit-expanderHeader {{
        background: linear-gradient(135deg, #F8FAFC 0%, white 100%);
        border-radius: 10px;
        font-weight: 600;
        color: {PRIMARY_COLOR};
        border: 1px solid #E2E8F0;
    }}
    
    .streamlit-expanderHeader:hover {{
        background: linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #F8FAFC 100%);
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-top-color: {PRIMARY_COLOR} !important;
    }}
    
    /* Info Box */
    .info-box {{
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 4px solid {SECONDARY_COLOR};
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: {TEXT_PRIMARY};
    }}
    
    .warning-box {{
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: {TEXT_PRIMARY};
    }}
    
    /* Session Info */
    .session-info {{
        background: white;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        font-size: 0.85rem;
        color: {TEXT_SECONDARY};
        border: 1px solid #E2E8F0;
    }}
    
    .stat-box {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #F1F5F9;
    }}
    
    .stat-box:last-child {{
        border-bottom: none;
    }}
    
    .stat-label {{
        color: {TEXT_SECONDARY};
        font-weight: 500;
    }}
    
    .stat-value {{
        color: {PRIMARY_COLOR};
        font-weight: 700;
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    .stDeployButton {{
        display: none !important;
        visibility: hidden !important;
    }}
    
    [data-testid="stSidebarNav"] {{
        display: block !important;
    }}
    
    button[kind="header"] {{
        display: block !important;
        visibility: visible !important;
    }}
    
    [data-testid="collapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        color: {PRIMARY_COLOR} !important;
        background: white !important;
        border: 2px solid {PRIMARY_COLOR} !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        margin: 1rem !important;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.2) !important;
    }}
    
    [data-testid="collapsedControl"]:hover {{
        background: {BACKGROUND_LIGHT} !important;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.3) !important;
    }}
    
    header {{
        visibility: visible !important;
    }}
    
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    
    section[data-testid="stSidebar"] button[kind="header"] {{
        color: {PRIMARY_COLOR} !important;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: #F1F5F9;
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {PRIMARY_COLOR};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {SECONDARY_COLOR};
    }}
    
    /* Profile Card Styling - Google-like */
    .profile-card {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.2);
        position: relative;
        overflow: hidden;
    }}
    
    .profile-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}
    
    .profile-image-wrapper {{
        position: relative;
        width: 70px;
        height: 70px;
        margin: 0 auto 0.75rem;
        border: 3px solid white;
        border-radius: 50%;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%);
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .profile-initial {{
        font-size: 2rem;
        font-weight: 600;
        color: {PRIMARY_COLOR};
        text-transform: uppercase;
        user-select: none;
    }}
    
    .profile-image {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }}
    
    .profile-info h3 {{
        margin: 0.5rem 0 0.25rem 0;
        color: white;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: -0.3px;
    }}
    
    .profile-info p {{
        margin: 0 0 0.75rem 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.75rem;
        font-weight: 500;
        word-break: break-all;
    }}
    
    /* Logout Button Styling */
    div[data-testid="stVerticalBlock"] > div:has(button[key="logout_btn"]) {{
        margin-top: -0.75rem !important;
    }}
    
    button[key="logout_btn"] {{
        width: 100%;
        background: rgba(255, 255, 255, 0.95) !important;
        color: {PRIMARY_COLOR} !important;
        border: none !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
        font-family: 'Inter', sans-serif !important;
    }}
    
    button[key="logout_btn"]:hover {{
        background: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }}
    
    button[key="logout_btn"]:active {{
        transform: translateY(0) !important;
    }}
    
    /* Sidebar Divider */
    .sidebar-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
        margin: 0.5rem 0;
        border: none;
    }}
    
    .control-panel-header {{
        font-size: 0.7rem;
        font-weight: 700;
        color: {TEXT_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 0.5rem 0 0.35rem 0;
        margin: 0;
    }}
    
    .section-content {{
        margin-bottom: 0.5rem;
    }}

    .reference-card.reference-highlight {{
        box-shadow: 0 0 0 2px #0F766E, 0 0 18px rgba(15,118,110,0.5);
        transition: box-shadow 0.3s ease;
    }}
    </style>
    """, unsafe_allow_html=True)

# ------------------- SESSION STATE -------------------

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
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

def get_user_initial(name: str) -> str:
    """Get the first letter of the user's name"""
    if name:
        return name[0].upper()
    return "U"

def transcribe_audio(audio_bytes):
    """Transcribe audio using Hugging Face Whisper API"""
    if not audio_bytes:
        return None
        
    API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "audio/wav" 
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=audio_bytes)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("text", "").strip()
        else:
            st.error(f"Transcription failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to transcription service: {e}")
        return None

def clean_text_for_pdf(text):
    """Clean text to remove problematic characters for latin-1 encoding"""
    replacements = {
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Ellipsis
        '\u2022': '*',  # Bullet
        '\u00a0': ' ',  # Non-breaking space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.encode('latin-1', errors='ignore').decode('latin-1')

def export_conversation_pdf():
    """Export conversation as PDF with proper formatting and encoding"""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 20)
        pdf.set_text_color(15, 118, 110)
        pdf.cell(0, 10, 'CircularIQ Conversation Export', 0, 1, 'C')
        pdf.ln(5)
        
        # Metadata
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(0, 6, f'User: {st.session_state.get("user_email", "Unknown")}', 0, 1)
        pdf.cell(0, 6, f'Date: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 0, 1)
        pdf.cell(0, 6, f'Model: {st.session_state.model}', 0, 1)
        pdf.cell(0, 6, f'Total Queries: {st.session_state.total_queries}', 0, 1)
        pdf.ln(10)
        
        # Conversation
        for i, msg in enumerate(st.session_state.messages, 1):
            # Role header
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(15, 118, 110)
            role_text = f"User (Message {i})" if msg["role"] == "user" else f"CircularIQ Assistant (Message {i})"
            pdf.cell(0, 8, role_text, 0, 1)
            
            # Message content - clean the text
            pdf.set_font('Arial', '', 10)
            pdf.set_text_color(0, 0, 0)
            cleaned_content = clean_text_for_pdf(msg['content'])
            pdf.multi_cell(0, 6, cleaned_content)
            pdf.ln(3)
            
            # References
            if "references" in msg and msg["references"]:
                pdf.set_font('Arial', 'I', 9)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 6, f'Sources: {len(msg["references"])} documents referenced', 0, 1)
                
                for j, doc in enumerate(msg["references"][:3], 1):
                    # Handle both dict and Document objects
                    if isinstance(doc, dict):
                        src = clean_text_for_pdf(doc.get("metadata", {}).get("source", "Unknown"))
                        page = doc.get("metadata", {}).get("page", "?")
                    else:
                        src = clean_text_for_pdf(doc.metadata.get("source", "Unknown"))
                        page = doc.metadata.get("page", "?")
                    
                    pdf.set_font('Arial', '', 8)
                    pdf.cell(0, 5, f'  {j}. {src} (Page {page})', 0, 1)
                
                pdf.ln(2)
            
            pdf.ln(5)
        
        # Footer
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(107, 114, 128)
        pdf.multi_cell(0, 5, 'CircularIQ - Circular Bioeconomy Decision Support Assistant\nDeveloped by International Water Management Institute (IWMI)')
        
        # Convert to bytes
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            pdf_output = pdf_output.encode('latin-1')
        return pdf_output
    
    except Exception as e:
        st.error(f"❌ PDF generation failed: {e}")
        return None

# ------------------- MAIN APP -------------------

def main():
    st.set_page_config(
        page_title="CircularIQ - CBE Decision Support",
        layout="wide",
        page_icon="🔄",
        initial_sidebar_state="expanded"
    )
    
    # 🔐 AUTHENTICATION CHECK
    if not check_google_auth():
        return
    
    load_custom_css()
    init_session_state()
    
    # Initialize float for fixed positioning
    float_init()

    # JS helper: smooth scroll + highlight the target card
    st.markdown("""
    <script>
    window.highlightSource = function(targetId) {
        const anchor = document.getElementById(targetId);
        if (!anchor) return;
        anchor.scrollIntoView({behavior: 'smooth', block: 'center'});
        const card = document.getElementById(targetId + '-card');
        if (card) {
            card.classList.add('reference-highlight');
            setTimeout(() => card.classList.remove('reference-highlight'), 1800);
        }
    };
    </script>
    """, unsafe_allow_html=True)
    
    # Get user email
    user_email = st.session_state.google_user.get("email")
    st.session_state.user_email = user_email
    
    # Load chat history from file on first run
    if "chat_loaded" not in st.session_state:
        chat_data = load_chat_history(user_email)
        if chat_data:
            st.session_state.messages = chat_data.get("messages", [])
            st.session_state.total_queries = chat_data.get("total_queries", 0)
            st.session_state.model = chat_data.get("model", "DeepSeek")
        st.session_state.chat_loaded = True
    
    # Show user info in sidebar
    if st.session_state.get("google_authenticated"):
        user = st.session_state.google_user
        user_name = user.get('name', 'User')
        user_initial = get_user_initial(user_name)
        
        with st.sidebar:
            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-image-wrapper">
                    <img src="{user['picture']}" 
                         alt="{user_name}" 
                         class="profile-image" 
                         loading="lazy" 
                         onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                    <div class="profile-initial" style="display: none;">{user_initial}</div>
                </div>
                <div class="profile-info">
                    <h3>{user_name}</h3>
                    <p>{user['email']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Actual Streamlit logout button
            if st.button("↗️ Sign Out", key="logout_btn", use_container_width=True):
                # Import logout function
                from google_auth import logout
                logout()
    
    # Professional loading screen
    if not st.session_state.rag_loaded:
        spinner_text = "🔄 Switching AI Model... Please wait." if st.session_state.is_switching else "🚀 Initializing CircularIQ Assistant... Please wait."
        with st.spinner(spinner_text):
            rag = get_rag_pipeline(st.session_state.model)
            if not rag.load_index():
                st.markdown('<div class="warning-box">⚠️ <strong>Knowledge Base Not Found</strong><br>Please contact your system administrator to build the document index.</div>', unsafe_allow_html=True)
                return
            st.session_state.rag_loaded = True
            st.session_state.rag = rag
            st.session_state.is_switching = False
            st.rerun()
    
    rag = st.session_state.rag
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <style>
        .sidebar-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
            margin: 0.5rem 0;
            border: none;
        }}
        
        .control-panel-header {{
            font-size: 0.7rem;
            font-weight: 700;
            color: {TEXT_SECONDARY};
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 0.5rem 0 0.35rem 0;
            margin: 0;
        }}
        
        .section-content {{
            margin-bottom: 0.5rem;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Divider after profile
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<h4 class="control-panel-header">AI Model</h4>', unsafe_allow_html=True)
        
        # Model Selection - without section box
        st.markdown('<div class="section-content">', unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Select AI Engine",
            ["DeepSeek", "GPT-4o-mini"],
            index=0 if st.session_state.model == "DeepSeek" else 1,
            help="Choose the AI model for processing your queries",
            label_visibility="collapsed"
        )
        
        if selected_model != st.session_state.model:
            st.toast(f"✅ Switched to {selected_model}", icon="🔄")
            st.session_state.model = selected_model
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.session_state.rag_loaded = False
            st.session_state.is_switching = True
            # Save empty chat
            save_chat_history(user_email, [], 0, selected_model)
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<h4 class="control-panel-header">Chat Management</h4>', unsafe_allow_html=True)
        
        # Chat Management - without section box
        st.markdown('<div class="section-content">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 New", use_container_width=True, help="Start a new conversation"):
                st.session_state.rag.clear_conversation()
                st.session_state.messages = []
                st.session_state.total_queries = 0
                # Save to file
                save_chat_history(user_email, [], 0, st.session_state.model)
                st.toast("✨ New conversation started!", icon="🎉")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear all messages"):
                if len(st.session_state.messages) > 0:
                    st.session_state.rag.clear_conversation() 
                    st.session_state.messages = []
                    st.session_state.total_queries = 0
                    # Save to file
                    save_chat_history(user_email, [], 0, st.session_state.model)
                    st.toast("🧹 Chat cleared!", icon="✅")
                    st.rerun()
                else:
                    st.toast("⚠️ Chat is already empty!", icon="ℹ️")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<h4 class="control-panel-header">Export Chat</h4>', unsafe_allow_html=True)
        
        # Export Option - without section box
        st.markdown('<div class="section-content">', unsafe_allow_html=True)
        
        pdf_content = export_conversation_pdf()
        if pdf_content:
            st.download_button(
                label="📕 Download PDF",
                data=pdf_content,
                file_name=f"CircularIQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download conversation as PDF"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # About Section
        with st.expander("ℹ️ About CircularIQ"):
            st.markdown("""
            **CircularIQ** is the Circular Bioeconomy Decision Support Assistant (CBE-DSA), 
            developed by **IWMI**.
            
            **Purpose:**
            - Evidence-based decision support
            - Circular bioeconomy insights
            - Sustainable waste management
            - Research dissemination
            
            **Target Users:**
            - Policymakers
            - Industry professionals
            - Entrepreneurs & Investors
            - Development partners
            """)
    
    # Header
    st.markdown(f"""
    <div class="header-container">
        <div class="brand-container">
            <span class="brand-icon">🔄</span>
            <h1 class="header-title">CircularIQ</h1>
        </div>
        <p class="header-subtitle">Circular Bioeconomy Decision Support Assistant</p>
        <div style="text-align: center;">
            <span class="iwmi-badge">
                🌍 Powered by IWMI Research
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message for new users
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="info-box">
            <strong>👋 Welcome to CircularIQ!</strong><br>
            I'm your AI-powered assistant for circular bioeconomy and sustainable waste management. 
            Ask me questions about:<br>
            • Circular economy principles & business models<br>
            • Sustainable waste management & resource recovery<br>
            • Climate-smart agricultural systems<br>
            • Policy frameworks & financing strategies<br>
            • Innovation ecosystems & partnerships
        </div>
        """, unsafe_allow_html=True)
    
    # Check if documents are loaded
    if not rag.documents:
        st.markdown('<div class="warning-box">⚠️ <strong>Knowledge Base Empty</strong><br>No documents are currently loaded. Please contact support.</div>', unsafe_allow_html=True)
        return
    
    # Display chat messages from history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            content = message["content"]

            if message["role"] == "assistant":
                # Ensure each assistant message has a unique msg_id
                msg_id = message.get("msg_id", f"msg-{idx}")

                import re
                def repl(m, _msg_id=msg_id):
                    label = m.group(0)                 # e.g. "[Source 3]"
                    num = re.findall(r"\d+", label)[0] # "3"
                    target = f"{_msg_id}-source-{num}"
                    return f'<a href="#{target}" onclick="window.highlightSource(\'{target}\'); return false;">{label}</a>'

                content = re.sub(r"\[Source\s+\d+\]", repl, content)
                st.markdown(content, unsafe_allow_html=True)
            else:
                st.markdown(content)

            # References block
            if "references" in message and message["references"]:
                msg_id = message.get("msg_id", f"msg-{idx}")

                with st.expander(f"📚 View {len(message['references'])} Source Documents"):
                    for i, doc in enumerate(message["references"], 1):
                        # Handle both dict (from JSON) and Document objects
                        if isinstance(doc, dict):
                            meta = doc.get("metadata", {})
                            text = doc.get("page_content", "")
                        else:
                            meta = doc.metadata
                            text = doc.page_content

                        src = meta.get("source", "Unknown")
                        page = meta.get("page", "?")
                        doc_type = meta.get("type", "text")

                        if doc_type == "table":
                            type_badge = "🔢 TABLE"
                            badge_class = "badge-table"
                        elif doc_type == "heading":
                            type_badge = "📌 HEADING"
                            badge_class = "badge-heading"
                        else:
                            type_badge = "📄 TEXT"
                            badge_class = "badge-text"

                        # Unique ID per message + source index
                        anchor_id = f"{msg_id}-source-{i}"

                        st.markdown(f"""
                        <a id="{anchor_id}"></a>
                        <div class="reference-card highlight-target" id="{anchor_id}-card">
                            <div class="reference-header">
                                <span>{i}. {src} (Page {page})</span>
                                <span class="reference-badge {badge_class}">{type_badge}</span>
                            </div>
                            <div class="reference-content">{text[:300]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

    # ===== Custom Chat Input Widget (Fixed at Bottom) =====
    footer_container = st.container()
    with footer_container:
        # Use message count as key to reset widget after each send
        widget_key = f"chat_widget_{len(st.session_state.messages)}"
        user_input = chat_input_widget(key=widget_key)
    
    # Float the container - transparent background, pointer-events: none allows scrolling through it
    footer_container.float("bottom: 0px; background-color: transparent; padding: 10px 0; pointer-events: none;")
    
    # Re-enable pointer events for the widget inside using CSS
    st.markdown("""
    <style>
    [data-testid="stVerticalBlock"] > div:has(iframe[title*="chat_input_widget"]) {
        pointer-events: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Process user input from custom widget (text or audio)
    prompt = None
    
    if user_input:
        if "text" in user_input and user_input["text"]:
            # Text input from typing
            prompt = user_input["text"]
        elif "audioFile" in user_input:
            # Audio input from recording: accept multiple serialized formats
            raw_audio = user_input.get("audioFile")
            audio_bytes = None
            try:
                if isinstance(raw_audio, (bytes, bytearray)):
                    audio_bytes = bytes(raw_audio)
                elif isinstance(raw_audio, dict):
                    # Uint8Array serialized as {"0":v0,"1":v1,...} - values may be int or str
                    audio_bytes = bytes([int(raw_audio[k]) for k in sorted(raw_audio, key=int)])
                elif isinstance(raw_audio, (list, tuple)):
                    audio_bytes = bytes(raw_audio)
                elif isinstance(raw_audio, str):
                    # data URL: data:audio/wav;base64,<payload>
                    if raw_audio.startswith("data:") and "," in raw_audio:
                        b64 = raw_audio.split(",", 1)[1]
                        audio_bytes = base64.b64decode(b64)
                    else:
                        # attempt base64 decode
                        try:
                            audio_bytes = base64.b64decode(raw_audio)
                        except Exception:
                            audio_bytes = None
                else:
                    audio_bytes = bytes(list(raw_audio))
            except Exception as e:
                st.error(f"❌ Failed to parse audio payload: {e}")
                audio_bytes = None
            if audio_bytes:
                with st.spinner("🎙️ Transcribing audio..."):
                    transcribed_text = transcribe_audio(audio_bytes)
                    if transcribed_text:
                        prompt = transcribed_text
                        st.info(f"📝 Transcribed: {prompt}")

    if prompt:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Persist user message immediately to avoid losing it if processing fails
        save_chat_history(user_email, st.session_state.messages, st.session_state.total_queries, st.session_state.model)
        
        # Process with RAG
        with st.spinner("🔍 Analyzing IWMI research documents..."):
            try:
                answer, references = rag.query(prompt)

                msg_id = f"msg-{len(st.session_state.messages)}"
                
                # Add assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "references": references,
                    "msg_id": msg_id
                })
                st.session_state.total_queries += 1
            
            except Exception as e:
                error_msg = f"⚠️ **Processing Error**\n\nI encountered an issue while processing your query: `{str(e)}`\n\nPlease try rephrasing your question or contact support if the issue persists."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "references": [],
                    "msg_id": f"msg-{len(st.session_state.messages)}"
                })
        
        # Save chat history to file after each message
        save_chat_history(user_email, st.session_state.messages, st.session_state.total_queries, st.session_state.model)
        
        # Rerun to display the updated messages
        st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; color: #94A3B8; font-size: 0.85rem;">
        <p>🌱 <strong>CircularIQ</strong> - Empowering Evidence-Based Decisions for a Sustainable Future</p>
        <p style="font-size: 0.75rem; margin-top: 0.5rem;">
            Developed by International Water Management Institute (IWMI) | 
            <a href="https://www.iwmi.cgiar.org" target="_blank" style="color: #0F766E; text-decoration: none;">www.iwmi.cgiar.org</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()