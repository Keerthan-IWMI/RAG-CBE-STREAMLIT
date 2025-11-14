import streamlit as st
from datetime import datetime
import json
import os
from io import BytesIO
from fpdf import FPDF

# Import RAG pipeline
from pre_update_rag_pipeline import RAGPipeline

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

# Professional Color Palette (Modern Blue-Green Scheme)
PRIMARY_COLOR = "#0F766E"  # Teal
SECONDARY_COLOR = "#06B6D4"  # Cyan
ACCENT_COLOR = "#10B981"  # Emerald
BACKGROUND_LIGHT = "#F0FDFA"  # Light teal
TEXT_PRIMARY = "#0F172A"  # Slate
TEXT_SECONDARY = "#475569"  # Slate gray

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
        padding-top: 2rem;
    }}
    
    .sidebar-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {PRIMARY_COLOR};
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .sidebar-section {{
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
    }}
    
    /* Buttons */
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        color: white;
        border: none;
        padding: 0.65rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
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
        padding: 0.65rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
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

    /* Make header bar blend into the background by matching the color */
    header[data-testid="stHeader"] {{
        background: linear-gradient(180deg, {BACKGROUND_LIGHT} 100%, #ECFDF5 0%, #F0F9FF 10%);
        border-bottom: none;
        
    }}
    
    # /* Hide Deploy button */
    # .stDeployButton {{
    #     display: none !important;
    #     visibility: hidden !important;
    # }}
    
    # [data-testid="stToolbar"] {{
    #     display: none !important;
    #     visibility: hidden !important;
    # }}
    
    # /* Hide the entire top-right menu area */
    # header[data-testid="stHeader"] > div:first-child {{
    #     display: none !important;
    # }}
    
    # /* Make sure sidebar toggle button is always visible */
    # [data-testid="stSidebarNav"] {{
    #     display: block !important;
    # }}
    
    # button[kind="header"] {{
    #     display: block !important;
    #     visibility: visible !important;
    # }}
    
    # /* Sidebar collapse button styling */
    # [data-testid="collapsedControl"] {{
    #     display: flex !important;
    #     visibility: visible !important;
    #     color: {PRIMARY_COLOR} !important;
    #     background: white !important;
    #     border: 2px solid {PRIMARY_COLOR} !important;
    #     border-radius: 8px !important;
    #     padding: 0.5rem !important;
    #     margin: 1rem !important;
    #     box-shadow: 0 2px 8px rgba(15, 118, 110, 0.2) !important;
    # }}
    
    # [data-testid="collapsedControl"]:hover {{
    #     background: {BACKGROUND_LIGHT} !important;
    #     transform: scale(1.05);
    #     box-shadow: 0 4px 12px rgba(15, 118, 110, 0.3) !important;
    # }}
    
    # /* Ensure the header area shows the toggle */
    # header {{
    #     visibility: visible !important;
    # }}
    
    # header[data-testid="stHeader"] {{
    #     background-color: transparent !important;
    # }}


    
    /* Sidebar open/close button */
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
    """Save conversation to JSON file"""
    try:
        # Convert messages to JSON-serializable format
        serializable_messages = []
        for msg in st.session_state.messages:
            msg_copy = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            # Convert Document objects to dictionaries
            if "references" in msg and msg["references"]:
                msg_copy["references"] = [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in msg["references"]
                ]
            else:
                msg_copy["references"] = []
            
            serializable_messages.append(msg_copy)
        
        conversation_data = {
            "id": st.session_state.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": serializable_messages,
            "total_queries": st.session_state.total_queries,
            "model": st.session_state.model
        }
        
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            with open(CONVERSATION_HISTORY_FILE, "r", encoding="utf-8") as f:
                all_conversations = json.load(f)
        else:
            all_conversations = []
        
        all_conversations = [c for c in all_conversations if c["id"] != st.session_state.conversation_id]
        all_conversations.append(conversation_data)
        
        with open(CONVERSATION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(all_conversations, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to save conversation: {e}")
        return False

def clean_text_for_pdf(text):
    """Clean text to remove problematic characters for latin-1 encoding"""
    # Replace smart quotes and other unicode characters
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
    
    # Remove any remaining non-latin1 characters
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
        pdf.cell(0, 6, f'Session ID: {st.session_state.conversation_id}', 0, 1)
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
                
                # Add reference details
                for j, doc in enumerate(msg["references"][:3], 1):  # Limit to first 3 references
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
    
    load_custom_css()
    init_session_state()
    
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
        st.markdown('<div class="sidebar-title">⚙️ Control Panel</div>', unsafe_allow_html=True)
        
        # Model Selection
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**🤖 AI Model**")
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
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.rag_loaded = False
            st.session_state.is_switching = True
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat Management
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**🔧 Chat Management**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 New", use_container_width=True, help="Start a new conversation"):
                st.session_state.rag.clear_conversation()
                st.session_state.messages = []
                st.session_state.total_queries = 0
                st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.toast("✨ New conversation started!", icon="🎉")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear all messages"):
                if len(st.session_state.messages) > 0:
                    st.session_state.rag.clear_conversation() 
                    st.session_state.messages = []
                    st.session_state.total_queries = 0
                    st.toast("🧹 Chat cleared!", icon="✅")
                    st.rerun()
                else:
                    st.toast("⚠️ Chat is already empty!", icon="ℹ️")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export Option
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**💾 Export Chat**")
        
        pdf_content = export_conversation_pdf()
        if pdf_content:
            st.download_button(
                label="📕 Download PDF",
                data=pdf_content,
                file_name=f"CircularIQ_{st.session_state.conversation_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download conversation as PDF"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Session Statistics
        st.markdown('<div class="session-info">', unsafe_allow_html=True)
        st.markdown("**📊 Session Stats**")
        st.markdown(f'<div class="stat-box"><span class="stat-label">Session ID</span><span class="stat-value">{st.session_state.conversation_id[:8]}...</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-box"><span class="stat-label">Queries</span><span class="stat-value">{st.session_state.total_queries}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-box"><span class="stat-label">Messages</span><span class="stat-value">{len(st.session_state.messages)}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-box"><span class="stat-label">Model</span><span class="stat-value">{st.session_state.model}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            
            if "references" in message and message["references"]:
                with st.expander(f"📚 View {len(message['references'])} Source Documents"):
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
                        else:
                            type_badge = "📄 TEXT"
                            badge_class = "badge-text"
                        
                        st.markdown(f"""
                        <div class="reference-card">
                            <div class="reference-header">
                                <span>{i}. {src} (Page {page})</span>
                                <span class="reference-badge {badge_class}">{type_badge}</span>
                            </div>
                            <div class="reference-content">{doc.page_content[:300]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("💬 Ask me about circular bioeconomy, waste management, or sustainable practices..."):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process with RAG
        with st.spinner("🔍 Analyzing IWMI research documents..."):
            try:
                answer, references = rag.query(prompt)
                
                # Add assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "references": references
                })
                st.session_state.total_queries += 1
            
            except Exception as e:
                error_msg = f"⚠️ **Processing Error**\n\nI encountered an issue while processing your query: `{str(e)}`\n\nPlease try rephrasing your question or contact support if the issue persists."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "references": []
                })
        
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