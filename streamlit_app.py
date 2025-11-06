# ===============================================
# FILE 2: streamlit_app.py
# Streamlit UI - Import and use RAGPipeline
# ===============================================

"""
To run this app:
1. Save rag_pipeline.py in the same directory
2. Run: streamlit run streamlit_app.py
"""

import streamlit as st
from datetime import datetime
import json
import os
from collections import deque

# Import RAG pipeline
from rag_pipeline import RAGPipeline

# ------------------- CONFIG -------------------
PDF_FOLDER = "C://iwmi-remote-work//CBE-Chatbot//New folder//cbe//agri and waste water"
INDEX_FILE = "pdf_index_enhanced.pkl"

AZURE_OPENAI_KEY = st.secrets["api_key"]
AZURE_OPENAI_ENDPOINT = "https://iwmi-chat-demo.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"

MAX_CONVERSATION_MEMORY = 10
CONVERSATION_HISTORY_FILE = "conversation_history.json"

# ------------------- CUSTOM CSS -------------------
def load_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 5px 0;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ------------------- SESSION STATE -------------------
def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None

# ------------------- HELPER FUNCTIONS -------------------
@st.cache_resource
def get_rag_pipeline():
    """Get or create RAG pipeline (cached)"""
    pipeline = RAGPipeline(
        pdf_folder=PDF_FOLDER,
        index_file=INDEX_FILE,
        azure_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT
    )
    return pipeline

def save_conversation():
    """Save current conversation to file"""
    try:
        conversation_data = {
            "id": st.session_state.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "total_queries": st.session_state.total_queries
        }
        
        # Load existing conversations
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            with open(CONVERSATION_HISTORY_FILE, "r") as f:
                all_conversations = json.load(f)
        else:
            all_conversations = []
        
        # Update or add current conversation
        all_conversations = [c for c in all_conversations if c["id"] != st.session_state.conversation_id]
        all_conversations.append(conversation_data)
        
        # Save
        with open(CONVERSATION_HISTORY_FILE, "w") as f:
            json.dump(all_conversations, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")
        return False

def export_conversation_markdown():
    """Export conversation as markdown"""
    md_content = f"# Conversation Export\n\n"
    md_content += f"**ID:** {st.session_state.conversation_id}\n\n"
    md_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += "---\n\n"
    
    for msg in st.session_state.messages:
        role = "🧑 **User**" if msg["role"] == "user" else "🤖 **Assistant**"
        md_content += f"{role}:\n{msg['content']}\n\n"
    
    return md_content

# ------------------- MAIN APP -------------------
def main():
    st.set_page_config(
        page_title="PDF Knowledge Chatbot",
        layout="wide",
        page_icon="📚",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    init_session_state()
    
    # Get RAG pipeline
    rag = get_rag_pipeline()
    
    # ------------------- SIDEBAR -------------------
    with st.sidebar:
        st.title("⚙️ Control Panel")
        
        st.divider()
        
        # Index Management
        st.subheader("📁 Index Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Rebuild Index", use_container_width=True):
                if os.path.exists(INDEX_FILE):
                    os.remove(INDEX_FILE)
                
                with st.spinner("Building index..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        num_chunks = rag.build_index(
                            progress_callback=lambda p: progress_bar.progress(p),
                            status_callback=lambda s: status_text.text(s)
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"✅ Indexed {num_chunks} chunks!")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Index building failed: {e}")
        
        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.total_queries = 0
                st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.rerun()
        
        # Load index if not loaded
        if not rag.documents:
            with st.spinner("Loading index..."):
                if not rag.load_index():
                    st.warning("No index found. Click 'Rebuild Index' to create one.")
        
        st.divider()
        
        # Statistics
        if rag.documents:
            st.subheader("📊 Statistics")
            stats = rag.get_stats()
            
            st.metric("Total Chunks", stats["total_chunks"])
            st.metric("Total Queries", st.session_state.total_queries)
            
            if stats.get("content_types"):
                st.write("**Content Types:**")
                for content_type, count in stats["content_types"].items():
                    st.text(f"  {content_type}: {count}")
        
        st.divider()
        
        # Conversation Management
        st.subheader("💾 Conversation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save", use_container_width=True):
                if save_conversation():
                    st.success("Saved!")
        
        with col2:
            if st.button("📥 Export", use_container_width=True):
                md_content = export_conversation_markdown()
                st.download_button(
                    label="Download MD",
                    data=md_content,
                    file_name=f"conversation_{st.session_state.conversation_id}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        
        st.divider()
        
        st.caption("🤖 Powered by all-mpnet-base-v2")
        st.caption(f"📅 Session: {st.session_state.conversation_id}")
    
    # ------------------- MAIN CONTENT -------------------
    st.title("📚 PDF Knowledge Chatbot")
    st.caption("Ask questions about your agricultural and wastewater management documents")
    
    # Check if index is ready
    if not rag.documents:
        st.warning("⚠️ Please build the index first using the sidebar button.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show references if available
            if "references" in message and message["references"]:
                with st.expander(f"📖 View {len(message['references'])} references"):
                    for i, doc in enumerate(message["references"], 1):
                        src = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "?")
                        doc_type = doc.metadata.get("type", "text")
                        
                        type_badge = ""
                        if doc_type == "table":
                            type_badge = "🔢 TABLE"
                        elif doc_type == "heading":
                            type_badge = "📌 HEADING"
                        
                        st.markdown(f"""
                        <div style="border:1px solid #ddd; border-radius:8px; padding:10px; margin-bottom:8px; background-color:#f9f9f9;">
                            <b>{i}. {src} (Page {page})</b> {type_badge}<br>
                            <span style="color:#444; font-size:0.9em;">{doc.page_content[:300]}...</span>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                try:
                    answer, references = rag.query(prompt)
                    
                    st.markdown(answer)
                    
                    # Show references
                    if references:
                        with st.expander(f"📖 View {len(references)} references"):
                            for i, doc in enumerate(references, 1):
                                src = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", "?")
                                doc_type = doc.metadata.get("type", "text")
                                
                                type_badge = ""
                                if doc_type == "table":
                                    type_badge = "🔢 TABLE"
                                elif doc_type == "heading":
                                    type_badge = "📌 HEADING"
                                
                                st.markdown(f"""
                                <div style="border:1px solid #ddd; border-radius:8px; padding:10px; margin-bottom:8px; background-color:#f9f9f9;">
                                    <b>{i}. {src} (Page {page})</b> {type_badge}<br>
                                    <span style="color:#444; font-size:0.9em;">{doc.page_content[:300]}...</span>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Save to history
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
