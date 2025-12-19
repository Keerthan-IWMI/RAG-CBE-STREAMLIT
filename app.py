import streamlit as st
from datetime import datetime
import json
import os
from fpdf import FPDF
import hashlib
from uuid import uuid4
import requests
import base64
from streamlit_chat_widget import chat_input_widget
from streamlit_float import float_init
import tiktoken
import logging

# Import RAG pipeline
from cbe_agent import RAGPipeline
from google_auth import check_google_auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Maximum agent loop iterations
MAX_AGENT_LOOPS = 5

# Professional Color Palette
PRIMARY_COLOR = "#0F766E"
SECONDARY_COLOR = "#06B6D4"
ACCENT_COLOR = "#10B981"
BACKGROUND_LIGHT = "#F0FDFA"
TEXT_PRIMARY = "#0F172A"
TEXT_SECONDARY = "#475569"

# =============== SESSION STATE ===============

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
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "saved_chat" not in st.session_state:
        st.session_state.saved_chat = None
    if "chat_loaded" not in st.session_state:
        st.session_state.chat_loaded = False

# =============== CUSTOM CSS (same as before, keeping it concise) ===============

def load_custom_css(dark_mode=False):
    # Using same CSS as original - keeping it for brevity
    if dark_mode:
        bg_main = "#0f172a"
        bg_card = "#1e293b"
        text_primary = "#f1f5f9"
        text_secondary = "#94a3b8"
        border_color = "#334155"
        accent = "#22d3ee"
    else:
        bg_main = BACKGROUND_LIGHT
        bg_card = "white"
        text_primary = TEXT_PRIMARY
        text_secondary = TEXT_SECONDARY
        border_color = "#E2E8F0"
        accent = PRIMARY_COLOR
    
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }}
    
    .main {{
        background: {"#0f172a" if dark_mode else f"linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #ECFDF5 50%, #F0F9FF 100%)"};
        padding: 0;
    }}
    
    .header-container {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 8px 32px rgba(15, 118, 110, 0.15);
        margin-bottom: 2rem;
    }}
    
    .header-title {{
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0;
    }}
    
    .stChatMessage {{
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }}
    
    .reference-card {{
        background: {bg_card};
        border: 1.5px solid {border_color};
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
    }}
    
    .reference-card:hover {{
        border-color: {accent};
        transform: translateY(-2px);
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


# In-memory store for guest sessions
@st.cache_resource(show_spinner=False)
def _guest_store():
    return {}

# =============== MAIN APP ===============

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
    
    init_session_state()

    # For guests, restore conversation from in-memory cache
    guest_session_id = st.query_params.get("guest_session")
    if st.session_state.get("guest_authenticated") and guest_session_id:
        store = _guest_store()
        st.session_state.guest_session_id = guest_session_id
        cached = store.get(guest_session_id)
        if cached:
            st.session_state.messages = cached.get("messages", [])
            st.session_state.total_queries = cached.get("total_queries", 0)
            st.session_state.model = cached.get("model", st.session_state.model)
    
    load_custom_css(st.session_state.dark_mode)
    float_init()
    
    # Get user email
    user_email = st.session_state.google_user.get("email")
    st.session_state.user_email = user_email
    
    # Load chat history from file once per session
    if not st.session_state.get("chat_loaded", False):
        if not st.session_state.get("guest_authenticated"):
            chat_data = load_chat_history(user_email)
            if chat_data:
                st.session_state.saved_chat = chat_data
                st.session_state.total_queries = chat_data.get("total_queries", st.session_state.total_queries)
                st.session_state.model = chat_data.get("model", st.session_state.model)
        st.session_state.chat_loaded = True
    
    # Show user info in sidebar (simplified for brevity - same as original)
    with st.sidebar:
        user = st.session_state.google_user
        user_name = user.get('name', 'User')
        user_initial = get_user_initial(user_name)
        
        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-image-wrapper">
                <img src="{user['picture']}" alt="{user_name}" class="profile-image">
            </div>
            <div class="profile-info">
                <h3>{user_name}</h3>
                <p>{user['email']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("↗️ Sign Out", key="logout_btn", use_container_width=True):
            from google_auth import logout
            logout()
        
        if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode", 
                    key="theme_toggle_btn", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # Initialize RAG pipeline
    if not st.session_state.rag_loaded:
        spinner_text = "🔄 Switching AI Model..." if st.session_state.is_switching else "🚀 Initializing CircularIQ Assistant..."
        with st.spinner(spinner_text):
            rag = get_rag_pipeline(st.session_state.model)
            if not rag.load_index():
                st.error("⚠️ Knowledge Base Not Found")
                return
            st.session_state.rag_loaded = True
            st.session_state.rag = rag
            st.session_state.is_switching = False
            st.rerun()
    
    rag = st.session_state.rag
    
    # Get LLM client for agent
    llm_client = get_llm_client(st.session_state.model)
    
    # Sidebar controls (simplified)
    with st.sidebar:
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h4 class="control-panel-header">AI Model</h4>', unsafe_allow_html=True)
        
        selected_model = st.selectbox(
            "Select AI Engine",
            ["DeepSeek", "GPT-4o-mini"],
            index=0 if st.session_state.model == "DeepSeek" else 1,
            label_visibility="collapsed"
        )
        
        if selected_model != st.session_state.model:
            st.toast(f"✅ Switched to {selected_model}", icon="🔄")
            st.session_state.model = selected_model
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.session_state.rag_loaded = False
            st.session_state.is_switching = True
            save_chat_history(user_email, [], 0, selected_model)
            st.rerun()
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h4 class="control-panel-header">Chat Management</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 New", use_container_width=True):
                if st.session_state.messages:
                    archive_messages(user_email, st.session_state.messages, 
                                   st.session_state.total_queries, st.session_state.model)
                st.session_state.messages = []
                st.session_state.total_queries = 0
                save_chat_history(user_email, [], 0, st.session_state.model)
                st.toast("✨ New conversation started!", icon="🎉")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                if st.session_state.messages:
                    archive_messages(user_email, st.session_state.messages,
                                   st.session_state.total_queries, st.session_state.model)
                st.session_state.messages = []
                st.session_state.total_queries = 0
                save_chat_history(user_email, [], 0, st.session_state.model)
                st.toast("🗑️ Chat cleared!", icon="✅")
                st.rerun()
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h4 class="control-panel-header">Export Chat</h4>', unsafe_allow_html=True)
        
        pdf_content = export_conversation_pdf()
        if pdf_content:
            st.download_button(
                label="📕 Download PDF",
                data=pdf_content,
                file_name=f"CircularIQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    # Header
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">🔄 CircularIQ</h1>
        <p style="text-align: center; color: white;">Circular Bioeconomy Decision Support Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    if len(st.session_state.messages) == 0:
        st.info("""
        👋 **Welcome to CircularIQ!**
        
        I'm your AI-powered assistant for circular bioeconomy and sustainable waste management.
        Ask me questions about circular economy, waste management, water reuse, and sustainable agriculture.
        """)
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            content = message["content"]
            
            if message["role"] == "assistant":
                msg_id = message.get("msg_id", f"msg-{idx}")
                import re
                def repl(m, _msg_id=msg_id):
                    label = m.group(0)
                    num = re.findall(r"\d+", label)[0]
                    target = f"{_msg_id}-source-{num}"
                    return f'<a href="#{target}" onclick="window.highlightSource(\'{target}\'); return false;">{label}</a>'
                
                content = re.sub(r"\[Source\s+\d+\]", repl, content)
                st.markdown(content, unsafe_allow_html=True)
            else:
                st.markdown(content)
            
            # References
            if "references" in message and message["references"]:
                msg_id = message.get("msg_id", f"msg-{idx}")
                
                with st.expander(f"📚 View {len(message['references'])} Source Documents"):
                    for i, doc in enumerate(message["references"], 1):
                        if isinstance(doc, dict):
                            src = doc.get("source", "Unknown")
                            page = doc.get("page", "?")
                            doc_type = doc.get("type", "text")
                            text = doc.get("content", "")
                        else:
                            src = "Unknown"
                            page = "?"
                            doc_type = "text"
                            text = str(doc)[:300]
                        
                        anchor_id = f"{msg_id}-source-{i}"
                        
                        st.markdown(f"""
                        <a id="{anchor_id}"></a>
                        <div class="reference-card" id="{anchor_id}-card">
                            <div class="reference-header">
                                <span>{i}. {src} (Page {page})</span>
                                <span class="reference-badge">{doc_type.upper()}</span>
                            </div>
                            <div class="reference-content">{text[:300]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Generate PDF data for widget
    pdf_data_b64 = None
    if st.session_state.messages:
        pdf_content = export_conversation_pdf()
        if pdf_content:
            pdf_data_b64 = base64.b64encode(pdf_content).decode()
    
    # Custom Chat Input Widget
    footer_container = st.container()
    with footer_container:
        widget_key = f"chat_widget_{len(st.session_state.messages)}"
        user_input = None
        try:
            user_input = chat_input_widget(
                key=widget_key,
                pdf_data=pdf_data_b64,
                pdf_filename=f"CircularIQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                dark_mode=st.session_state.dark_mode,
                show_suggestions=(len(st.session_state.messages) == 0)
            )
        except Exception as e:
            st.error("⚠️ Custom chat input widget failed. Using fallback.")
            text_fallback = st.chat_input("Start typing here...")
            if text_fallback:
                user_input = {"text": text_fallback}
    
    footer_container.float("bottom: 0px; background-color: transparent; padding: 10px 0;")
    
    # Process user input
    prompt = None
    
    if user_input:
        if "text" in user_input and user_input["text"]:
            prompt = user_input["text"]
        elif "audioFile" in user_input:
            raw_audio = user_input.get("audioFile")
            audio_bytes = None
            try:
                if isinstance(raw_audio, (bytes, bytearray)):
                    audio_bytes = bytes(raw_audio)
                elif isinstance(raw_audio, str) and "," in raw_audio:
                    b64 = raw_audio.split(",", 1)[1]
                    audio_bytes = base64.b64decode(b64)
            except Exception as e:
                st.error(f"❌ Failed to parse audio: {e}")
            
            if audio_bytes:
                with st.spinner("🎙️ Transcribing audio..."):
                    transcribed_text = transcribe_audio(audio_bytes)
                
                if transcribed_text:
                    prompt = transcribed_text
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat_history(user_email, st.session_state.messages, st.session_state.total_queries, st.session_state.model)
        
        # Build conversation history for agent (only user/assistant messages, no tool messages)
        conv_history = []
        for msg in st.session_state.messages[:-1]:  # Exclude current user message
            if msg["role"] in ["user", "assistant"]:
                conv_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Run agent loop
        with st.spinner("🤖 CircularIQ is thinking..."):
            try:
                # Determine LLM parameters
                if st.session_state.model == "GPT-4o-mini":
                    llm_type = "azure"
                    model_deployment = AZURE_OPENAI_DEPLOYMENT
                else:
                    llm_type = "deepseek"
                    model_deployment = None
                
                answer, retrieved_docs, loop_count = run_agent_loop(
                    user_question=prompt,
                    conversation_history=conv_history,
                    rag_pipeline=rag,
                    llm_client=llm_client,
                    llm_type=llm_type,
                    model_deployment=model_deployment
                )
                
                msg_id = f"msg-{len(st.session_state.messages)}"
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "references": retrieved_docs,
                    "msg_id": msg_id
                })
                st.session_state.total_queries += 1
                
                logger.info(f"Agent completed query in {loop_count} iterations")
            
            except Exception as e:
                error_msg = f"⚠️ **Processing Error**\n\nI encountered an issue: `{str(e)}`\n\nPlease try again."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "references": [],
                    "msg_id": f"msg-{len(st.session_state.messages)}"
                })
        
        # Save chat history
        if st.session_state.get("guest_authenticated") and st.session_state.get("guest_session_id"):
            store = _guest_store()
            store[st.session_state.guest_session_id] = {
                "messages": st.session_state.messages,
                "total_queries": st.session_state.total_queries,
                "model": st.session_state.model,
            }
        else:
            save_chat_history(user_email, st.session_state.messages, st.session_state.total_queries, st.session_state.model)
        
        st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; color: #94A3B8; font-size: 0.85rem;">
        <p>🌱 <strong>CircularIQ</strong> - Empowering Evidence-Based Decisions</p>
        <p>Developed by IWMI | <a href="https://www.iwmi.cgiar.org">www.iwmi.cgiar.org</a></p>
    </div>
    """, unsafe_allow_html=True)


def get_tool_definitions():
    """Define available tools for the agent"""
    return [
        {
            "type": "function",
            "function": {
                "name": "retrieve_documents",
                "description": "Retrieve relevant documents from the IWMI knowledge base to answer questions about circular bioeconomy, waste management, water reuse, and sustainable agriculture. Use this tool when you need factual information from research documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's question or query to search for relevant documents"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of documents to retrieve (default: 8)",
                            "default": 8
                        }
                    },
                    "required": ["question"]
                }
            }
        }
    ]

def execute_tool(tool_name: str, tool_args: dict, rag_pipeline) -> dict:
    """Execute a tool call and return the result"""
    if tool_name == "retrieve_documents":
        question = tool_args.get("question", "")
        top_k = tool_args.get("top_k", 8)
        
        result = rag_pipeline.retrieve_documents(question, top_k)
        return result
    else:
        return {
            "success": False,
            "message": f"Unknown tool: {tool_name}",
            "documents": [],
            "count": 0
        }

# =============== AGENT LOOP ===============

def run_agent_loop(user_question: str, conversation_history: list, rag_pipeline, llm_client, llm_type: str, model_deployment: str = None) -> tuple:
    """
    Run agent loop with tool calling.
    Returns: (final_answer, retrieved_documents, loop_count)
    """
    tools = get_tool_definitions()
    
    system_prompt = """You are the Circular Bioeconomy Decision Support Assistant (CBE-DSA) - an AI-powered chatbot developed to disseminate applied research and evidence-based insights from the International Water Management Institute (IWMI) and related partners.

Your primary goal is to help users, including policymakers, industry professionals, entrepreneurs, investors, and development partners, make informed, evidence-based decisions in the circular bioeconomy and sustainable waste management.

Role and Behaviour:
- Serve as a research-driven knowledge advisor, interpreting academic and technical content into concise, practical, and actionable insights.
- Remain accurate, context-aware, and user-oriented, tailoring responses to the user's role (e.g., policymaker vs. entrepreneur).
- Use the retrieve_documents tool to search IWMI documents when you need factual information to answer questions.
- After retrieving documents, synthesize the information into a clear, structured response.
- For follow-up questions, use conversation history to provide coherent, contextual responses.
- Sense the tone of the question to understand if the user needs generic or specific information.

Tone and Communication Style:
- Professional, clear, neutral, and factual.
- Use plain language; cite sources when needed.
- Emphasize practical impact and innovation.

Response Format:
- **Overview:** Provide a summary of the findings (20-30 words for specific questions, 50-100 words for response to generic questions).
- **Key Points:**
    • Use bullet points starting with "•".
    • After EACH bullet point, insert a blank line (exactly one empty line).
    • Each bullet point must be one sentence only.
- **Implications:** Present the practical or policy relevance as bullet points (20-30 words for specific questions, 50-100 words for response to generic questions).
- **If comparative or quantitative data are available**, display them using a **Markdown table** (| Column | Column |).
- **Always cite sources** in [Source X] format after each claim.
- **Avoid long paragraphs**; favour bullet points and tabular summaries for clarity.

CONVERSATION FLOW & ENGAGEMENT (MANDATORY)
- **Every response must end with a proactive, context-aware follow-up question or suggestion.**
- Prefer follow-up questions that can be answered using details from previous exchanges or current information.
- This keeps the conversation alive and guides the user toward deeper insights.

Restrictions and Limitations:
- Do not fabricate references, data, or methodologies.
- Always clarify if a recommendation is derived from evidence or an inferred interpretation.
- Avoid expressing political bias, or speculative prejudices.
- Refrain from giving prescriptive financial or legal advice.

TOOL USAGE:
- When you need information to answer a question, use the retrieve_documents tool.
- After receiving tool results, synthesize the information and provide a comprehensive answer.
- If no relevant documents are found, acknowledge this and provide general guidance based on your training."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user question
    messages.append({"role": "user", "content": user_question})
    
    all_retrieved_docs = []
    loop_count = 0
    
    for iteration in range(MAX_AGENT_LOOPS):
        loop_count += 1
        logger.info(f"Agent loop iteration {loop_count}/{MAX_AGENT_LOOPS}")
        
        try:
            # Call LLM with tools
            if llm_type == "azure":
                response = llm_client.chat.completions.create(
                    model=model_deployment,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=3500,
                    temperature=0.1,
                )
                
                response_message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                
            else:  # deepseek
                headers = {
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": DEEPSEEK_MODEL_NAME,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "max_tokens": 3500,
                    "temperature": 0.1
                }
                
                r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
                
                if r.status_code != 200:
                    logger.error(f"DeepSeek API error: {r.status_code} - {r.text}")
                    return f"Sorry, model error: {r.status_code}", [], loop_count
                
                data = r.json()
                response_message = data["choices"][0]["message"]
                finish_reason = data["choices"][0]["finish_reason"]
            
            # Check if tool calls are needed
            tool_calls = getattr(response_message, 'tool_calls', None) or response_message.get('tool_calls')
            
            if not tool_calls or finish_reason == "stop":
                # No more tool calls, return final answer
                final_content = response_message.content if hasattr(response_message, 'content') else response_message.get('content', '')
                logger.info(f"Agent completed in {loop_count} iterations")
                return final_content, all_retrieved_docs, loop_count
            
            # Add assistant message with tool calls to history
            messages.append({
                "role": "assistant",
                "content": response_message.content if hasattr(response_message, 'content') else response_message.get('content'),
                "tool_calls": tool_calls if isinstance(tool_calls, list) else [tc.__dict__ if hasattr(tc, '__dict__') else tc for tc in tool_calls]
            })
            
            # Execute each tool call
            for tool_call in tool_calls:
                if hasattr(tool_call, 'function'):
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id
                else:
                    function_name = tool_call['function']['name']
                    function_args = json.loads(tool_call['function']['arguments'])
                    tool_call_id = tool_call['id']
                
                logger.info(f"Executing tool: {function_name} with args: {function_args}")
                
                # Execute tool
                tool_result = execute_tool(function_name, function_args, rag_pipeline)
                
                # Collect retrieved documents
                if tool_result.get("success") and tool_result.get("documents"):
                    all_retrieved_docs.extend(tool_result["documents"])
                
                # Format tool result for LLM
                tool_response_content = json.dumps(tool_result)
                
                # Add tool response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_response_content
                })
            
        except Exception as e:
            logger.error(f"Agent loop error: {e}")
            return f"Sorry, I encountered an error: {str(e)}", all_retrieved_docs, loop_count
    
    # Max iterations reached
    logger.warning(f"Agent reached max iterations ({MAX_AGENT_LOOPS})")
    return "I apologize, but I need more time to process your request. Please try rephrasing your question.", all_retrieved_docs, loop_count

# =============== CHAT HISTORY MANAGEMENT ===============

def get_chat_history_file(email: str) -> str:
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)
    safe_email = hashlib.md5(email.encode()).hexdigest()
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_email}_chat.json")

def save_chat_history(email: str, messages: list, total_queries: int, model: str):
    if st.session_state.get("guest_authenticated"):
        return True
    try:
        file_path = get_chat_history_file(email)
        
        serializable_messages = []
        for msg in messages:
            msg_copy = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            refs = []
            if "references" in msg and msg["references"]:
                for doc in msg["references"]:
                    try:
                        if isinstance(doc, dict):
                            refs.append(doc)
                        else:
                            refs.append({"content": str(doc), "metadata": {}})
                    except Exception:
                        refs.append({"content": "", "metadata": {}})
            msg_copy["references"] = refs
            
            serializable_messages.append(msg_copy)
        
        existing_title = None
        try:
            existing_title = st.session_state.get("saved_chat", {}).get("title") if isinstance(st.session_state.get("saved_chat"), dict) else None
        except Exception:
            existing_title = None

        if not existing_title and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    existing_title = existing.get("title")
            except Exception:
                existing_title = None

        chat_data = {
            "user_email": email,
            "timestamp": datetime.now().isoformat(),
            "messages": serializable_messages,
            "total_queries": total_queries,
            "model": model,
        }
        if existing_title and messages:
            chat_data["title"] = existing_title
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to save chat history: {e}")
        return False

def load_chat_history(email: str) -> dict:
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
    try:
        file_path = get_chat_history_file(email)
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting chat history: {e}")
        return False

def _archive_filename_for(email: str, timestamp: str, title: str | None = None) -> str:
    safe_email = hashlib.md5(email.encode("utf-8")).hexdigest()
    uid = uuid4().hex[:8]
    if title:
        slug = "".join(c if c.isalnum() else "_" for c in title)[:40]
        return os.path.join(CHAT_HISTORY_DIR, f"{safe_email}_archive_{timestamp}_{slug}_{uid}.json")
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_email}_archive_{timestamp}_{uid}.json")

def archive_current_history(email: str) -> str | None:
    try:
        current = get_chat_history_file(email)
        if not os.path.exists(current):
            return None
        try:
            with open(current, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive_path = _archive_filename_for(email, ts)
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return archive_path
    except Exception as e:
        print(f"Error archiving chat history: {e}")
        return None

def archive_messages(email: str, messages: list, total_queries: int = 0, model: str = None, title: str | None = None) -> str | None:
    try:
        if not messages:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive_path = _archive_filename_for(email, ts, title)
        chat_data = {
            "user_email": email,
            "timestamp": datetime.now().isoformat(),
            "title": title or "",
            "messages": messages,
            "total_queries": total_queries,
            "model": model or st.session_state.get("model")
        }
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        return archive_path
    except Exception as e:
        print(f"Error archiving messages: {e}")
        return None

def rename_saved_chat(email: str, new_title: str) -> bool:
    try:
        path = get_chat_history_file(email)
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["title"] = new_title
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error renaming saved chat: {e}")
        return False

def list_archived_histories(email: str) -> list:
    try:
        safe = hashlib.md5(email.encode("utf-8")).hexdigest()
        files = []
        if os.path.exists(CHAT_HISTORY_DIR):
            for fn in os.listdir(CHAT_HISTORY_DIR):
                if fn.startswith(f"{safe}_archive_") and fn.endswith(".json"):
                    files.append(os.path.join(CHAT_HISTORY_DIR, fn))
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files
    except Exception as e:
        print(f"Error listing archives: {e}")
        return []

def load_archived_history(path: str) -> dict | None:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading archive {path}: {e}")
        return None

def delete_archived_history(path: str) -> bool:
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
    except Exception as e:
        print(f"Error deleting archive {path}: {e}")
        return False

def rename_archived_history(path: str, new_title: str) -> str | None:
    try:
        if not os.path.exists(path):
            return None
        basename = os.path.basename(path)
        parts = basename.split("_archive_")
        if len(parts) < 2:
            return None
        prefix = parts[0]
        ts = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y%m%d_%H%M%S_%f")
        slug = "".join(c if c.isalnum() else "_" for c in new_title)[:60]
        new_path = os.path.join(CHAT_HISTORY_DIR, f"{prefix}_archive_{ts}_{slug}.json")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

        if data is not None:
            try:
                data["title"] = new_title
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        if os.path.exists(new_path):
            try:
                alt_new_path = os.path.join(CHAT_HISTORY_DIR, f"{prefix}_archive_{ts}_{slug}_{uuid4().hex[:6]}.json")
                os.replace(path, alt_new_path)
                return alt_new_path
            except Exception:
                return path
        else:
            try:
                os.replace(path, new_path)
                return new_path
            except Exception:
                return path
    except Exception as e:
        print(f"Error renaming archive {path}: {e}")
        return None

# =============== HELPER FUNCTIONS ===============

def get_user_initial(name: str) -> str:
    if name:
        return name[0].upper()
    return "U"

def transcribe_audio(audio_bytes):
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
    replacements = {
        '\u201c': '"',
        '\u201d': '"',
        '\u2018': "'",
        '\u2019': "'",
        '\u2013': '-',
        '\u2014': '--',
        '\u2026': '...',
        '\u2022': '*',
        '\u00a0': ' ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.encode('latin-1', errors='ignore').decode('latin-1')

def export_conversation_pdf():
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 20)
        pdf.set_text_color(15, 118, 110)
        pdf.cell(0, 10, 'CircularIQ Conversation Export', 0, 1, 'C')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(0, 6, f'User: {st.session_state.get("user_email", "Unknown")}', 0, 1)
        pdf.cell(0, 6, f'Date: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 0, 1)
        pdf.cell(0, 6, f'Model: {st.session_state.model}', 0, 1)
        pdf.cell(0, 6, f'Total Queries: {st.session_state.total_queries}', 0, 1)
        pdf.ln(10)
        
        for i, msg in enumerate(st.session_state.messages, 1):
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(15, 118, 110)
            role_text = f"User (Message {i})" if msg["role"] == "user" else f"CircularIQ Assistant (Message {i})"
            pdf.cell(0, 8, role_text, 0, 1)
            
            pdf.set_font('Arial', '', 10)
            pdf.set_text_color(0, 0, 0)
            cleaned_content = clean_text_for_pdf(msg['content'])
            pdf.multi_cell(0, 6, cleaned_content)
            pdf.ln(3)
            
            if "references" in msg and msg["references"]:
                pdf.set_font('Arial', 'I', 9)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 6, f'Sources: {len(msg["references"])} documents referenced', 0, 1)
                
                for j, doc in enumerate(msg["references"][:3], 1):
                    if isinstance(doc, dict):
                        src = clean_text_for_pdf(doc.get("metadata", {}).get("source", "Unknown"))
                        page = doc.get("metadata", {}).get("page", "?")
                    else:
                        src = clean_text_for_pdf(doc.get("source", "Unknown"))
                        page = doc.get("page", "?")
                    
                    pdf.set_font('Arial', '', 8)
                    pdf.cell(0, 5, f'  {j}. {src} (Page {page})', 0, 1)
                
                pdf.ln(2)
            
            pdf.ln(5)
        
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(107, 114, 128)
        pdf.multi_cell(0, 5, 'CircularIQ - Circular Bioeconomy Decision Support Assistant\nDeveloped by International Water Management Institute (IWMI)')
        
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            pdf_output = pdf_output.encode('latin-1')
        return pdf_output
    
    except Exception as e:
        st.error(f"❌ PDF generation failed: {e}")
        return None


# =============== RAG PIPELINE ===============

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
    else:
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

@st.cache_resource(show_spinner=False)
def get_llm_client(selected_model: str):
    """Get LLM client for agent loop"""
    if selected_model == "GPT-4o-mini":
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version="2024-02-15-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
    else:
        return None
if __name__ == "__main__":
    main() 
