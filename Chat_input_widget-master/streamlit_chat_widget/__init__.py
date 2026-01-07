import streamlit.components.v1 as components
import os

_RELEASE = True

if _RELEASE:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend", "build")
    _component_func = components.declare_component("chat_input_widget", path=build_dir)
else:
    _component_func = components.declare_component("chat_input_widget", url="http://localhost:3000")

def chat_input_widget(
    key=None, 
    pdf_data=None, 
    pdf_filename="conversation.pdf", 
    dark_mode=False, 
    show_suggestions=False,
    enable_react_ui=False,
    user_email="",
    user_name="",
    conversations=None
):
    """
    Custom chat input widget with text, audio, and PDF download support.
    
    Args:
        key: Unique key for the component
        pdf_data: Base64-encoded PDF data for download
        pdf_filename: Filename for the downloaded PDF
        dark_mode: Whether dark mode is enabled
        show_suggestions: Whether to show suggestion chips (for empty chat)
        enable_react_ui: Whether to render React-based sidebar/header overlays
        user_email: User's email for sidebar display
        user_name: User's name for sidebar display  
        conversations: List of conversation dicts with 'id' and 'title' for history
    
    Returns:
        dict with user input (text, audioFile, action, or filter)
    """
    component_value = _component_func(
        key=key,
        pdf_data=pdf_data,
        pdf_filename=pdf_filename,
        dark_mode=dark_mode,
        show_suggestions=show_suggestions,
        enable_react_ui=enable_react_ui,
        user_email=user_email,
        user_name=user_name,
        conversations=conversations or [],
        default=None
    )
    return component_value