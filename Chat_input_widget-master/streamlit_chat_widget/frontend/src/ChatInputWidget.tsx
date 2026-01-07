import React, { useState, useEffect } from "react";
import { Streamlit, withStreamlitConnection, ComponentProps } from "streamlit-component-lib";
import './ChatInputWidget.css';
import ActionButtons from "./components/ActionButtons";
import FilterSidebar from "./components/FilterSidebar";
import FileUploadModal from "./components/FileUploadModal";
import SupportModal from "./components/SupportModal";
import InputField from "./components/InputField";
import MicButton from "./components/MicButton";
import SendButton from "./components/SendButton";
import RecordingIndicator from "./components/RecordingIndicator";
import ReactSidebar from "./components/ReactSidebar";
import ReactHeader from "./components/ReactHeader";


interface ChatInputWidgetProps extends ComponentProps {
  args: {
    pdf_data?: string;
    pdf_filename?: string;
    dark_mode?: boolean;
    show_suggestions?: boolean;
    // New: optional UI overlay props
    enable_react_ui?: boolean;
    user_email?: string;
    user_name?: string;
    conversations?: Array<{ id: string; title: string }>;
  };
}

// Suggestion prompts for empty chat state
const SUGGESTIONS = [
  "Circular economy in agriculture",
  "Wastewater reuse practices",
  "Biochar applications",
  "Compost for soil health",
];

const ChatInputWidget: React.FC<ChatInputWidgetProps> = ({ args }) => {
  const [inputText, setInputText] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [showFilter, setShowFilter] = useState(false);
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [showSupport, setShowSupport] = useState(false);
  const [filters, setFilters] = useState<{ yearStart?: string; yearEnd?: string; author?: string; keywords?: string }>({ yearStart: "", yearEnd: "", author: "", keywords: "" });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  // no explicit anchor required for inline popover

  // Pdf data from args (used for download)
  const pdfData = args.pdf_data ?? null;
  const pdfFilename = args.pdf_filename ?? "conversation.pdf";
  const darkMode = args.dark_mode ?? false;
  const showSuggestions = args.show_suggestions ?? false;
  
  // New UI overlay props
  const enableReactUI = args.enable_react_ui ?? false;
  const userEmail = args.user_email ?? "";
  const userName = args.user_name ?? "";
  const conversations = args.conversations ?? [];

  // Inject CSS to hide native Streamlit sidebar/header when React UI is enabled
  // Use a ref to track if we've injected to prevent flicker on re-renders
  const cssInjectedRef = React.useRef(false);
  
  useEffect(() => {
    if (!enableReactUI) return;
    
    let doc: Document;
    try {
      doc = window.parent?.document ?? document;
    } catch {
      doc = document;
    }
    
    const styleId = "react-ui-overlay-styles";
    
    // Only inject once to prevent flicker
    if (cssInjectedRef.current) return;
    cssInjectedRef.current = true;
    
    // Check if already exists
    if (doc.getElementById(styleId)) return;
    
    const style = doc.createElement('style');
    style.id = styleId;
    style.innerHTML = `
      /* Hide native Streamlit sidebar completely */
      section[data-testid="stSidebar"] {
        display: none !important;
      }
      
      /* Hide the sidebar collapse/expand button (>>) */
      [data-testid="collapsedControl"],
      button[data-testid="collapsedControl"],
      div[data-testid="collapsedControl"] {
        display: none !important;
      }
      
      /* Hide native Streamlit header */
      header[data-testid="stHeader"] {
        display: none !important;
      }
      
      /* Hide deploy button and toolbar */
      .stDeployButton,
      [data-testid="stToolbar"] {
        display: none !important;
      }
      
      /* Adjust main content area for React sidebar (280px) and header (64px) */
      .main .block-container {
        padding-left: 300px !important;
        padding-top: 80px !important;
        max-width: 100% !important;
      }
      
      /* Hide the Python-rendered header-container */
      .header-container {
        display: none !important;
      }
      
      /* Hide any remaining streamlit elements */
      .stAppViewBlockContainer > div:first-child > div:first-child {
        display: none !important;
      }
      
      /* Ensure body doesn't have conflicting styles */
      body {
        overflow-x: hidden;
      }
    `;
    doc.head.appendChild(style);
    
    // Don't cleanup on unmount to prevent flicker - styles are needed throughout session
  }, [enableReactUI]);

  useEffect(() => {
    Streamlit.setFrameHeight();
  }, []);

  useEffect(() => {
    // When filter popover is closed, restore the default frame height.
    // We avoid resetting the frame height while the popover is open to prevent clipping.
    if (!showFilter) Streamlit.setFrameHeight();
  }, [showFilter]);

  const handleSendText = () => {
    if (!inputText.trim()) return;
    Streamlit.setComponentValue({ text: inputText.trim() });
    setInputText("");
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendText();
    }
  };

  const handleSendAudio = (base64: string) => {
    Streamlit.setComponentValue({ audioFile: base64 });
  };

  const handleDownload = () => {
    if (!pdfData) return;
    try {
      const bytes = atob(pdfData);
      const arr = new Uint8Array(new ArrayBuffer(bytes.length));
      for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
      const blob = new Blob([arr], { type: "application/pdf" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = pdfFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Download failed", e);
    }
  };

  const handleAttach = () => {
    setShowFileUpload(true);
  };

  const handleFileUploadClose = () => {
    setShowFileUpload(false);
  };

  const handleApplyFilter = () => {
    Streamlit.setComponentValue({ filter: filters });
    setShowFilter(false);
  };

  const handleCancelFilter = () => {
    Streamlit.setComponentValue({ filter: null });
    setShowFilter(false);
  };

  const onToggleFilter = () => {
    setShowFilter((s) => !s);
  };

  const onRecordingStateChange = (v: boolean) => setIsRecording(v);

  // Dynamic dark mode styles for chat bar
  const chatBarStyle: React.CSSProperties = darkMode ? {
    background: '#1e293b',
    borderColor: '#334155',
    boxShadow: '0 -4px 20px rgba(0,0,0,0.3)',
  } : {};

  const handleSuggestionClick = (suggestion: string) => {
    Streamlit.setComponentValue({ text: suggestion });
  };

  return (
    <div className="chat-widget-wrapper">
      {/* Render React UI overlays when enabled */}
      {enableReactUI && (
        <>
          <ReactSidebar 
            darkMode={darkMode} 
            userEmail={userEmail}
            userName={userName}
            conversations={conversations}
            onCollapseChange={setSidebarCollapsed}
          />
          <ReactHeader darkMode={darkMode} sidebarCollapsed={sidebarCollapsed} />
        </>
      )}
      
      {showSuggestions && (
        <div className={`suggestion-chips ${darkMode ? 'dark' : ''}`}>
          {SUGGESTIONS.map((s, i) => (
            <button
              key={i}
              className="suggestion-chip"
              onClick={() => handleSuggestionClick(s)}
            >
              {s}
            </button>
          ))}
        </div>
      )}
      <div className={`chat-bar-container ${darkMode ? 'dark-mode' : ''}`} style={chatBarStyle}>
      <ActionButtons
        onDownload={handleDownload}
        onAttach={handleAttach}
        onToggleFilter={onToggleFilter}
        onSupport={() => setShowSupport(true)}
        showFilter={showFilter}
        pdfDataAvailable={!!pdfData}
        darkMode={darkMode}
        filterPopover={
          <FilterSidebar
            visible={showFilter}
            filters={filters}
            onChange={(k, v) => setFilters((prev) => ({ ...prev, [k]: v }))}
            onApply={handleApplyFilter}
            onCancel={handleCancelFilter}
            darkMode={darkMode}
          />
        }
      />

      <InputField value={inputText} onChange={setInputText} onKeyPress={handleKeyPress} placeholder="Start typing to talk with RAG Agent" darkMode={darkMode} />

      <div className="right-actions">
        {isRecording && <RecordingIndicator />}
        <MicButton onSendAudio={handleSendAudio} onRecordingChange={onRecordingStateChange} darkMode={darkMode} />
        <SendButton active={!!inputText.trim()} onClick={handleSendText} darkMode={darkMode} />
      </div>

      <FileUploadModal visible={showFileUpload} onClose={handleFileUploadClose} darkMode={darkMode} />
      <SupportModal visible={showSupport} onClose={() => setShowSupport(false)} darkMode={darkMode} />
    </div>
    </div>
  );
};

export default withStreamlitConnection(ChatInputWidget);