import React, { useState, useEffect } from "react";
import { Streamlit, withStreamlitConnection, ComponentProps } from "streamlit-component-lib";
import './ChatInputWidget.css';
import ActionButtons from "./components/ActionButtons";
import FilterSidebar from "./components/FilterSidebar";
import FileUploadModal from "./components/FileUploadModal";
import InputField from "./components/InputField";
import MicButton from "./components/MicButton";
import SendButton from "./components/SendButton";
import RecordingIndicator from "./components/RecordingIndicator";


interface ChatInputWidgetProps extends ComponentProps {
  args: {
    pdf_data?: string;
    pdf_filename?: string;
  };
}

const ChatInputWidget: React.FC<ChatInputWidgetProps> = ({ args }) => {
  const [inputText, setInputText] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [showFilter, setShowFilter] = useState(false);
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [filters, setFilters] = useState<{ year?: string; author?: string; keywords?: string }>({ year: "", author: "", keywords: "" });
  // no explicit anchor required for inline popover

  // Pdf data from args (used for download)
  const pdfData = args.pdf_data ?? null;
  const pdfFilename = args.pdf_filename ?? "conversation.pdf";

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

  return (
    <div className="chat-bar-container">
      <ActionButtons
        onDownload={handleDownload}
        onAttach={handleAttach}
        onToggleFilter={onToggleFilter}
        showFilter={showFilter}
        pdfDataAvailable={!!pdfData}
        filterPopover={
          <FilterSidebar
            visible={showFilter}
            filters={filters}
            onChange={(k, v) => setFilters((prev) => ({ ...prev, [k]: v }))}
            onApply={handleApplyFilter}
            onCancel={handleCancelFilter}
          />
        }
      />

      <InputField value={inputText} onChange={setInputText} onKeyPress={handleKeyPress} placeholder="Start typing to talk with RAG Agent" />

      <div className="right-actions">
        {isRecording && <RecordingIndicator />}
        <MicButton onSendAudio={handleSendAudio} onRecordingChange={onRecordingStateChange} />
        <SendButton active={!!inputText.trim()} onClick={handleSendText} />
      </div>

      <FileUploadModal visible={showFileUpload} onClose={handleFileUploadClose} />
    </div>
  );
};

export default withStreamlitConnection(ChatInputWidget);