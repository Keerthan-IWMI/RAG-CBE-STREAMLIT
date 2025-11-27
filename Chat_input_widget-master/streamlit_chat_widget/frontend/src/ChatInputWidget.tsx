import React, { useState, useEffect, useRef } from "react";
import { Streamlit, withStreamlitConnection, ComponentProps } from "streamlit-component-lib";
import SendIcon from "@mui/icons-material/ArrowUpward";
import MicIcon from "@mui/icons-material/MicNoneOutlined";
import StopIcon from "@mui/icons-material/Stop";
import DownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";
import AttachFileOutlinedIcon from "@mui/icons-material/AttachFileOutlined";
import LocationOnOutlinedIcon from "@mui/icons-material/LocationOnOutlined";
import './ChatInputWidget.css';


interface ChatInputWidgetProps extends ComponentProps {
  args: {
    pdf_data?: string;
    pdf_filename?: string;
  };
}

const ChatInputWidget: React.FC<ChatInputWidgetProps> = ({ args }) => {
  const [inputText, setInputText] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Pdf data from args (used for download)
  const pdfData = args.pdf_data ?? null;
  const pdfFilename = args.pdf_filename ?? "conversation.pdf";

  useEffect(() => {
    Streamlit.setFrameHeight();
  }, []);

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

  const handleMicClick = async () => {
    if (isRecording && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current = null;
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      recorder.ondataavailable = (ev) => {
        if (ev.data.size > 0) audioChunksRef.current.push(ev.data);
      };

      recorder.onstop = () => {
        if (audioChunksRef.current.length === 0) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = reader.result as string; // data:audio/wav;base64,...
          // Single send: set component value once on stop (user-initiated)
          Streamlit.setComponentValue({ audioFile: base64String });
          // local cleanup
          audioChunksRef.current = [];
        };
        reader.readAsDataURL(audioBlob);
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      console.error("Mic access denied:", err);
    }
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
    Streamlit.setComponentValue({ attach: true });
  };

  const handleLocation = () => {
    Streamlit.setComponentValue({ location: true });
  };

  return (
    <div className="chat-bar-container">
      <div className="left-actions">
        <button className="action-btn download-btn" title="Download Conversation" onClick={handleDownload} disabled={!pdfData}>
          <DownloadOutlinedIcon />
        </button>
        <button className="action-btn" title="Attach file" onClick={handleAttach}>
          <AttachFileOutlinedIcon />
        </button>
        <button className="action-btn" title="Add location" onClick={handleLocation}>
          <LocationOnOutlinedIcon />
        </button>
      </div>

      <input
        type="text"
        className="chat-input-field"
        placeholder="Start typing to ask Water Copilot..."
        value={inputText}
        onKeyDown={handleKeyPress}
        onChange={(e) => setInputText((e.target as HTMLInputElement).value)}
      />

      <div className="right-actions">
        {isRecording && (
          <div className="recording-indicator"><div className="recording-dot"></div><span className="recording-text">Recording...</span></div>
        )}

        <button className={`action-btn mic-btn ${isRecording ? "recording" : ""}`} title={isRecording ? "Stop recording" : "Start recording"} onClick={handleMicClick}>
          {isRecording ? <StopIcon /> : <MicIcon />}
        </button>

        <button className={`send-btn ${inputText.trim() ? "active" : ""}`} onClick={handleSendText} disabled={!inputText.trim()}>
          <SendIcon />
        </button>
      </div>
    </div>
  );
};

export default withStreamlitConnection(ChatInputWidget);