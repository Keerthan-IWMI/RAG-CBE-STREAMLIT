import React, { useState, useEffect, useRef, useCallback } from "react";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import { useReactMediaRecorder } from "react-media-recorder";
// Using outlined icons for cleaner look
import SendIcon from "@mui/icons-material/ArrowUpward";
import MicIcon from "@mui/icons-material/MicNoneOutlined";
import StopIcon from "@mui/icons-material/Stop";
import DownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";
import AttachFileOutlinedIcon from "@mui/icons-material/AttachFileOutlined";
import LocationOnOutlinedIcon from "@mui/icons-material/LocationOnOutlined";
import './ChatInputWidget.css';

const ChatInputWidget: React.FC = () => {
  const [inputText, setInputText] = useState<string>("");
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Media recorder hook
  const {
    startRecording,
    stopRecording,
    mediaBlobUrl,
    clearBlobUrl,
    status,
  } = useReactMediaRecorder({ audio: true });

  // Sync local recording state with hook status
  useEffect(() => {
    setIsRecording(status === "recording");
  }, [status]);

  // Debug build marker to help confirm correct bundle is loaded in the browser
  useEffect(() => {
    console.log('[ChatInputWidget] Build v2 loaded');
  }, []);

  // Set frame height on mount
  useEffect(() => {
    Streamlit.setFrameHeight();
  }, []);

  // Send data to Streamlit
  const sendDataToStreamlit = useCallback((data: { text?: string; audio?: Uint8Array; audioFile?: Uint8Array }) => {
    Streamlit.setComponentValue(data);
  }, []);

  // Send audio blob as bytes
  const sendAudioBlobAsBytes = useCallback(async (blobUrl: string) => {
    if (!blobUrl) return;
    try {
      const response = await fetch(blobUrl);
      const blob = await response.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      sendDataToStreamlit({ audioFile: uint8Array });
      clearBlobUrl();
    } catch (error) {
      console.error("Error processing audio:", error);
    }
  }, [sendDataToStreamlit, clearBlobUrl]);

  // Watch for new audio blob
  useEffect(() => {
    if (mediaBlobUrl) {
      sendAudioBlobAsBytes(mediaBlobUrl);
    }
  }, [mediaBlobUrl, sendAudioBlobAsBytes]);

  // Handle input change
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputText(event.target.value);
  };

  // Handle Enter key
  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && inputText.trim().length > 0) {
      event.preventDefault();
      handleSend();
    }
  };

  // Handle send button click
  const handleSend = () => {
    if (inputText.trim().length > 0) {
      sendDataToStreamlit({ text: inputText });
      setInputText("");
    }
  };

  // Handle mic button click
  const handleMicClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="chat-bar-container">
      {/* Left action buttons */}
      <div className="left-actions">
        <button className="action-btn" title="Download">
          <DownloadOutlinedIcon />
        </button>
        <button className="action-btn" title="Attach file">
          <AttachFileOutlinedIcon />
        </button>
        <button className="action-btn" title="Location">
          <LocationOnOutlinedIcon />
        </button>
      </div>

      {/* Center input */}
      <input
        ref={inputRef}
        type="text"
        className="chat-input-field"
        placeholder="Start typing to ask Water Copilot..."
        value={inputText}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
      />

      {/* Recording indicator */}
      {isRecording && (
        <div className="recording-indicator">
          <span className="recording-dot"></span>
          <span className="recording-text">Recording...</span>
        </div>
      )}

      {/* Right action buttons */}
      <div className="right-actions">
        <button 
          className={`action-btn mic-btn ${isRecording ? 'recording' : ''}`} 
          onClick={handleMicClick}
          title={isRecording ? "Stop recording" : "Start recording"}
        >
          {isRecording ? <StopIcon /> : <MicIcon />}
        </button>
        <button 
          className={`send-btn ${inputText.trim().length > 0 ? 'active' : ''}`}
          onClick={handleSend}
          disabled={inputText.trim().length === 0}
          title="Send message"
        >
          <SendIcon />
        </button>
      </div>
    </div>
  );
};

export default withStreamlitConnection(ChatInputWidget);
