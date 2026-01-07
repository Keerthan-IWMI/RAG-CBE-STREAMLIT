import React, { useState, useRef } from "react";

// Clean SVG icons
const MicIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/>
    <path d="M19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8"/>
  </svg>
);

const StopIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <rect x="6" y="6" width="12" height="12" rx="2"/>
  </svg>
);

interface MicButtonProps {
  onSendAudio: (base64: string) => void;
  onRecordingChange?: (isRecording: boolean) => void;
  darkMode?: boolean;
}

const MicButton: React.FC<MicButtonProps> = ({ onSendAudio, onRecordingChange, darkMode = false }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const handleClick = async () => {
    if (isRecording && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      onRecordingChange && onRecordingChange(false);
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
          onRecordingChange && onRecordingChange(false);
          return;
        }
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = reader.result as string;
          onSendAudio(base64String);
          audioChunksRef.current = [];
        };
        reader.readAsDataURL(audioBlob);
        stream.getTracks().forEach((t) => t.stop());
        onRecordingChange && onRecordingChange(false);
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      onRecordingChange && onRecordingChange(true);
    } catch (err) {
      console.error("Mic access denied:", err);
    }
  };

  const colors = darkMode ? {
    text: '#94a3b8',
    textHover: '#14b8a6',
    bgHover: 'rgba(20, 184, 166, 0.12)',
  } : {
    text: '#64748b',
    textHover: '#0F766E',
    bgHover: 'rgba(15, 118, 110, 0.08)',
  };

  const buttonStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 38,
    height: 38,
    border: 'none',
    background: isRecording ? '#fef2f2' : (isHovered ? colors.bgHover : 'transparent'),
    borderRadius: 10,
    cursor: 'pointer',
    color: isRecording ? '#dc2626' : (isHovered ? colors.textHover : colors.text),
    transition: 'all 0.15s ease',
    padding: 0,
  };

  return (
    <button 
      style={buttonStyle} 
      title={isRecording ? "Stop recording" : "Start recording"} 
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {isRecording ? <StopIcon /> : <MicIcon />}
    </button>
  );
};

export default MicButton;
