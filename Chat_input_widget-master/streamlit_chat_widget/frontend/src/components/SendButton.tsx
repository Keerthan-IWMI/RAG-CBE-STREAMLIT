import React, { useState } from "react";

// Clean SVG arrow icon
const ArrowUpIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 19V5M5 12l7-7 7 7"/>
  </svg>
);

interface SendButtonProps {
  active: boolean;
  onClick: () => void;
  darkMode?: boolean;
}

const SendButton: React.FC<SendButtonProps> = ({ active, onClick, darkMode = false }) => {
  const [isHovered, setIsHovered] = useState(false);

  const colors = darkMode ? {
    bg: active ? '#14b8a6' : '#334155',
    bgHover: active ? '#0d9488' : '#475569',
    text: active ? '#ffffff' : '#64748b',
  } : {
    bg: active ? '#0F766E' : '#e2e8f0',
    bgHover: active ? '#0d6560' : '#cbd5e1',
    text: active ? '#ffffff' : '#94a3b8',
  };

  const style: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 40,
    height: 40,
    border: 'none',
    background: isHovered && active ? colors.bgHover : colors.bg,
    borderRadius: 12,
    cursor: active ? 'pointer' : 'not-allowed',
    color: colors.text,
    transition: 'all 0.15s ease',
    padding: 0,
    transform: isHovered && active ? 'scale(1.05)' : 'scale(1)',
  };

  return (
    <button 
      style={style} 
      onClick={onClick} 
      disabled={!active}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      title="Send message"
    >
      <ArrowUpIcon />
    </button>
  );
};

export default SendButton;
