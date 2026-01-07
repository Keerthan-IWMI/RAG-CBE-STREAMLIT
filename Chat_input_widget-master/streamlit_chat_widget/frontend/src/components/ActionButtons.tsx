import React, { useRef, useState, useLayoutEffect } from "react";

// Clean SVG icons to match the React sidebar aesthetic
const Icons = {
  download: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
    </svg>
  ),
  attach: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/>
    </svg>
  ),
  filter: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
    </svg>
  ),
  support: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/>
    </svg>
  ),
};

interface ActionButtonsProps {
  onDownload: () => void;
  onAttach: () => void;
  onToggleFilter: () => void;
  onSupport: () => void;
  showFilter: boolean;
  pdfDataAvailable: boolean;
  filterPopover?: React.ReactNode;
  darkMode?: boolean;
}

const ActionButtons: React.FC<ActionButtonsProps> = ({ onDownload, onAttach, onToggleFilter, onSupport, showFilter, pdfDataAvailable, filterPopover, darkMode = false }) => {
  const filterBtnRef = useRef<HTMLButtonElement | null>(null);
  const [anchor, setAnchor] = useState<{ left: number; top: number } | null>(null);
  const [hoveredBtn, setHoveredBtn] = useState<string | null>(null);

  const colors = darkMode ? {
    text: '#94a3b8',
    textHover: '#14b8a6',
    bgHover: 'rgba(20, 184, 166, 0.12)',
    accent: '#14b8a6',
  } : {
    text: '#64748b',
    textHover: '#0F766E',
    bgHover: 'rgba(15, 118, 110, 0.08)',
    accent: '#0F766E',
  };

  useLayoutEffect(() => {
    const update = () => {
      const btn = filterBtnRef.current;
      if (!btn) return;
      if (!btn.parentElement) return;
      const btnRect = btn.getBoundingClientRect();
      const POP_WIDTH = 280;
      let left = btnRect.left + btnRect.width / 2;
      const WIN_WIDTH = window.innerWidth;
      if (left + POP_WIDTH / 2 > WIN_WIDTH) left = WIN_WIDTH - POP_WIDTH / 2 - 8;
      if (left - POP_WIDTH / 2 < 0) left = POP_WIDTH / 2 + 8;
      const top = btnRect.top + btnRect.height / 2;
      setAnchor({ left, top });
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const getButtonStyle = (id: string, disabled?: boolean): React.CSSProperties => ({
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 38,
    height: 38,
    border: 'none',
    background: hoveredBtn === id && !disabled ? colors.bgHover : 'transparent',
    borderRadius: 10,
    cursor: disabled ? 'not-allowed' : 'pointer',
    color: disabled ? '#cbd5e1' : (hoveredBtn === id ? colors.textHover : colors.text),
    transition: 'all 0.15s ease',
    padding: 0,
    opacity: disabled ? 0.5 : 1,
  });

  return (
    <div className="left-actions" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <button 
        style={getButtonStyle('download', !pdfDataAvailable)} 
        title="Download Conversation" 
        onClick={onDownload} 
        disabled={!pdfDataAvailable}
        onMouseEnter={() => setHoveredBtn('download')}
        onMouseLeave={() => setHoveredBtn(null)}
      >
        {Icons.download}
      </button>
      <button 
        style={getButtonStyle('attach')} 
        title="Attach file" 
        onClick={onAttach}
        onMouseEnter={() => setHoveredBtn('attach')}
        onMouseLeave={() => setHoveredBtn(null)}
      >
        {Icons.attach}
      </button>
      <button 
        ref={filterBtnRef} 
        style={{
          ...getButtonStyle('filter'),
          color: showFilter ? colors.accent : (hoveredBtn === 'filter' ? colors.textHover : colors.text),
          background: showFilter ? colors.bgHover : (hoveredBtn === 'filter' ? colors.bgHover : 'transparent'),
        }} 
        title="Filter results" 
        onClick={() => onToggleFilter()}
        onMouseEnter={() => setHoveredBtn('filter')}
        onMouseLeave={() => setHoveredBtn(null)}
      >
        {Icons.filter}
      </button>
      <button 
        style={getButtonStyle('support')} 
        title="Contact Support" 
        onClick={onSupport}
        onMouseEnter={() => setHoveredBtn('support')}
        onMouseLeave={() => setHoveredBtn(null)}
      >
        {Icons.support}
      </button>
      {filterPopover && React.isValidElement(filterPopover) ? React.cloneElement(filterPopover as any, { anchorOffset: anchor }) : filterPopover}
    </div>
  );
};

export default ActionButtons;
