import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";

interface ReactHeaderProps {
  darkMode?: boolean;
  sidebarCollapsed?: boolean;
}

const ReactHeader: React.FC<ReactHeaderProps> = ({ darkMode = false, sidebarCollapsed = false }) => {
  const portalRef = useRef<HTMLDivElement | null>(null);
  const [mounted, setMounted] = useState(false);

  const sidebarWidth = 280;
  
  const colors = darkMode ? {
    bg: '#1e293b',
    text: '#f1f5f9',
    textMuted: '#94a3b8',
    accent: '#14b8a6',
    border: '#334155',
  } : {
    bg: '#ffffff',
    text: '#0f172a',
    textMuted: '#64748b',
    accent: '#0F766E',
    border: '#e2e8f0',
  };

  useEffect(() => {
    let doc: Document = document;
    try {
      if (window.parent && window.parent.document && window.parent.document.body) {
        doc = window.parent.document;
      }
    } catch (e) {}
    
    const existing = doc.getElementById("react-header-portal");
    if (existing) existing.remove();
    
    const container = doc.createElement("div");
    container.id = "react-header-portal";
    doc.body.appendChild(container);
    portalRef.current = container;
    setMounted(true);

    return () => {
      if (portalRef.current && portalRef.current.parentNode) {
        portalRef.current.parentNode.removeChild(portalRef.current);
      }
    };
  }, []);

  if (!mounted || !portalRef.current) return null;

  const headerStyle: React.CSSProperties = {
    position: 'fixed',
    top: 0,
    left: sidebarCollapsed ? 0 : sidebarWidth,
    right: 0,
    height: 64,
    background: colors.bg,
    borderBottom: `1px solid ${colors.border}`,
    boxShadow: darkMode ? '0 2px 12px rgba(0,0,0,0.3)' : '0 2px 12px rgba(0,0,0,0.04)',
    zIndex: 999997,
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '0 32px',
    transition: 'left 0.25s cubic-bezier(0.4, 0, 0.2, 1), background 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease',
  };

  // Leaf/plant icon as SVG for consistency
  const LeafIcon = () => (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={colors.accent} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
      <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
    </svg>
  );

  return ReactDOM.createPortal(
    <div style={headerStyle}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <LeafIcon />
        <span style={{ 
          fontSize: 20, 
          fontWeight: 700, 
          color: colors.accent,
          letterSpacing: '-0.3px',
        }}>
          CircularIQ
        </span>
        <span style={{ 
          fontSize: 14, 
          color: colors.textMuted,
          marginLeft: 16,
          paddingLeft: 16,
          borderLeft: `1px solid ${colors.border}`,
          fontWeight: 400,
        }}>
          Circular Bioeconomy Decision Support
        </span>
      </div>
    </div>,
    portalRef.current
  );
};

export default ReactHeader;
