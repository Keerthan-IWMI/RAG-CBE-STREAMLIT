import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";
import { Streamlit } from "streamlit-component-lib";

interface Conversation {
  id: string;
  title: string;
}

interface ReactSidebarProps {
  darkMode?: boolean;
  userEmail?: string;
  userName?: string;
  conversations?: Conversation[];
  onCollapseChange?: (collapsed: boolean) => void;
}

// SVG Icons - clean, minimal, consistent
const Icons = {
  newChat: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 5v14M5 12h14"/>
    </svg>
  ),
  clear: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z"/>
    </svg>
  ),
  chat: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2v10z"/>
    </svg>
  ),
  download: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
    </svg>
  ),
  sun: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
    </svg>
  ),
  moon: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
    </svg>
  ),
  logout: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5-5-5M21 12H9"/>
    </svg>
  ),
  chevronLeft: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 18l-6-6 6-6"/>
    </svg>
  ),
  chevronRight: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 18l6-6-6-6"/>
    </svg>
  ),
  moreVert: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <circle cx="12" cy="5" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="19" r="2"/>
    </svg>
  ),
  edit: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
    </svg>
  ),
  trash: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z"/>
    </svg>
  ),
  check: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 6L9 17l-5-5"/>
    </svg>
  ),
  x: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6L6 18M6 6l12 12"/>
    </svg>
  ),
};

const ReactSidebar: React.FC<ReactSidebarProps> = ({ 
  darkMode = false, 
  userEmail = "",
  userName = "",
  conversations = [],
  onCollapseChange
}) => {
  const portalRef = useRef<HTMLDivElement | null>(null);
  const [mounted, setMounted] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const handleCollapse = (collapsed: boolean) => {
    setIsCollapsed(collapsed);
    onCollapseChange?.(collapsed);
  };

  const colors = darkMode ? {
    bg: '#1e293b',
    bgHover: '#334155',
    border: '#334155',
    text: '#f1f5f9',
    textMuted: '#94a3b8',
    accent: '#14b8a6',
    accentBg: 'rgba(20, 184, 166, 0.12)',
    danger: '#f87171',
    dangerBg: 'rgba(248, 113, 113, 0.12)',
  } : {
    bg: '#ffffff',
    bgHover: '#f1f5f9',
    border: '#e2e8f0',
    text: '#0f172a',
    textMuted: '#64748b',
    accent: '#0F766E',
    accentBg: 'rgba(15, 118, 110, 0.08)',
    danger: '#ef4444',
    dangerBg: 'rgba(239, 68, 68, 0.08)',
  };

  useEffect(() => {
    let doc: Document = document;
    try {
      if (window.parent && window.parent.document && window.parent.document.body) {
        doc = window.parent.document;
      }
    } catch (e) {}
    
    const existing = doc.getElementById("react-sidebar-portal");
    if (existing) existing.remove();
    
    const container = doc.createElement("div");
    container.id = "react-sidebar-portal";
    doc.body.appendChild(container);
    portalRef.current = container;
    setMounted(true);

    return () => {
      if (portalRef.current && portalRef.current.parentNode) {
        portalRef.current.parentNode.removeChild(portalRef.current);
      }
    };
  }, []);

  // Close menu when clicking outside
  useEffect(() => {
    if (!menuOpenId) return;
    const handleClick = () => setMenuOpenId(null);
    window.parent?.document.addEventListener('click', handleClick);
    return () => window.parent?.document.removeEventListener('click', handleClick);
  }, [menuOpenId]);

  if (!mounted || !portalRef.current) return null;

  const handleAction = (action: string, payload?: any) => {
    if (payload) {
      Streamlit.setComponentValue({ action, ...payload });
    } else {
      Streamlit.setComponentValue({ action });
    }
  };

  const handleRename = (convId: string, newTitle: string) => {
    Streamlit.setComponentValue({ action: `rename_${convId}`, newTitle });
    setEditingId(null);
    setEditTitle("");
  };

  const handleDelete = (convId: string) => {
    Streamlit.setComponentValue({ action: `delete_${convId}` });
    setMenuOpenId(null);
  };

  const sidebarWidth = 305; // increased by ~1.5%

  const sidebarStyle: React.CSSProperties = {
    position: 'fixed',
    top: 0,
    left: isCollapsed ? -sidebarWidth : 0,
    bottom: 0,
    width: sidebarWidth,
    background: colors.bg,
    borderRight: `1px solid ${colors.border}`,
    boxShadow: darkMode ? '4px 0 24px rgba(0,0,0,0.4)' : '4px 0 24px rgba(0,0,0,0.06)',
    zIndex: 999998,
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    display: 'flex',
    flexDirection: 'column',
    transition: 'left 0.25s cubic-bezier(0.4, 0, 0.2, 1), background 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease',
    overflow: 'hidden',
  };

  const toggleBtnStyle: React.CSSProperties = {
    position: 'fixed',
    top: 18,
    left: isCollapsed ? 18 : sidebarWidth + 18,
    width: 40,
    height: 40,
    borderRadius: 12,
    background: colors.bg,
    border: `1px solid ${colors.border}`,
    boxShadow: darkMode ? '0 4px 12px rgba(0,0,0,0.4)' : '0 4px 12px rgba(0,0,0,0.1)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    zIndex: 999999,
    color: colors.accent,
    transition: 'left 0.25s cubic-bezier(0.4, 0, 0.2, 1), background 0.3s ease, border-color 0.3s ease, transform 0.15s ease, color 0.3s ease',
  };

  const profileStyle: React.CSSProperties = {
    padding: '28px 24px',
    borderBottom: `1px solid ${colors.border}`,
    display: 'flex',
    alignItems: 'center',
    gap: 16,
  };

  const avatarStyle: React.CSSProperties = {
    width: 48,
    height: 48,
    borderRadius: '50%',
    background: `linear-gradient(135deg, ${colors.accent}, ${darkMode ? '#0d9488' : '#06b6d4'})`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontWeight: 600,
    fontSize: 20,
    flexShrink: 0,
  };

  const sectionStyle: React.CSSProperties = {
    padding: '20px 20px',
    flex: 1,
    overflowY: 'auto',
    overflowX: 'hidden',
  };

  const sectionTitleStyle: React.CSSProperties = {
    fontSize: 11,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: colors.textMuted,
    marginBottom: 10,
    marginTop: 20,
    paddingLeft: 12,
  };

  const MenuItem = ({ id, icon, label, onClick, danger }: { id: string; icon: React.ReactNode; label: string; onClick?: () => void; danger?: boolean }) => {
    const isHovered = hoveredItem === id;
    const hoverColor = danger ? colors.danger : colors.accent;
    const hoverBg = danger ? colors.dangerBg : colors.accentBg;
    
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '12px 14px',
          borderRadius: 10,
          cursor: 'pointer',
          color: isHovered ? hoverColor : colors.text,
          fontSize: 14,
          fontWeight: 500,
          background: isHovered ? hoverBg : 'transparent',
          transition: 'all 0.15s ease',
          marginBottom: 4,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
        onClick={onClick}
        onMouseEnter={() => setHoveredItem(id)}
        onMouseLeave={() => setHoveredItem(null)}
      >
        <span style={{ display: 'flex', alignItems: 'center', color: isHovered ? hoverColor : colors.textMuted, flexShrink: 0 }}>
          {icon}
        </span>
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</span>
      </div>
    );
  };

  // Conversation item with menu
  const ConversationItem = ({ conv, idx }: { conv: Conversation; idx: number }) => {
    const isHovered = hoveredItem === `conv-${idx}`;
    const isMenuOpen = menuOpenId === conv.id;
    const isEditing = editingId === conv.id;
    
    if (isEditing) {
      return (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '8px 12px',
          marginBottom: 4,
        }}>
          <input
            type="text"
            value={editTitle}
            onChange={(e) => setEditTitle(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleRename(conv.id, editTitle);
              if (e.key === 'Escape') { setEditingId(null); setEditTitle(""); }
            }}
            autoFocus
            style={{
              flex: 1,
              padding: '6px 10px',
              border: `1px solid ${colors.accent}`,
              borderRadius: 6,
              fontSize: 13,
              background: colors.bg,
              color: colors.text,
              outline: 'none',
            }}
          />
          <button
            onClick={() => handleRename(conv.id, editTitle)}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 28,
              height: 28,
              border: 'none',
              borderRadius: 6,
              background: colors.accent,
              color: 'white',
              cursor: 'pointer',
            }}
          >
            {Icons.check}
          </button>
          <button
            onClick={() => { setEditingId(null); setEditTitle(""); }}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 28,
              height: 28,
              border: `1px solid ${colors.border}`,
              borderRadius: 6,
              background: 'transparent',
              color: colors.textMuted,
              cursor: 'pointer',
            }}
          >
            {Icons.x}
          </button>
        </div>
      );
    }

    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '10px 12px',
          borderRadius: 10,
          cursor: 'pointer',
          color: isHovered ? colors.accent : colors.text,
          fontSize: 14,
          fontWeight: 500,
          background: isHovered ? colors.accentBg : 'transparent',
          transition: 'all 0.15s ease',
          marginBottom: 4,
          position: 'relative',
        }}
        onMouseEnter={() => setHoveredItem(`conv-${idx}`)}
        onMouseLeave={() => setHoveredItem(null)}
      >
        <span 
          style={{ display: 'flex', alignItems: 'center', color: isHovered ? colors.accent : colors.textMuted, flexShrink: 0 }}
          onClick={() => handleAction(`load_${conv.id}`)}
        >
          {Icons.chat}
        </span>
        <span 
          style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
          onClick={() => handleAction(`load_${conv.id}`)}
        >
          {conv.title.length > 22 ? conv.title.slice(0, 22) + '...' : conv.title}
        </span>
        
        {/* Three-dot menu button */}
        {(isHovered || isMenuOpen) && (
          <div style={{ position: 'relative' }}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setMenuOpenId(isMenuOpen ? null : conv.id);
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 24,
                height: 24,
                border: 'none',
                borderRadius: 4,
                background: isMenuOpen ? colors.bgHover : 'transparent',
                color: colors.textMuted,
                cursor: 'pointer',
              }}
            >
              {Icons.moreVert}
            </button>
            
            {/* Dropdown menu */}
            {isMenuOpen && (
              <div
                onClick={(e) => e.stopPropagation()}
                style={{
                  position: 'absolute',
                  top: '100%',
                  right: 0,
                  marginTop: 4,
                  background: colors.bg,
                  border: `1px solid ${colors.border}`,
                  borderRadius: 8,
                  boxShadow: darkMode ? '0 8px 24px rgba(0,0,0,0.5)' : '0 8px 24px rgba(0,0,0,0.15)',
                  zIndex: 10000,
                  minWidth: 120,
                  overflow: 'hidden',
                }}
              >
                <div
                  onClick={() => {
                    setEditingId(conv.id);
                    setEditTitle(conv.title);
                    setMenuOpenId(null);
                  }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    padding: '10px 14px',
                    cursor: 'pointer',
                    color: colors.text,
                    fontSize: 13,
                    transition: 'background 0.1s',
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = colors.bgHover}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  {Icons.edit}
                  <span>Rename</span>
                </div>
                <div
                  onClick={() => handleDelete(conv.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    padding: '10px 14px',
                    cursor: 'pointer',
                    color: colors.danger,
                    fontSize: 13,
                    transition: 'background 0.1s',
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = colors.dangerBg}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  {Icons.trash}
                  <span>Delete</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const userInitial = userName ? userName[0].toUpperCase() : (userEmail ? userEmail[0].toUpperCase() : 'G');
  const displayName = userName || (userEmail ? userEmail.split('@')[0] : 'Guest');

  return ReactDOM.createPortal(
    <>
      <div
        style={{
          ...toggleBtnStyle,
          background: hoveredItem === 'toggle' ? colors.bgHover : colors.bg,
          transform: hoveredItem === 'toggle' ? 'scale(1.05)' : 'scale(1)',
        }}
        onClick={() => handleCollapse(!isCollapsed)}
        onMouseEnter={() => setHoveredItem('toggle')}
        onMouseLeave={() => setHoveredItem(null)}
      >
        {isCollapsed ? Icons.chevronRight : Icons.chevronLeft}
      </div>

      <div style={sidebarStyle}>
        <div style={profileStyle}>
          <div style={avatarStyle}>{userInitial}</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ 
              fontSize: 15, 
              fontWeight: 600, 
              color: colors.text, 
              whiteSpace: 'nowrap', 
              overflow: 'hidden', 
              textOverflow: 'ellipsis',
              marginBottom: 2,
            }}>
              {displayName}
            </div>
            <div style={{ 
              fontSize: 13, 
              color: colors.textMuted, 
              whiteSpace: 'nowrap', 
              overflow: 'hidden', 
              textOverflow: 'ellipsis' 
            }}>
              {userEmail || 'Guest Session'}
            </div>
          </div>
        </div>

        <div style={sectionStyle}>
          <div style={{ ...sectionTitleStyle, marginTop: 0 }}>Chat</div>
          <MenuItem id="new" icon={Icons.newChat} label="New Chat" onClick={() => handleAction('new_chat')} />
          <MenuItem id="clear" icon={Icons.clear} label="Clear History" onClick={() => handleAction('clear_chat')} />

          {conversations.length > 0 && (
            <>
              <div style={sectionTitleStyle}>Recent Conversations</div>
              {conversations.slice(0, 8).map((conv, idx) => (
                <ConversationItem key={conv.id} conv={conv} idx={idx} />
              ))}
            </>
          )}

          <div style={sectionTitleStyle}>Export</div>
          <MenuItem id="pdf" icon={Icons.download} label="Download PDF" onClick={() => handleAction('download_pdf')} />
        </div>

        <div style={{ 
          padding: '16px 20px', 
          borderTop: `1px solid ${colors.border}`,
          background: darkMode ? 'rgba(0,0,0,0.15)' : 'rgba(0,0,0,0.02)',
        }}>
          <MenuItem 
            id="theme" 
            icon={darkMode ? Icons.sun : Icons.moon} 
            label={darkMode ? "Light Mode" : "Dark Mode"} 
            onClick={() => handleAction('set_theme', { theme: darkMode ? 'light' : 'dark' })} 
          />
          <MenuItem id="signout" icon={Icons.logout} label="Sign Out" onClick={() => handleAction('sign_out')} />
        </div>
      </div>
    </>,
    portalRef.current
  );
};

export default ReactSidebar;
