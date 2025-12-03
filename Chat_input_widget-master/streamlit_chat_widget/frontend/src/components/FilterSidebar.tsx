import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom";

interface FilterState {
  year?: string;
  author?: string;
  keywords?: string;
}

interface FilterSidebarProps {
  visible: boolean;
  filters: FilterState;
  onChange: (k: keyof FilterState, v: string) => void;
  onApply: () => void;
  onCancel: () => void;
}

/**
 * Inject drawer styles into a target document (parent or current).
 * We inject inline styles so that the drawer renders correctly in the parent document
 * which doesn't have our CSS file.
 */
function injectDrawerStyles(doc: Document, styleId: string): HTMLStyleElement | null {
  if (doc.getElementById(styleId)) return null; // already injected
  const style = doc.createElement("style");
  style.id = styleId;
  style.textContent = `
    .filter-drawer-wrapper { position: fixed; inset: 0; z-index: 999999; pointer-events: none; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    .filter-drawer-wrapper.open { pointer-events: auto; }
    .drawer-overlay { position: fixed; inset: 0; background: rgba(7,16,22,0.42); opacity: 0; transition: opacity 180ms ease; pointer-events: none; }
    .drawer-overlay.open { opacity: 1; pointer-events: auto; }
    .filter-drawer { position: fixed; right: 0; top: 0; bottom: 0; width: 320px; max-width: 92vw; background: linear-gradient(180deg,#ffffff 0%,#f8fffd 100%); border-left: 1px solid rgba(6,182,212,0.08); box-shadow: -12px 0 40px rgba(6,182,212,0.10); transform: translateX(100%); transition: transform 240ms cubic-bezier(.4,0,.2,1); display: flex; flex-direction: column; pointer-events: auto; }
    .filter-drawer.open { transform: translateX(0); }
    .filter-drawer-header { display: flex; align-items: center; justify-content: space-between; padding: 18px 20px; border-bottom: 1px solid rgba(6,182,212,0.06); }
    .filter-drawer-header h3 { margin: 0; font-size: 17px; font-weight: 600; color: #0F172A; }
    .filter-close { border: none; background: transparent; color: #64748B; cursor: pointer; font-size: 20px; line-height: 1; padding: 4px; border-radius: 6px; transition: background 150ms, color 150ms; }
    .filter-close:hover { background: rgba(6,182,212,0.08); color: #0F766E; }
    .filter-drawer-body { padding: 16px 20px 24px; overflow-y: auto; flex: 1 1 auto; }
    .filter-row { display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px; }
    .filter-row span { font-size: 13px; font-weight: 500; color: #334155; }
    .filter-row input { width: 100%; height: 38px; border-radius: 8px; border: 1px solid #E2E8F0; padding: 8px 12px; font-size: 14px; color: #0F172A; background: #fff; transition: border-color 150ms, box-shadow 150ms; }
    .filter-row input:focus { outline: none; border-color: #06B6D4; box-shadow: 0 0 0 3px rgba(6,182,212,0.12); }
    .filter-row input::placeholder { color: #94A3B8; }
    .filter-drawer-footer { padding: 14px 20px; border-top: 1px solid rgba(6,182,212,0.05); display: flex; gap: 10px; justify-content: flex-end; }
    .filter-apply, .filter-cancel { border: none; padding: 9px 18px; border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer; transition: background 150ms, transform 100ms; }
    .filter-apply { background: linear-gradient(135deg, #06B6D4 0%, #0E9AA4 100%); color: #fff; }
    .filter-apply:hover { background: linear-gradient(135deg, #0891B2 0%, #0D9488 100%); }
    .filter-apply:active { transform: scale(0.97); }
    .filter-cancel { background: #F1F5F9; color: #334155; }
    .filter-cancel:hover { background: #E2E8F0; }
  `;
  doc.head.appendChild(style);
  return style;
}

const FilterSidebar: React.FC<FilterSidebarProps> = ({ visible, filters, onChange, onApply, onCancel }) => {
  const portalRef = useRef<HTMLDivElement | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const styleRef = useRef<HTMLStyleElement | null>(null);
  const [targetDoc, setTargetDoc] = useState<Document | null>(null);

  // Create portal container in parent document (Streamlit main page) if accessible, else fallback to iframe
  useEffect(() => {
    let doc: Document = document;
    try {
      // Attempt to access parent document (same-origin check)
      if (window.parent && window.parent.document && window.parent.document.body) {
        doc = window.parent.document;
      }
    } catch (e) {
      // Cross-origin — use iframe document
    }
    setTargetDoc(doc);

    // Create container div
    portalRef.current = doc.createElement("div");
    portalRef.current.id = "filter-drawer-portal-root";
    doc.body.appendChild(portalRef.current);

    // Inject styles into target document
    styleRef.current = injectDrawerStyles(doc, "filter-drawer-injected-styles");

    return () => {
      if (portalRef.current && portalRef.current.parentNode) {
        portalRef.current.parentNode.removeChild(portalRef.current);
      }
      if (styleRef.current && styleRef.current.parentNode) {
        styleRef.current.parentNode.removeChild(styleRef.current);
      }
      portalRef.current = null;
      styleRef.current = null;
    };
  }, []);

  // Focus first input when opened; handle Escape and click-outside
  useEffect(() => {
    if (!visible || !targetDoc) return;
    const t = window.setTimeout(() => {
      rootRef.current?.querySelector("input")?.focus();
    }, 60);

    const onDocClick = (e: MouseEvent) => {
      if (!rootRef.current) return;
      if (!(e.target instanceof Node)) return;
      if (!rootRef.current.contains(e.target)) onCancel();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };

    targetDoc.addEventListener("mousedown", onDocClick);
    targetDoc.addEventListener("keydown", onKeyDown);
    return () => {
      clearTimeout(t);
      targetDoc.removeEventListener("mousedown", onDocClick);
      targetDoc.removeEventListener("keydown", onKeyDown);
    };
  }, [visible, onCancel, targetDoc]);

  if (!portalRef.current) return null;

  const drawer = (
    <div className={`filter-drawer-wrapper ${visible ? "open" : ""}`}>
      <div className={`drawer-overlay ${visible ? "open" : ""}`} onClick={onCancel} aria-hidden />
      <div ref={rootRef} className={`filter-drawer ${visible ? "open" : ""}`} role="dialog" aria-modal="true" tabIndex={-1}>
        <div className="filter-drawer-header">
          <h3>Filters</h3>
          <button className="filter-close" onClick={onCancel} aria-label="Close">✕</button>
        </div>
        <div className="filter-drawer-body">
          <label className="filter-row">
            <span>Year</span>
            <input value={filters.year ?? ""} onChange={(e) => onChange("year", e.target.value)} placeholder="e.g. 2023" />
          </label>
          <label className="filter-row">
            <span>Author</span>
            <input value={filters.author ?? ""} onChange={(e) => onChange("author", e.target.value)} placeholder="Author name" />
          </label>
          <label className="filter-row">
            <span>Keywords</span>
            <input value={filters.keywords ?? ""} onChange={(e) => onChange("keywords", e.target.value)} placeholder="e.g. compost, reuse" />
          </label>
        </div>
        <div className="filter-drawer-footer">
          <button className="filter-apply" onClick={onApply}>Apply</button>
          <button className="filter-cancel" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  );

  return ReactDOM.createPortal(drawer, portalRef.current);
};

export default FilterSidebar;
