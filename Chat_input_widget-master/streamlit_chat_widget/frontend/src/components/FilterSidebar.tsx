import React, { useCallback, useRef, useState, useEffect } from "react";
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

function injectDrawerStyles(doc: Document, styleId: string): HTMLStyleElement | null {
  if (doc.getElementById(styleId)) return null;
  const style = doc.createElement("style");
  style.id = styleId;
  style.textContent = `
    .filter-drawer-wrapper { position: fixed; inset: 0; z-index: 999999; pointer-events: none; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    .filter-drawer-wrapper.open { pointer-events: auto; }
    .drawer-overlay { position: fixed; inset: 0; background: rgba(15,23,42,0.4); opacity: 0; transition: opacity 200ms ease; pointer-events: none; backdrop-filter: blur(2px); }
    .drawer-overlay.open { opacity: 1; pointer-events: auto; }
    .filter-drawer { position: fixed; right: 0; top: 0; bottom: 0; width: 340px; max-width: 90vw; background: #fff; border-left: 1px solid #e2e8f0; box-shadow: -8px 0 32px rgba(15,23,42,0.12); transform: translateX(100%); transition: transform 260ms cubic-bezier(.4,0,.2,1); display: flex; flex-direction: column; pointer-events: auto; }
    .filter-drawer.open { transform: translateX(0); }
    .filter-drawer-header { display: flex; align-items: center; justify-content: space-between; padding: 20px 24px; border-bottom: 1px solid #f1f5f9; background: #fafbfc; }
    .filter-drawer-header h3 { margin: 0; font-size: 18px; font-weight: 600; color: #0f172a; display: flex; align-items: center; gap: 8px; }
    .filter-drawer-header h3 .icon { font-size: 20px; }
    .filter-close { border: none; background: #f1f5f9; color: #64748b; cursor: pointer; font-size: 18px; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; transition: all 150ms; }
    .filter-close:hover { background: #e2e8f0; color: #0f172a; }
    .filter-drawer-body { padding: 20px 24px; overflow-y: auto; flex: 1 1 auto; }
    .filter-reset-btn { display: inline-flex; align-items: center; gap: 6px; margin-bottom: 20px; background: transparent; color: #0891b2; border: none; padding: 0; font-size: 13px; font-weight: 500; cursor: pointer; transition: color 150ms; }
    .filter-reset-btn:hover { color: #0e7490; text-decoration: underline; }
    .filter-section { margin-bottom: 24px; }
    .filter-section:last-child { margin-bottom: 0; }
    .filter-label { display: flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 10px; }
    .filter-label .icon { font-size: 16px; opacity: 0.7; }
    .filter-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
    .filter-chip { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; border-radius: 20px; padding: 6px 14px; font-size: 13px; font-weight: 500; cursor: pointer; transition: all 150ms; white-space: nowrap; }
    .filter-chip:hover { background: #e2e8f0; border-color: #cbd5e1; }
    .filter-chip.selected { background: #0891b2; color: #fff; border-color: #0891b2; }
    .filter-chip.selected:hover { background: #0e7490; border-color: #0e7490; }
    .filter-input { width: 100%; height: 42px; border-radius: 10px; border: 1px solid #e2e8f0; padding: 0 14px; font-size: 14px; color: #0f172a; background: #fff; transition: all 150ms; box-sizing: border-box; }
    .filter-input:focus { outline: none; border-color: #0891b2; box-shadow: 0 0 0 3px rgba(8,145,178,0.1); }
    .filter-input::placeholder { color: #94a3b8; }
    .filter-drawer-footer { padding: 16px 24px; border-top: 1px solid #f1f5f9; display: flex; gap: 12px; justify-content: flex-end; background: #fafbfc; }
    .filter-btn { border: none; padding: 10px 20px; border-radius: 10px; font-weight: 600; font-size: 14px; cursor: pointer; transition: all 150ms; }
    .filter-btn-apply { background: #0891b2; color: #fff; }
    .filter-btn-apply:hover { background: #0e7490; }
    .filter-btn-apply:active { transform: scale(0.98); }
    .filter-btn-cancel { background: #f1f5f9; color: #475569; }
    .filter-btn-cancel:hover { background: #e2e8f0; }
  `;
  doc.head.appendChild(style);
  return style;
}

const YEAR_PRESETS = ["2025", "2024", "2023", "2022", "Any"];
const KEYWORD_PRESETS = ["compost", "reuse", "wastewater", "biochar", "fertilizer"];

const FilterSidebar: React.FC<FilterSidebarProps> = ({ visible, filters, onChange, onApply, onCancel }) => {
  const portalRef = useRef<HTMLDivElement | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const styleRef = useRef<HTMLStyleElement | null>(null);
  const [targetDoc, setTargetDoc] = useState<Document | null>(null);

  const handleReset = useCallback(() => {
    onChange("year", "");
    onChange("author", "");
    onChange("keywords", "");
  }, [onChange]);

  useEffect(() => {
    let doc: Document = document;
    try {
      if (window.parent && window.parent.document && window.parent.document.body) {
        doc = window.parent.document;
      }
    } catch (e) {}
    setTargetDoc(doc);
    portalRef.current = doc.createElement("div");
    portalRef.current.id = "filter-drawer-portal-root";
    doc.body.appendChild(portalRef.current);
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

  return ReactDOM.createPortal(
    <div className={`filter-drawer-wrapper ${visible ? "open" : ""}`}>
      <div className={`drawer-overlay ${visible ? "open" : ""}`} onClick={onCancel} aria-hidden />
      <div ref={rootRef} className={`filter-drawer ${visible ? "open" : ""}`} role="dialog" aria-modal="true" tabIndex={-1}>
        <div className="filter-drawer-header">
          <h3><span className="icon">☰</span> Filters</h3>
          <button className="filter-close" onClick={onCancel} aria-label="Close">✕</button>
        </div>
        <div className="filter-drawer-body">
          <button className="filter-reset-btn" type="button" onClick={handleReset}>
            ↺ Reset All
          </button>

          {/* Year Section */}
          <div className="filter-section">
            <div className="filter-label">
              <span className="icon">📅</span> Year
            </div>
            <div className="filter-chips">
              {YEAR_PRESETS.map((label) => (
                <button
                  key={label}
                  className={`filter-chip ${filters.year === label ? "selected" : ""}`}
                  onClick={() => onChange("year", filters.year === label ? "" : label)}
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>
            <input
              className="filter-input"
              value={filters.year ?? ""}
              onChange={(e) => onChange("year", e.target.value)}
              placeholder="Or type a year..."
            />
          </div>

          {/* Author Section */}
          <div className="filter-section">
            <div className="filter-label">
              <span className="icon">👤</span> Author
            </div>
            <input
              className="filter-input"
              value={filters.author ?? ""}
              onChange={(e) => onChange("author", e.target.value)}
              placeholder="Enter author name..."
            />
          </div>

          {/* Keywords Section */}
          <div className="filter-section">
            <div className="filter-label">
              <span className="icon">🏷️</span> Keywords
            </div>
            <div className="filter-chips">
              {KEYWORD_PRESETS.map((kw) => {
                const kws = filters.keywords ? filters.keywords.split(",").map(s => s.trim().toLowerCase()) : [];
                const isSelected = kws.includes(kw.toLowerCase());
                return (
                  <button
                    key={kw}
                    className={`filter-chip ${isSelected ? "selected" : ""}`}
                    onClick={() => {
                      if (isSelected) {
                        onChange("keywords", kws.filter(k => k !== kw.toLowerCase()).join(", "));
                      } else {
                        onChange("keywords", [...kws, kw].filter(Boolean).join(", "));
                      }
                    }}
                    type="button"
                  >
                    {kw}
                  </button>
                );
              })}
            </div>
            <input
              className="filter-input"
              value={filters.keywords ?? ""}
              onChange={(e) => onChange("keywords", e.target.value)}
              placeholder="Or type keywords..."
            />
          </div>
        </div>
        <div className="filter-drawer-footer">
          <button className="filter-btn filter-btn-cancel" onClick={onCancel}>Cancel</button>
          <button className="filter-btn filter-btn-apply" onClick={onApply}>Apply Filters</button>
        </div>
      </div>
    </div>,
    portalRef.current
  );
};

export default FilterSidebar;

