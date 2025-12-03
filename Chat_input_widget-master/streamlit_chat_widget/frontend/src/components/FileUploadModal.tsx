import React, { useState, useCallback, useRef, useEffect } from "react";
import ReactDOM from "react-dom";

interface FileUploadModalProps {
  visible: boolean;
  onClose: () => void;
  onFileSelect?: (file: File) => void;
}

function injectModalStyles(doc: Document, styleId: string): HTMLStyleElement | null {
  if (doc.getElementById(styleId)) return null;
  const style = doc.createElement("style");
  style.id = styleId;
  style.textContent = `
    .file-upload-modal-wrapper { position: fixed; inset: 0; z-index: 999999; pointer-events: none; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; align-items: center; justify-content: center; }
    .file-upload-modal-wrapper.open { pointer-events: auto; }
    .file-upload-overlay { position: fixed; inset: 0; background: rgba(15,23,42,0.5); opacity: 0; transition: opacity 200ms ease; pointer-events: none; backdrop-filter: blur(4px); }
    .file-upload-overlay.open { opacity: 1; pointer-events: auto; }
    .file-upload-modal { background: #fff; border-radius: 16px; box-shadow: 0 20px 60px rgba(15,23,42,0.2); width: 440px; max-width: 92vw; transform: scale(0.95) translateY(10px); opacity: 0; transition: all 200ms cubic-bezier(.4,0,.2,1); pointer-events: auto; overflow: hidden; }
    .file-upload-modal.open { transform: scale(1) translateY(0); opacity: 1; }
    .file-upload-header { display: flex; align-items: center; justify-content: space-between; padding: 18px 24px; border-bottom: 1px solid #f1f5f9; }
    .file-upload-header h3 { margin: 0; font-size: 17px; font-weight: 600; color: #0f172a; }
    .file-upload-close { border: none; background: #f1f5f9; color: #64748b; cursor: pointer; font-size: 16px; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; transition: all 150ms; }
    .file-upload-close:hover { background: #e2e8f0; color: #0f172a; }
    .file-upload-body { padding: 24px; }
    .file-drop-zone { border: 2px dashed #cbd5e1; border-radius: 12px; padding: 40px 24px; text-align: center; transition: all 200ms; cursor: pointer; background: #fafbfc; }
    .file-drop-zone:hover { border-color: #0891b2; background: #f0fdfa; }
    .file-drop-zone.dragging { border-color: #0891b2; background: #ecfeff; border-style: solid; }
    .file-drop-zone.has-file { border-color: #10b981; background: #ecfdf5; border-style: solid; }
    .drop-icon { font-size: 48px; margin-bottom: 12px; opacity: 0.6; }
    .drop-title { font-size: 15px; font-weight: 600; color: #334155; margin-bottom: 6px; }
    .drop-subtitle { font-size: 13px; color: #64748b; }
    .file-selected { display: flex; align-items: center; gap: 12px; padding: 12px 16px; background: #f0fdf4; border-radius: 10px; margin-top: 16px; }
    .file-selected-icon { font-size: 24px; }
    .file-selected-info { flex: 1; text-align: left; }
    .file-selected-name { font-size: 14px; font-weight: 500; color: #0f172a; word-break: break-all; }
    .file-selected-size { font-size: 12px; color: #64748b; }
    .file-remove { border: none; background: #fef2f2; color: #ef4444; width: 28px; height: 28px; border-radius: 6px; cursor: pointer; font-size: 14px; transition: all 150ms; }
    .file-remove:hover { background: #fee2e2; }
    .file-upload-footer { padding: 16px 24px; border-top: 1px solid #f1f5f9; display: flex; gap: 12px; justify-content: flex-end; }
    .file-upload-btn { border: none; padding: 10px 20px; border-radius: 10px; font-weight: 600; font-size: 14px; cursor: pointer; transition: all 150ms; }
    .file-upload-btn-cancel { background: #f1f5f9; color: #475569; }
    .file-upload-btn-cancel:hover { background: #e2e8f0; }
    .file-upload-btn-upload { background: #0891b2; color: #fff; }
    .file-upload-btn-upload:hover { background: #0e7490; }
    .file-upload-btn-upload:disabled { background: #cbd5e1; cursor: not-allowed; }
  `;
  doc.head.appendChild(style);
  return style;
}

const FileUploadModal: React.FC<FileUploadModalProps> = ({ visible, onClose, onFileSelect }) => {
  const portalRef = useRef<HTMLDivElement | null>(null);
  const styleRef = useRef<HTMLStyleElement | null>(null);
  const [targetDoc, setTargetDoc] = useState<Document | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let doc: Document = document;
    try {
      if (window.parent && window.parent.document && window.parent.document.body) {
        doc = window.parent.document;
      }
    } catch (e) {}
    setTargetDoc(doc);
    portalRef.current = doc.createElement("div");
    portalRef.current.id = "file-upload-modal-portal";
    doc.body.appendChild(portalRef.current);
    styleRef.current = injectModalStyles(doc, "file-upload-modal-styles");
    return () => {
      if (portalRef.current?.parentNode) {
        portalRef.current.parentNode.removeChild(portalRef.current);
      }
      if (styleRef.current?.parentNode) {
        styleRef.current.parentNode.removeChild(styleRef.current);
      }
      portalRef.current = null;
      styleRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!visible) {
      setSelectedFile(null);
      setIsDragging(false);
    }
  }, [visible]);

  useEffect(() => {
    if (!visible || !targetDoc) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    targetDoc.addEventListener("keydown", onKeyDown);
    return () => targetDoc.removeEventListener("keydown", onKeyDown);
  }, [visible, onClose, targetDoc]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleZoneClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleUpload = () => {
    if (selectedFile && onFileSelect) {
      onFileSelect(selectedFile);
    }
    onClose();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  if (!portalRef.current) return null;

  return ReactDOM.createPortal(
    <div className={`file-upload-modal-wrapper ${visible ? "open" : ""}`}>
      <div className={`file-upload-overlay ${visible ? "open" : ""}`} onClick={onClose} />
      <div className={`file-upload-modal ${visible ? "open" : ""}`}>
        <div className="file-upload-header">
          <h3>📎 Attach File</h3>
          <button className="file-upload-close" onClick={onClose}>✕</button>
        </div>
        <div className="file-upload-body">
          <div
            className={`file-drop-zone ${isDragging ? "dragging" : ""} ${selectedFile ? "has-file" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleZoneClick}
          >
            <div className="drop-icon">{selectedFile ? "✅" : "📄"}</div>
            <div className="drop-title">
              {selectedFile ? "File selected!" : "Drag & drop your file here"}
            </div>
            <div className="drop-subtitle">
              {selectedFile ? "Click to change file" : "or click to browse"}
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            style={{ display: "none" }}
            onChange={handleFileInput}
            accept=".pdf,.doc,.docx,.txt,.csv,.xlsx,.xls"
          />
          {selectedFile && (
            <div className="file-selected">
              <span className="file-selected-icon">📄</span>
              <div className="file-selected-info">
                <div className="file-selected-name">{selectedFile.name}</div>
                <div className="file-selected-size">{formatFileSize(selectedFile.size)}</div>
              </div>
              <button className="file-remove" onClick={handleRemoveFile} title="Remove file">✕</button>
            </div>
          )}
        </div>
        <div className="file-upload-footer">
          <button className="file-upload-btn file-upload-btn-cancel" onClick={onClose}>Cancel</button>
          <button 
            className="file-upload-btn file-upload-btn-upload" 
            onClick={handleUpload}
            disabled={!selectedFile}
          >
            Attach
          </button>
        </div>
      </div>
    </div>,
    portalRef.current
  );
};

export default FileUploadModal;
