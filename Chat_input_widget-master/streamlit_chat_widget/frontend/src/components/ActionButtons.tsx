import React, { useRef, useState, useLayoutEffect } from "react";
import DownloadOutlinedIcon from "@mui/icons-material/FileDownloadOutlined";
import AttachFileOutlinedIcon from "@mui/icons-material/AttachFileOutlined";
import FilterListOutlinedIcon from "@mui/icons-material/FilterListOutlined";

interface ActionButtonsProps {
  onDownload: () => void;
  onAttach: () => void;
  onToggleFilter: () => void;
  showFilter: boolean;
  pdfDataAvailable: boolean;
  filterPopover?: React.ReactNode;
}

const ActionButtons: React.FC<ActionButtonsProps> = ({ onDownload, onAttach, onToggleFilter, showFilter, pdfDataAvailable, filterPopover }) => {
  const filterBtnRef = useRef<HTMLButtonElement | null>(null);
  const [anchor, setAnchor] = useState<{ left: number; top: number } | null>(null);

  useLayoutEffect(() => {
    const update = () => {
      const btn = filterBtnRef.current;
      if (!btn) return;
      if (!btn.parentElement) return;
      const btnRect = btn.getBoundingClientRect();
      const POP_WIDTH = 280;
      // compute anchor relative to viewport (absolute) - center of filter button
      let left = btnRect.left + btnRect.width / 2;
      // no absolute parent coordinates needed now; we compute left/top relative to viewport
      // keep the popover inside parent width
      // clamp against viewport boundaries
      const WIN_WIDTH = window.innerWidth;
      if (left + POP_WIDTH / 2 > WIN_WIDTH) left = WIN_WIDTH - POP_WIDTH / 2 - 8;
      if (left - POP_WIDTH / 2 < 0) left = POP_WIDTH / 2 + 8;
      const top = btnRect.top + btnRect.height / 2;
      setAnchor({ left, top });
      // No debug logs in production
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  return (
    <div className="left-actions">
      <button className="action-btn download-btn" title="Download Conversation" onClick={onDownload} disabled={!pdfDataAvailable}>
        <DownloadOutlinedIcon />
      </button>
      <button className="action-btn" title="Attach file" onClick={onAttach}>
        <AttachFileOutlinedIcon />
      </button>
      <button ref={filterBtnRef} className={`action-btn filter-btn ${showFilter ? "open" : ""}`} title="Filter results" onClick={() => onToggleFilter()}>
        <FilterListOutlinedIcon />
      </button>
      {filterPopover && React.isValidElement(filterPopover) ? React.cloneElement(filterPopover as any, { anchorOffset: anchor }) : filterPopover}
    </div>
  );
};

export default ActionButtons;
