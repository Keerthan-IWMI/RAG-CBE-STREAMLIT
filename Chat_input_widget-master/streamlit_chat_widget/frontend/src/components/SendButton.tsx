import React from "react";
import SendIcon from "@mui/icons-material/ArrowUpward";

interface SendButtonProps {
  active: boolean;
  onClick: () => void;
}

const SendButton: React.FC<SendButtonProps> = ({ active, onClick }) => {
  return (
    <button className={`send-btn ${active ? "active" : ""}`} onClick={onClick} disabled={!active}>
      <SendIcon />
    </button>
  );
};

export default SendButton;
