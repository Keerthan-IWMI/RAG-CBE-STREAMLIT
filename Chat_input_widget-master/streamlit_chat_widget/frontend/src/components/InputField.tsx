import React from "react";

interface InputFieldProps {
  value: string;
  onChange: (v: string) => void;
  onKeyPress: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  placeholder?: string;
}

const InputField: React.FC<InputFieldProps> = ({ value, onChange, onKeyPress, placeholder }) => {
  return (
    <input
      type="text"
      className="chat-input-field"
      placeholder={placeholder}
      value={value}
      onKeyDown={onKeyPress}
      onChange={(e) => onChange((e.target as HTMLInputElement).value)}
    />
  );
};

export default InputField;
