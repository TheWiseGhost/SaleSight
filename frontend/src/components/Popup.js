import React, { useEffect, useRef } from "react";
import ReactDOM from "react-dom";
import { motion } from "framer-motion";

const Popup = ({ isOpen, onClose, children }) => {
  const PopupRef = useRef();

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (PopupRef.current && !PopupRef.current.contains(event.target)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    } else {
      document.removeEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return ReactDOM.createPortal(
    <div className="fixed inset-0 bg-gray-900 bg-opacity-10 z-50 flex justify-end">
      <motion.div
        ref={PopupRef}
        initial={{ opacity: 0, x: 150 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 150 }}
        transition={{ duration: 0.75 }}
        drag
        dragConstraints={{
          top: 0,
          bottom: 0,
          left: -window.innerWidth + (window.innerWidth * 2) / 5,
          right: 0,
        }}
        className="bg-white w-2/5 px-4 h-full py-2 rounded-l-lg shadow-lg overflow-y-auto relative"
      >
        <button
          onClick={onClose}
          className="absolute top-4 right-4 font-abo-one text-gray-500 hover:text-black text-3xl font-bold rounded-full py-2 px-3"
        >
          &times;
        </button>
        {children}
      </motion.div>
    </div>,
    document.getElementById("popup-root")
  );
};

export default Popup;
