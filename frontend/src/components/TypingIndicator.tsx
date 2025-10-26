/**
 * Typing indicator animation
 */
import React from 'react';
import { motion } from 'framer-motion';

const TypingIndicator: React.FC = () => {
  return (
    <div className="flex items-center space-x-2 rounded-lg bg-gray-100 px-4 py-3">
      <motion.div
        className="h-2 w-2 rounded-full bg-gray-400"
        animate={{ y: [0, -5, 0] }}
        transition={{ duration: 0.5, repeat: Infinity, delay: 0 }}
      />
      <motion.div
        className="h-2 w-2 rounded-full bg-gray-400"
        animate={{ y: [0, -5, 0] }}
        transition={{ duration: 0.5, repeat: Infinity, delay: 0.1 }}
      />
      <motion.div
        className="h-2 w-2 rounded-full bg-gray-400"
        animate={{ y: [0, -5, 0] }}
        transition={{ duration: 0.5, repeat: Infinity, delay: 0.2 }}
      />
    </div>
  );
};

export default TypingIndicator;
