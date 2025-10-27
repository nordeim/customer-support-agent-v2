/**
 * Input area component for message composition
 */
import { forwardRef, KeyboardEvent } from 'react';
import {
  PaperAirplaneIcon,
  PaperClipIcon,
} from '@heroicons/react/24/outline';

interface InputAreaProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onAttach?: () => void;
  disabled?: boolean;
  placeholder?: string;
}

const InputArea = forwardRef<HTMLTextAreaElement, InputAreaProps>(
  ({ value, onChange, onSend, onAttach, disabled, placeholder }, ref) => {
    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!disabled && value.trim()) {
          onSend();
        }
      }
    };

    const adjustTextareaHeight = (element: HTMLTextAreaElement) => {
      element.style.height = 'auto';
      element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
    };

    return (
      <div className="border-t bg-white px-6 py-4">
        <div className="flex items-end space-x-3">
          {/* Attachment button */}
          <button
            onClick={onAttach}
            disabled={disabled}
            className="flex-shrink-0 rounded-full p-2 text-gray-500 hover:bg-gray-100 hover:text-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Attach file"
          >
            <PaperClipIcon className="h-5 w-5" />
          </button>

          {/* Input field */}
          <div className="flex-1 relative">
            <textarea
              ref={ref}
              value={value}
              onChange={(e) => {
                onChange(e.target.value);
                adjustTextareaHeight(e.target);
              }}
              onKeyDown={handleKeyDown}
              disabled={disabled}
              placeholder={placeholder || 'Type your message...'}
              rows={1}
              className="w-full resize-none rounded-lg border border-gray-300 px-4 py-2 pr-12 text-sm placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:bg-gray-50 disabled:text-gray-500"
              style={{ minHeight: '40px' }}
            />
            
            {/* Character count */}
            {value.length > 0 && (
              <div className="absolute bottom-2 right-12 text-xs text-gray-400">
                {value.length}/4000
              </div>
            )}
          </div>

          {/* Send button */}
          <button
            onClick={onSend}
            disabled={disabled || !value.trim()}
            className="flex-shrink-0 rounded-full bg-blue-600 p-2 text-white hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            title="Send message (Enter)"
          >
            <PaperAirplaneIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Hints */}
        <div className="mt-2 flex items-center justify-between text-xs text-gray-400">
          <div className="flex items-center space-x-3">
            <span>Press Enter to send, Shift+Enter for new line</span>
            <span>•</span>
            <span>Cmd+K to clear chat</span>
          </div>
          <div className="flex items-center space-x-2">
            <span>Powered by AI</span>
          </div>
        </div>
      </div>
    );
  }
);

InputArea.displayName = 'InputArea';

export default InputArea;
