/**
 * Main chat interface component
 */
import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PaperClipIcon,
  XMarkIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import { useChat } from '../hooks/useChat';
import MessageList from './MessageList';
import InputArea from './InputArea';
import SourcesPanel from './SourcesPanel';
import FileUploadModal from './FileUploadModal';
import EscalationBanner from './EscalationBanner';
import ConnectionStatus from './ConnectionStatus';
import WelcomeScreen from './WelcomeScreen';
import toast from 'react-hot-toast';
import { Source } from '../types';

export const ChatInterface: React.FC = () => {
  // State
  const [inputValue, setInputValue] = useState('');
  const [attachments, setAttachments] = useState<File[]>([]);
  const [showSources, setShowSources] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [currentSources, setCurrentSources] = useState<Source[]>([]);
  const [isTyping, setIsTyping] = useState(false);

  // Refs
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout>();

  // Use chat hook
  const {
    session,
    messages,
    isLoading,
    isConnected,
    isTyping: otherTyping,
    error,
    escalation,
    initializeSession,
    sendMessage,
    sendTypingIndicator,
    clearChat,
    endSession,
  } = useChat({
    autoConnect: true,
    onEscalation: (info) => {
      toast.custom(
        (t) => (
          <EscalationBanner
            escalation={info}
            onClose={() => toast.dismiss(t.id)}
          />
        ),
        { duration: Infinity }
      );
    },
  });

  // Initialize session on mount
  useEffect(() => {
    if (!session) {
      initializeSession();
    }
  }, [session, initializeSession]);

  // Update sources when last message changes
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (lastMessage?.sources) {
      setCurrentSources(lastMessage.sources);
      if (lastMessage.sources.length > 0) {
        setShowSources(true);
      }
    }
  }, [messages]);

  // Handle input change with typing indicator
  const handleInputChange = (value: string) => {
    setInputValue(value);

    // Send typing indicator
    if (!isTyping && value.length > 0) {
      setIsTyping(true);
      sendTypingIndicator(true);
    }

    // Clear typing indicator after delay
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }

    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(false);
      sendTypingIndicator(false);
    }, 1000);
  };

  // Handle send message
  const handleSend = async () => {
    if (!inputValue.trim() && attachments.length === 0) {
      return;
    }

    // Clear typing indicator
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }
    setIsTyping(false);
    sendTypingIndicator(false);

    // Send message
    const messageContent = inputValue.trim();
    setInputValue('');
    
    try {
      await sendMessage(messageContent, attachments);
      setAttachments([]);
    } catch (error) {
      // Error handled by hook
    }
  };

  // Handle file selection
  const handleFileSelect = (files: File[]) => {
    // Validate file size
    const maxSize = 10 * 1024 * 1024; // 10MB
    const validFiles = files.filter((file) => {
      if (file.size > maxSize) {
        toast.error(`${file.name} exceeds maximum size of 10MB`);
        return false;
      }
      return true;
    });

    setAttachments((prev) => [...prev, ...validFiles]);
    setShowUploadModal(false);

    if (validFiles.length > 0) {
      toast.success(`${validFiles.length} file(s) attached`);
    }
  };

  // Remove attachment
  const removeAttachment = (index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  };

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + K to clear chat
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        clearChat();
      }
      
      // Cmd/Ctrl + / to focus input
      if ((e.metaKey || e.ctrlKey) && e.key === '/') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [clearChat]);

  // Show welcome screen if no messages
  if (!session) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-center">
          <ArrowPathIcon className="mx-auto h-12 w-12 animate-spin text-blue-600" />
          <p className="mt-4 text-gray-600">Initializing chat...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Main Chat Area */}
      <div className="flex flex-1 flex-col">
        {/* Header */}
        <header className="border-b bg-white px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-900">
                Customer Support Assistant
              </h1>
              <ConnectionStatus isConnected={isConnected} />
            </div>
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setShowSources(!showSources)}
                className={`rounded-lg px-3 py-2 text-sm font-medium transition ${
                  showSources
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <InformationCircleIcon className="inline h-5 w-5 mr-1" />
                Sources {currentSources.length > 0 && `(${currentSources.length})`}
              </button>
              <button
                onClick={clearChat}
                className="rounded-lg px-3 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100"
              >
                Clear
              </button>
              <button
                onClick={endSession}
                className="rounded-lg px-3 py-2 text-sm font-medium text-red-600 hover:bg-red-50"
              >
                End Session
              </button>
            </div>
          </div>
        </header>

        {/* Escalation Banner */}
        <AnimatePresence>
          {escalation?.required && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <EscalationBanner escalation={escalation} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Messages Area */}
        <div className="flex-1 overflow-hidden">
          {messages.length === 0 ? (
            <WelcomeScreen
              onSampleQuestion={(question) => {
                setInputValue(question);
                inputRef.current?.focus();
              }}
            />
          ) : (
            <MessageList
              messages={messages}
              isTyping={otherTyping}
            />
          )}
        </div>

        {/* Error Banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="border-t border-red-200 bg-red-50 px-6 py-3"
            >
              <div className="flex items-center">
                <ExclamationTriangleIcon className="h-5 w-5 text-red-500 mr-2" />
                <span className="text-sm text-red-800">{error}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Attachments Bar */}
        <AnimatePresence>
          {attachments.length > 0 && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="border-t bg-gray-50 px-6 py-3"
            >
              <div className="flex flex-wrap gap-2">
                {attachments.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center rounded-lg bg-white px-3 py-2 shadow-sm"
                  >
                    <PaperClipIcon className="h-4 w-4 text-gray-400 mr-2" />
                    <span className="text-sm text-gray-700">{file.name}</span>
                    <button
                      onClick={() => removeAttachment(index)}
                      className="ml-2 text-gray-400 hover:text-gray-600"
                    >
                      <XMarkIcon className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input Area */}
        <InputArea
          ref={inputRef}
          value={inputValue}
          onChange={handleInputChange}
          onSend={handleSend}
          onAttach={() => setShowUploadModal(true)}
          disabled={isLoading}
          placeholder={
            isLoading
              ? 'Waiting for response...'
              : 'Type your message here...'
          }
        />
      </div>

      {/* Sources Panel */}
      <AnimatePresence>
        {showSources && currentSources.length > 0 && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 384, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="border-l bg-white"
          >
            <SourcesPanel
              sources={currentSources}
              onClose={() => setShowSources(false)}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* File Upload Modal */}
      <AnimatePresence>
        {showUploadModal && (
          <FileUploadModal
            onSelect={handleFileSelect}
            onClose={() => setShowUploadModal(false)}
            maxFiles={5}
            maxSizeMB={10}
            acceptedFormats={[
              '.pdf',
              '.doc',
              '.docx',
              '.txt',
              '.csv',
              '.xlsx',
              '.png',
              '.jpg',
              '.jpeg',
            ]}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default ChatInterface;
