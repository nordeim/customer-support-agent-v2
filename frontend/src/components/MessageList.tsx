/**
 * Message list component for displaying chat messages
 */
import React, { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  UserCircleIcon,
  SparklesIcon,
  PaperClipIcon,
  DocumentTextIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import { Message } from '../types';
import TypingIndicator from './TypingIndicator';

interface MessageListProps {
  messages: Message[];
  isLoading?: boolean;
  isTyping?: boolean;
}

const MessageList: React.FC<MessageListProps> = ({
  messages,
  isLoading,
  isTyping,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  return (
    <div className="flex h-full flex-col overflow-y-auto px-6 py-4">
      <AnimatePresence initial={false}>
        {messages.map((message) => (
          <MessageItem key={message.id} message={message} />
        ))}

        {isTyping && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex items-start mb-6"
          >
            <div className="flex-shrink-0 mr-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-blue-600">
                <SparklesIcon className="h-5 w-5 text-white" />
              </div>
            </div>
            <TypingIndicator />
          </motion.div>
        )}
      </AnimatePresence>

      <div ref={messagesEndRef} />
    </div>
  );
};

const MessageItem: React.FC<{ message: Message }> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3 }}
      className={`mb-6 flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`flex max-w-3xl ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 ${isUser ? 'ml-3' : 'mr-3'}`}>
          {isUser ? (
            <UserCircleIcon className="h-8 w-8 text-gray-400" />
          ) : (
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-blue-600">
              <SparklesIcon className="h-5 w-5 text-white" />
            </div>
          )}
        </div>

        {/* Message Content */}
        <div className="flex flex-col">
          <div
            className={`rounded-lg px-4 py-3 ${
              isUser
                ? 'bg-blue-600 text-white'
                : 'bg-white border border-gray-200'
            }`}
          >
            {/* Message text */}
            {message.isStreaming ? (
              <div className="text-sm">{message.content}</div>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                className={`prose max-w-none ${
                  isUser ? 'prose-invert' : ''
                }`}
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={tomorrow}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}

            {/* Attachments */}
            {message.attachments && message.attachments.length > 0 && (
              <div className="mt-3 space-y-2">
                {message.attachments.map((attachment, index) => (
                  <div
                    key={index}
                    className={`flex items-center rounded px-2 py-1 ${
                      isUser
                        ? 'bg-blue-700/50'
                        : 'bg-gray-100'
                    }`}
                  >
                    <PaperClipIcon className="h-4 w-4 mr-2" />
                    <span className="text-sm">{attachment.filename}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Sources */}
            {message.sources && message.sources.length > 0 && (
              <div className="mt-3 space-y-1">
                <div className="flex items-center text-xs text-gray-500 mb-1">
                  <DocumentTextIcon className="h-3 w-3 mr-1" />
                  Sources ({message.sources.length})
                </div>
              </div>
            )}

            {/* Error */}
            {message.error && (
              <div className="mt-2 text-sm text-red-600">
                ⚠️ {message.error}
              </div>
            )}
          </div>

          {/* Metadata */}
          <div className="mt-1 flex items-center space-x-3 text-xs text-gray-400">
            <span>{format(new Date(message.timestamp), 'HH:mm')}</span>
            
            {message.metadata?.processingTime && (
              <span className="flex items-center">
                <ClockIcon className="h-3 w-3 mr-0.5" />
                {message.metadata.processingTime.toFixed(2)}s
              </span>
            )}
            
            {message.metadata?.confidence && (
              <span>
                {(message.metadata.confidence * 100).toFixed(0)}% confident
              </span>
            )}

            {message.isStreaming && (
              <span className="text-blue-500 animate-pulse">
                Streaming...
              </span>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default MessageList;
