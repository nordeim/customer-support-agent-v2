/**
 * Custom hook for chat functionality
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import toast from 'react-hot-toast';
import api from '../services/api';
import websocket from '../services/websocket';
import {
  Message,
  ChatState,
  WebSocketMessage,
  EscalationInfo,
} from '../types';

export interface UseChatOptions {
  autoConnect?: boolean;
  onEscalation?: (info: EscalationInfo) => void;
  onError?: (error: string) => void;
}

export function useChat(options: UseChatOptions = {}) {
  const { autoConnect = true, onEscalation, onError } = options;

  // State
  const [state, setState] = useState<ChatState>({
    session: null,
    messages: [],
    isLoading: false,
    isConnected: false,
    isTyping: false,
    error: null,
    escalation: null,
  });

  // Refs
  const typingTimeoutRef = useRef<NodeJS.Timeout>();
  const streamingMessageRef = useRef<string>('');
  const mountedRef = useRef(true);

  // Initialize session
  const initializeSession = useCallback(async (userId?: string) => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const session = await api.createSession(userId);
      
      if (!mountedRef.current) return;

      setState((prev) => ({
        ...prev,
        session: {
          id: session.sessionId,
          userId: session.userId,
          threadId: session.threadId,
          status: session.status as any,
          createdAt: session.createdAt,
          metadata: session.metadata,
        },
        isLoading: false,
      }));

      // Connect WebSocket if auto-connect enabled
      if (autoConnect) {
        websocket.connect(session.sessionId);
      }

      return session;
    } catch (error: any) {
      const message = error.response?.data?.message || 'Failed to create session';
      setState((prev) => ({ ...prev, isLoading: false, error: message }));
      onError?.(message);
      throw error;
    }
  }, [autoConnect, onError]);

  // Send message
  const sendMessage = useCallback(
    async (content: string, attachments?: File[]) => {
      if (!state.session) {
        toast.error('No active session');
        return;
      }

      // Validate message
      const trimmedContent = content.trim();
      if (!trimmedContent && !attachments?.length) {
        return;
      }

      // Add user message immediately
      const userMessage: Message = {
        id: uuidv4(),
        role: 'user',
        content: trimmedContent,
        timestamp: new Date().toISOString(),
        attachments: attachments?.map((file) => ({
          filename: file.name,
          size: file.size,
          type: file.type,
        })),
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
        isLoading: true,
        error: null,
      }));

      try {
        // Send via API or WebSocket
        if (websocket.isConnected()) {
          // Send via WebSocket for streaming
          streamingMessageRef.current = '';
          websocket.sendMessage(trimmedContent, attachments);
        } else {
          // Send via API
          const response = await api.sendMessage(
            state.session.id,
            trimmedContent,
            attachments
          );

          if (!mountedRef.current) return;

          // Add assistant response
          const assistantMessage: Message = {
            id: uuidv4(),
            role: 'assistant',
            content: response.message,
            timestamp: response.timestamp,
            sources: response.sources,
            metadata: {
              processingTime: response.processingTime,
              confidence: response.confidence,
            },
          };

          setState((prev) => ({
            ...prev,
            messages: [...prev.messages, assistantMessage],
            isLoading: false,
          }));

          // Handle escalation
          if (response.requiresEscalation) {
            const escalationInfo: EscalationInfo = {
              required: true,
              ticketId: response.metadata?.ticket_id,
              reason: response.metadata?.reasons,
              priority: response.metadata?.priority,
            };

            setState((prev) => ({ ...prev, escalation: escalationInfo }));
            onEscalation?.(escalationInfo);
          }
        }
      } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to send message';
        setState((prev) => ({ ...prev, isLoading: false, error: message }));
        toast.error(message);
      }
    },
    [state.session, onEscalation]
  );

  // Load message history
  const loadHistory = useCallback(async () => {
    if (!state.session) return;

    try {
      const { messages } = await api.getMessages(state.session.id);
      
      if (!mountedRef.current) return;

      setState((prev) => ({
        ...prev,
        messages: messages.map((msg: any) => ({
          id: msg.id || uuidv4(),
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp || msg.created_at,
          attachments: msg.attachments,
          sources: msg.sources,
          metadata: {
            processingTime: msg.processing_time,
            toolsUsed: msg.tools_used,
          },
        })),
      }));
    } catch (error) {
      console.error('Failed to load message history:', error);
    }
  }, [state.session]);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback(
    (message: WebSocketMessage) => {
      switch (message.type) {
        case 'connected':
          setState((prev) => ({ ...prev, isConnected: true }));
          break;

        case 'text':
          // Streaming text update
          streamingMessageRef.current += message.content || '';
          
          setState((prev) => {
            const messages = [...prev.messages];
            let lastMessage = messages[messages.length - 1];

            if (!lastMessage || lastMessage.role !== 'assistant') {
              // Create new assistant message
              lastMessage = {
                id: uuidv4(),
                role: 'assistant',
                content: streamingMessageRef.current,
                timestamp: message.timestamp,
                isStreaming: true,
              };
              messages.push(lastMessage);
            } else {
              // Update existing message
              lastMessage.content = streamingMessageRef.current;
              lastMessage.isStreaming = true;
            }

            return { ...prev, messages };
          });
          break;

        case 'sources':
          // Update sources for last message
          setState((prev) => {
            const messages = [...prev.messages];
            const lastMessage = messages[messages.length - 1];
            if (lastMessage && lastMessage.role === 'assistant') {
              lastMessage.sources = message.data;
            }
            return { ...prev, messages };
          });
          break;

        case 'complete':
          // Mark streaming as complete
          setState((prev) => {
            const messages = [...prev.messages];
            const lastMessage = messages[messages.length - 1];
            if (lastMessage && lastMessage.role === 'assistant') {
              lastMessage.isStreaming = false;
            }
            return { ...prev, messages, isLoading: false };
          });
          streamingMessageRef.current = '';
          break;

        case 'escalation':
          // Handle escalation notification
          const escalationInfo: EscalationInfo = {
            required: true,
            reason: message.data?.reason,
            ticketId: message.data?.ticket_id,
            priority: message.data?.priority,
          };
          setState((prev) => ({ ...prev, escalation: escalationInfo }));
          onEscalation?.(escalationInfo);
          break;

        case 'error':
          setState((prev) => ({
            ...prev,
            isLoading: false,
            error: message.content || 'An error occurred',
          }));
          toast.error(message.content || 'An error occurred');
          break;

        case 'typing':
          // Handle typing indicator from other users
          setState((prev) => ({ ...prev, isTyping: true }));
          
          // Clear existing timeout
          if (typingTimeoutRef.current) {
            clearTimeout(typingTimeoutRef.current);
          }
          
          // Hide typing after 3 seconds
          typingTimeoutRef.current = setTimeout(() => {
            setState((prev) => ({ ...prev, isTyping: false }));
          }, 3000);
          break;
      }
    },
    [onEscalation]
  );

  // Handle connection changes
  const handleConnectionChange = useCallback((connected: boolean) => {
    setState((prev) => ({ ...prev, isConnected: connected }));
    
    if (connected) {
      toast.success('Connected to chat');
    } else {
      toast.error('Connection lost. Reconnecting...');
    }
  }, []);

  // Clear chat
  const clearChat = useCallback(() => {
    setState((prev) => ({ ...prev, messages: [], error: null }));
  }, []);

  // End session
  const endSession = useCallback(async () => {
    if (!state.session) return;

    try {
      await api.endSession(state.session.id);
      websocket.disconnect();
      
      setState({
        session: null,
        messages: [],
        isLoading: false,
        isConnected: false,
        isTyping: false,
        error: null,
        escalation: null,
      });
      
      toast.success('Session ended');
    } catch (error) {
      toast.error('Failed to end session');
    }
  }, [state.session]);

  // Send typing indicator
  const sendTypingIndicator = useCallback((isTyping: boolean) => {
    if (websocket.isConnected()) {
      websocket.sendTyping(isTyping);
    }
  }, []);

  // Effects
  useEffect(() => {
    mountedRef.current = true;

    // Register WebSocket handlers
    const unsubscribeMessage = websocket.onMessage(handleWebSocketMessage);
    const unsubscribeConnection = websocket.onConnectionChange(handleConnectionChange);

    return () => {
      mountedRef.current = false;
      unsubscribeMessage();
      unsubscribeConnection();
      
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, [handleWebSocketMessage, handleConnectionChange]);

  // Load history when session changes
  useEffect(() => {
    if (state.session) {
      loadHistory();
    }
  }, [state.session, loadHistory]);

  return {
    // State
    session: state.session,
    messages: state.messages,
    isLoading: state.isLoading,
    isConnected: state.isConnected,
    isTyping: state.isTyping,
    error: state.error,
    escalation: state.escalation,

    // Actions
    initializeSession,
    sendMessage,
    clearChat,
    endSession,
    sendTypingIndicator,
    loadHistory,
  };
}

export default useChat;
