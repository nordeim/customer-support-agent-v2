# Phase 5: Frontend Implementation - Meticulous Plan & Complete Code

## 📋 Implementation Plan

### Overview
We'll create a production-ready React frontend with TypeScript that provides:
1. **Modern Chat Interface** - Clean, responsive UI with Tailwind CSS
2. **Real-time Communication** - WebSocket support for streaming responses
3. **File Upload** - Drag-and-drop file attachment support
4. **Rich Features** - Source citations, escalation alerts, typing indicators
5. **State Management** - React hooks and context for state management
6. **Error Handling** - Comprehensive error boundaries and user feedback

### Architecture & Structure
```
frontend/
├── src/
│   ├── components/         # React components
│   ├── hooks/              # Custom React hooks
│   ├── services/           # API and WebSocket services
│   ├── contexts/           # React contexts
│   ├── types/              # TypeScript definitions
│   ├── utils/              # Utility functions
│   ├── styles/             # CSS and styling
│   ├── assets/             # Images and static files
│   ├── App.tsx             # Main app component
│   ├── main.tsx            # Entry point
│   └── vite-env.d.ts       # Vite type definitions
├── public/                 # Public assets
├── package.json            # Dependencies
├── tsconfig.json           # TypeScript config
├── vite.config.ts          # Vite configuration
├── tailwind.config.js      # Tailwind CSS config
├── postcss.config.js       # PostCSS config
├── .env.example           # Environment variables
└── Dockerfile             # Docker configuration
```

---

## 🛠️ Complete Implementation Files

### File 1: Package Configuration

**`frontend/package.json`**
```json
{
  "name": "customer-support-ai-frontend",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage",
    "lint": "eslint src --ext ts,tsx --report-unused-disable-directives",
    "format": "prettier --write \"src/**/*.{ts,tsx,css}\"",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.5",
    "socket.io-client": "^4.7.4",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0",
    "react-syntax-highlighter": "^15.5.0",
    "react-dropzone": "^14.2.3",
    "react-hot-toast": "^2.4.1",
    "framer-motion": "^11.0.3",
    "clsx": "^2.1.0",
    "date-fns": "^3.3.1",
    "zustand": "^4.5.0",
    "react-intersection-observer": "^9.8.1",
    "@heroicons/react": "^2.1.1",
    "react-error-boundary": "^4.0.12"
  },
  "devDependencies": {
    "@types/react": "^18.2.48",
    "@types/react-dom": "^18.2.18",
    "@types/react-syntax-highlighter": "^15.5.11",
    "@typescript-eslint/eslint-plugin": "^6.19.1",
    "@typescript-eslint/parser": "^6.19.1",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.17",
    "eslint": "^8.56.0",
    "eslint-plugin-react": "^7.33.2",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.5",
    "postcss": "^8.4.33",
    "prettier": "^3.2.4",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.3.3",
    "vite": "^5.0.12",
    "vitest": "^1.2.2",
    "@vitest/ui": "^1.2.2",
    "@testing-library/react": "^14.1.2",
    "@testing-library/jest-dom": "^6.2.0",
    "@testing-library/user-event": "^14.5.2"
  }
}
```

### File 2: TypeScript Types

**`frontend/src/types/index.ts`**
```typescript
/**
 * TypeScript type definitions for the application
 */

// User types
export interface User {
  id: string;
  name?: string;
  email?: string;
  avatar?: string;
}

// Session types
export interface Session {
  id: string;
  userId?: string;
  threadId?: string;
  status: 'active' | 'ended' | 'escalated';
  createdAt: string;
  metadata?: Record<string, any>;
}

// Message types
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  attachments?: Attachment[];
  sources?: Source[];
  isStreaming?: boolean;
  error?: string;
  metadata?: MessageMetadata;
}

export interface MessageMetadata {
  processingTime?: number;
  toolsUsed?: string[];
  confidence?: number;
  escalated?: boolean;
  ticketId?: string;
}

// Attachment types
export interface Attachment {
  id?: string;
  filename: string;
  size: number;
  type: string;
  url?: string;
  preview?: string;
  uploadProgress?: number;
  error?: string;
}

// Source types
export interface Source {
  content: string;
  metadata: {
    source?: string;
    page?: number;
    section?: string;
    [key: string]: any;
  };
  relevanceScore: number;
  rank?: number;
}

// Chat state
export interface ChatState {
  session: Session | null;
  messages: Message[];
  isLoading: boolean;
  isConnected: boolean;
  isTyping: boolean;
  error: string | null;
  escalation: EscalationInfo | null;
}

// Escalation types
export interface EscalationInfo {
  required: boolean;
  ticketId?: string;
  reason?: string[];
  priority?: 'low' | 'normal' | 'high' | 'urgent';
  estimatedWaitTime?: number;
}

// API Response types
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
  requestId?: string;
}

export interface ChatResponse {
  message: string;
  sources: Source[];
  requiresEscalation: boolean;
  confidence: number;
  sessionId: string;
  timestamp: string;
  processingTime?: number;
  metadata?: Record<string, any>;
}

export interface SessionResponse {
  sessionId: string;
  userId?: string;
  threadId?: string;
  status: string;
  createdAt: string;
  metadata?: Record<string, any>;
}

// WebSocket types
export interface WebSocketMessage {
  type: 'connected' | 'message' | 'text' | 'sources' | 'status' | 
        'error' | 'complete' | 'typing' | 'escalation' | 'pong';
  content?: string;
  data?: any;
  timestamp: string;
}

// Settings
export interface AppSettings {
  apiUrl: string;
  wsUrl: string;
  maxFileSize: number;
  supportedFileTypes: string[];
  theme: 'light' | 'dark' | 'auto';
  enableNotifications: boolean;
  enableSounds: boolean;
}

// UI State
export interface UIState {
  sidebarOpen: boolean;
  sourcesOpen: boolean;
  settingsOpen: boolean;
  uploadDrawerOpen: boolean;
  theme: 'light' | 'dark';
}
```

### File 3: API Service

**`frontend/src/services/api.ts`**
```typescript
/**
 * API service for backend communication
 */
import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import toast from 'react-hot-toast';
import {
  ApiResponse,
  ChatResponse,
  SessionResponse,
  Session,
  Message,
  Source,
} from '../types';

// Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // Add request ID
    config.headers['X-Request-ID'] = generateRequestId();

    // Log request in development
    if (import.meta.env.DEV) {
      console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data);
    }

    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    // Log response in development
    if (import.meta.env.DEV) {
      console.log(`[API] Response:`, response.data);
    }

    // Extract request ID for tracking
    const requestId = response.headers['x-request-id'];
    if (requestId) {
      response.data.requestId = requestId;
    }

    return response;
  },
  (error: AxiosError) => {
    handleApiError(error);
    return Promise.reject(error);
  }
);

// Error handler
function handleApiError(error: AxiosError) {
  console.error('[API] Error:', error);

  let message = 'An unexpected error occurred';

  if (error.response) {
    // Server responded with error
    const data: any = error.response.data;
    message = data.message || data.detail || message;

    switch (error.response.status) {
      case 401:
        message = 'Authentication required';
        // Redirect to login
        window.location.href = '/login';
        break;
      case 403:
        message = 'Access denied';
        break;
      case 404:
        message = 'Resource not found';
        break;
      case 429:
        message = 'Too many requests. Please slow down.';
        break;
      case 500:
        message = 'Server error. Please try again later.';
        break;
    }
  } else if (error.request) {
    // Request made but no response
    message = 'Network error. Please check your connection.';
  }

  // Show error toast
  toast.error(message);
}

// Helper to generate request ID
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// API methods
class ApiService {
  // Session management
  async createSession(userId?: string): Promise<SessionResponse> {
    const response = await apiClient.post('/api/sessions', {
      user_id: userId,
      metadata: {
        source: 'web',
        userAgent: navigator.userAgent,
      },
    });
    return response.data;
  }

  async getSession(sessionId: string): Promise<Session> {
    const response = await apiClient.get(`/api/sessions/${sessionId}`);
    return response.data;
  }

  async listSessions(userId?: string): Promise<Session[]> {
    const params = userId ? { user_id: userId } : {};
    const response = await apiClient.get('/api/sessions', { params });
    return response.data;
  }

  async endSession(sessionId: string): Promise<void> {
    await apiClient.patch(`/api/sessions/${sessionId}/status`, {
      status: 'ended',
    });
  }

  // Chat operations
  async sendMessage(
    sessionId: string,
    message: string,
    attachments?: File[]
  ): Promise<ChatResponse> {
    const formData = new FormData();
    formData.append('message', message);

    if (attachments) {
      attachments.forEach((file) => {
        formData.append('attachments', file);
      });
    }

    const response = await apiClient.post(
      `/api/chat/sessions/${sessionId}/messages`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }

  async getMessages(
    sessionId: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<{ messages: Message[]; total: number }> {
    const response = await apiClient.get(
      `/api/chat/sessions/${sessionId}/messages`,
      {
        params: { limit, offset },
      }
    );
    return response.data;
  }

  async searchKnowledgeBase(
    query: string,
    limit: number = 5
  ): Promise<Source[]> {
    const response = await apiClient.post('/api/chat/search', {
      query,
      limit,
    });
    return response.data;
  }

  // File upload
  async uploadFile(
    file: File,
    sessionId: string,
    onProgress?: (progress: number) => void
  ): Promise<{
    filename: string;
    size: number;
    processed: boolean;
    preview?: string;
  }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    const response = await apiClient.post('/api/chat/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress?.(progress);
        }
      },
    });

    return response.data;
  }

  // Health check
  async checkHealth(): Promise<{
    status: string;
    services: Record<string, string>;
  }> {
    const response = await apiClient.get('/health/ready');
    return response.data;
  }

  // Authentication
  async login(username: string, password: string): Promise<{ token: string }> {
    const response = await apiClient.post('/auth/login', {
      username,
      password,
    });

    const { token } = response.data;
    localStorage.setItem('auth_token', token);

    return response.data;
  }

  async logout(): Promise<void> {
    localStorage.removeItem('auth_token');
    await apiClient.post('/auth/logout');
  }

  async getCurrentUser(): Promise<User | null> {
    try {
      const response = await apiClient.get('/auth/me');
      return response.data;
    } catch {
      return null;
    }
  }
}

// Export singleton instance
export const api = new ApiService();
export default api;
```

### File 4: WebSocket Service

**`frontend/src/services/websocket.ts`**
```typescript
/**
 * WebSocket service for real-time communication
 */
import { io, Socket } from 'socket.io-client';
import { WebSocketMessage } from '../types';

// Configuration
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

export type MessageHandler = (message: WebSocketMessage) => void;
export type ConnectionHandler = (connected: boolean) => void;

class WebSocketService {
  private socket: Socket | null = null;
  private sessionId: string | null = null;
  private messageHandlers: Set<MessageHandler> = new Set();
  private connectionHandlers: Set<ConnectionHandler> = new Set();
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private pingInterval: NodeJS.Timeout | null = null;

  /**
   * Connect to WebSocket server
   */
  connect(sessionId: string): void {
    if (this.socket?.connected && this.sessionId === sessionId) {
      console.log('[WS] Already connected to session:', sessionId);
      return;
    }

    this.disconnect();
    this.sessionId = sessionId;
    this.reconnectAttempts = 0;

    console.log('[WS] Connecting to session:', sessionId);

    // Create WebSocket connection
    const wsUrl = `${WS_URL}/ws?session_id=${sessionId}`;
    
    // Using native WebSocket for better compatibility
    this.setupNativeWebSocket(wsUrl);
  }

  /**
   * Setup native WebSocket connection
   */
  private setupNativeWebSocket(url: string): void {
    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('[WS] Connected');
        this.reconnectAttempts = 0;
        this.notifyConnectionHandlers(true);
        this.startPingInterval(ws);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('[WS] Failed to parse message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('[WS] Error:', error);
      };

      ws.onclose = (event) => {
        console.log('[WS] Disconnected:', event.reason || 'Connection closed');
        this.notifyConnectionHandlers(false);
        this.stopPingInterval();

        // Attempt to reconnect
        if (this.sessionId && this.reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
          this.scheduleReconnect();
        }
      };

      // Store WebSocket reference
      this.socket = ws as any; // Type casting for compatibility
    } catch (error) {
      console.error('[WS] Failed to create WebSocket:', error);
      this.scheduleReconnect();
    }
  }

  /**
   * Send message through WebSocket
   */
  send(type: string, data: any): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.error('[WS] Cannot send - not connected');
      return;
    }

    const message = {
      type,
      ...data,
      timestamp: new Date().toISOString(),
    };

    try {
      (this.socket as any).send(JSON.stringify(message));
      console.log('[WS] Sent:', type, data);
    } catch (error) {
      console.error('[WS] Failed to send message:', error);
    }
  }

  /**
   * Send chat message
   */
  sendMessage(content: string, attachments?: any[]): void {
    this.send('message', {
      content,
      attachments,
    });
  }

  /**
   * Send typing indicator
   */
  sendTyping(isTyping: boolean): void {
    this.send('typing', { isTyping });
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    this.stopPingInterval();

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.socket) {
      if ((this.socket as any).readyState === WebSocket.OPEN) {
        (this.socket as any).close();
      }
      this.socket = null;
    }

    this.sessionId = null;
    this.notifyConnectionHandlers(false);
    console.log('[WS] Disconnected');
  }

  /**
   * Register message handler
   */
  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  /**
   * Register connection status handler
   */
  onConnectionChange(handler: ConnectionHandler): () => void {
    this.connectionHandlers.add(handler);
    return () => this.connectionHandlers.delete(handler);
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket !== null && 
           (this.socket as any).readyState === WebSocket.OPEN;
  }

  // Private methods

  private handleMessage(message: WebSocketMessage): void {
    console.log('[WS] Received:', message.type, message);

    // Handle special message types
    switch (message.type) {
      case 'connected':
        console.log('[WS] Connection confirmed');
        break;
      case 'pong':
        // Pong received, connection is alive
        break;
      case 'error':
        console.error('[WS] Server error:', message.content);
        break;
    }

    // Notify all handlers
    this.messageHandlers.forEach((handler) => {
      try {
        handler(message);
      } catch (error) {
        console.error('[WS] Handler error:', error);
      }
    });
  }

  private notifyConnectionHandlers(connected: boolean): void {
    this.connectionHandlers.forEach((handler) => {
      try {
        handler(connected);
      } catch (error) {
        console.error('[WS] Connection handler error:', error);
      }
    });
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      RECONNECT_DELAY * Math.pow(2, this.reconnectAttempts - 1),
      30000
    );

    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (this.sessionId) {
        this.connect(this.sessionId);
      }
    }, delay);
  }

  private startPingInterval(ws: WebSocket): void {
    this.stopPingInterval();

    // Send ping every 30 seconds
    this.pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
  }

  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }
}

// Export singleton instance
export const websocket = new WebSocketService();
export default websocket;
```

### File 5: Custom Hooks

**`frontend/src/hooks/useChat.ts`**
```typescript
/**
 * Custom hook for chat functionality
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import toast from 'react-hot-toast';
import api from '../services/api';
import websocket from '../services/websocket';
import {
  Session,
  Message,
  ChatState,
  WebSocketMessage,
  Attachment,
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
```

### File 6: Main Chat Interface Component

**`frontend/src/components/ChatInterface.tsx`**
```typescript
/**
 * Main chat interface component
 */
import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PaperAirplaneIcon,
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
  }, []);

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
              isLoading={isLoading}
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
```

### File 7: Message List Component

**`frontend/src/components/MessageList.tsx`**
```typescript
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
```

### File 8: Input Area Component

**`frontend/src/components/InputArea.tsx`**
```typescript
/**
 * Input area component for message composition
 */
import React, { forwardRef, KeyboardEvent } from 'react';
import {
  PaperAirplaneIcon,
  PaperClipIcon,
  FaceSmileIcon,
  MicrophoneIcon,
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
```

### File 9: Additional Components

**`frontend/src/components/SourcesPanel.tsx`**
```typescript
/**
 * Sources panel for displaying referenced sources
 */
import React from 'react';
import { XMarkIcon, DocumentTextIcon, StarIcon } from '@heroicons/react/24/outline';
import { Source } from '../types';

interface SourcesPanelProps {
  sources: Source[];
  onClose: () => void;
}

const SourcesPanel: React.FC<SourcesPanelProps> = ({ sources, onClose }) => {
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-6 py-4">
        <h2 className="text-lg font-semibold text-gray-900">Sources</h2>
        <button
          onClick={onClose}
          className="rounded-lg p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
        >
          <XMarkIcon className="h-5 w-5" />
        </button>
      </div>

      {/* Sources list */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-4">
          {sources.map((source, index) => (
            <SourceCard key={index} source={source} index={index} />
          ))}
        </div>
      </div>
    </div>
  );
};

const SourceCard: React.FC<{ source: Source; index: number }> = ({
  source,
  index,
}) => {
  const relevancePercent = Math.round(source.relevanceScore * 100);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 hover
