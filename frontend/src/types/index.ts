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
  session_id: string;
  user_id?: string;
  thread_id?: string;
  status: string;
  created_at: string;
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
