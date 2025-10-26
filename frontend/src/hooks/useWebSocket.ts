/**
 * Custom hook for WebSocket connection management
 * Provides a React-friendly interface for real-time communication
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { WebSocketMessage } from '../types';

export interface UseWebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
  heartbeat?: boolean;
  heartbeatInterval?: number;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
  onReconnecting?: (attempt: number) => void;
}

export interface UseWebSocketReturn {
  // State
  isConnected: boolean;
  isConnecting: boolean;
  isReconnecting: boolean;
  lastMessage: WebSocketMessage | null;
  error: Error | null;
  reconnectAttempt: number;
  
  // Actions
  connect: (sessionId?: string) => void;
  disconnect: () => void;
  send: (type: string, data: any) => void;
  sendMessage: (content: string, attachments?: any[]) => void;
  sendTyping: (isTyping: boolean) => void;
  sendPing: () => void;
  
  // Utilities
  getReadyState: () => number;
  clearError: () => void;
}

const DEFAULT_OPTIONS: Partial<UseWebSocketOptions> = {
  autoConnect: false,
  reconnect: true,
  reconnectInterval: 3000,
  reconnectAttempts: 5,
  heartbeat: true,
  heartbeatInterval: 30000,
};

export function useWebSocket(
  sessionId?: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  // Merge options with defaults
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // State
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  
  // Refs
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<NodeJS.Timeout>();
  const heartbeatTimer = useRef<NodeJS.Timeout>();
  const sessionIdRef = useRef(sessionId);
  const mountedRef = useRef(true);
  
  // Update session ID ref
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);
  
  /**
   * Start heartbeat mechanism
   */
  const startHeartbeat = useCallback(() => {
    if (!opts.heartbeat) return;
    
    const sendPing = () => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        send('ping', { timestamp: Date.now() });
      }
    };
    
    // Clear existing timer
    if (heartbeatTimer.current) {
      clearInterval(heartbeatTimer.current);
    }
    
    // Start new heartbeat
    heartbeatTimer.current = setInterval(sendPing, opts.heartbeatInterval);
  }, [opts.heartbeat, opts.heartbeatInterval]);
  
  /**
   * Stop heartbeat
   */
  const stopHeartbeat = useCallback(() => {
    if (heartbeatTimer.current) {
      clearInterval(heartbeatTimer.current);
      heartbeatTimer.current = undefined;
    }
  }, []);
  
  /**
   * Handle WebSocket open event
   */
  const handleOpen = useCallback((event: Event) => {
    console.log('[useWebSocket] Connected');
    
    if (!mountedRef.current) return;
    
    setIsConnected(true);
    setIsConnecting(false);
    setIsReconnecting(false);
    setReconnectAttempt(0);
    setError(null);
    
    // Start heartbeat
    startHeartbeat();
    
    // Call user callback
    opts.onOpen?.(event);
  }, [opts, startHeartbeat]);
  
  /**
   * Handle WebSocket message event
   */
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      if (!mountedRef.current) return;
      
      // Update last message
      setLastMessage(message);
      
      // Handle pong messages internally
      if (message.type === 'pong') {
        console.log('[useWebSocket] Pong received');
        return;
      }
      
      // Call user callback
      opts.onMessage?.(message);
      
    } catch (err) {
      console.error('[useWebSocket] Failed to parse message:', err);
      setError(new Error('Failed to parse WebSocket message'));
    }
  }, [opts]);
  
  /**
   * Handle WebSocket error event
   */
  const handleError = useCallback((event: Event) => {
    console.error('[useWebSocket] Error:', event);
    
    if (!mountedRef.current) return;
    
    const err = new Error('WebSocket error occurred');
    setError(err);
    
    // Call user callback
    opts.onError?.(event);
  }, [opts]);
  
  /**
   * Handle WebSocket close event
   */
  const handleClose = useCallback((event: CloseEvent) => {
    console.log('[useWebSocket] Disconnected:', event.reason || 'Connection closed');
    
    if (!mountedRef.current) return;
    
    setIsConnected(false);
    setIsConnecting(false);
    stopHeartbeat();
    
    // Call user callback
    opts.onClose?.(event);
    
    // Attempt reconnection if enabled
    if (
      opts.reconnect &&
      !event.wasClean &&
      reconnectAttempt < (opts.reconnectAttempts || 5)
    ) {
      setIsReconnecting(true);
      
      const attempt = reconnectAttempt + 1;
      setReconnectAttempt(attempt);
      
      // Notify about reconnection
      opts.onReconnecting?.(attempt);
      
      // Calculate delay with exponential backoff
      const delay = Math.min(
        (opts.reconnectInterval || 3000) * Math.pow(1.5, attempt - 1),
        30000
      );
      
      console.log(`[useWebSocket] Reconnecting in ${delay}ms (attempt ${attempt})`);
      
      reconnectTimer.current = setTimeout(() => {
        if (mountedRef.current && sessionIdRef.current) {
          connect(sessionIdRef.current);
        }
      }, delay);
    }
  }, [opts, reconnectAttempt, stopHeartbeat]);
  
  /**
   * Connect to WebSocket server
   */
  const connect = useCallback((newSessionId?: string) => {
    // Use provided session ID or fall back to current
    const sid = newSessionId || sessionIdRef.current;
    
    if (!sid) {
      console.error('[useWebSocket] No session ID provided');
      setError(new Error('Session ID is required'));
      return;
    }
    
    // Update session ID ref
    sessionIdRef.current = sid;
    
    // Don't connect if already connected or connecting
    if (ws.current?.readyState === WebSocket.OPEN) {
      console.log('[useWebSocket] Already connected');
      return;
    }
    
    if (ws.current?.readyState === WebSocket.CONNECTING) {
      console.log('[useWebSocket] Already connecting');
      return;
    }
    
    // Close existing connection
    if (ws.current) {
      ws.current.close();
    }
    
    setIsConnecting(true);
    setError(null);
    
    try {
      // Construct WebSocket URL
      const baseUrl = opts.url || import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
      const wsUrl = `${baseUrl}/ws?session_id=${sid}`;
      
      console.log('[useWebSocket] Connecting to:', wsUrl);
      
      // Create WebSocket connection
      const websocket = new WebSocket(wsUrl);
      
      // Attach event handlers
      websocket.onopen = handleOpen;
      websocket.onmessage = handleMessage;
      websocket.onerror = handleError;
      websocket.onclose = handleClose;
      
      // Store reference
      ws.current = websocket;
      
    } catch (err) {
      console.error('[useWebSocket] Failed to create WebSocket:', err);
      setError(err as Error);
      setIsConnecting(false);
    }
  }, [opts.url, handleOpen, handleMessage, handleError, handleClose]);
  
  /**
   * Disconnect from WebSocket server
   */
  const disconnect = useCallback(() => {
    console.log('[useWebSocket] Disconnecting');
    
    // Clear reconnect timer
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = undefined;
    }
    
    // Stop heartbeat
    stopHeartbeat();
    
    // Close WebSocket
    if (ws.current) {
      ws.current.close(1000, 'User disconnected');
      ws.current = null;
    }
    
    // Reset state
    setIsConnected(false);
    setIsConnecting(false);
    setIsReconnecting(false);
    setReconnectAttempt(0);
    setLastMessage(null);
  }, [stopHeartbeat]);
  
  /**
   * Send data through WebSocket
   */
  const send = useCallback((type: string, data: any = {}) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      console.error('[useWebSocket] Cannot send - not connected');
      setError(new Error('WebSocket is not connected'));
      return;
    }
    
    try {
      const message = {
        type,
        ...data,
        timestamp: new Date().toISOString(),
      };
      
      ws.current.send(JSON.stringify(message));
      console.log('[useWebSocket] Sent:', type, data);
      
    } catch (err) {
      console.error('[useWebSocket] Failed to send message:', err);
      setError(err as Error);
    }
  }, []);
  
  /**
   * Send chat message
   */
  const sendMessage = useCallback((content: string, attachments?: any[]) => {
    send('message', { content, attachments });
  }, [send]);
  
  /**
   * Send typing indicator
   */
  const sendTyping = useCallback((isTyping: boolean) => {
    send('typing', { isTyping });
  }, [send]);
  
  /**
   * Send ping message
   */
  const sendPing = useCallback(() => {
    send('ping', {});
  }, [send]);
  
  /**
   * Get WebSocket ready state
   */
  const getReadyState = useCallback((): number => {
    return ws.current?.readyState ?? WebSocket.CLOSED;
  }, []);
  
  /**
   * Clear error state
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);
  
  // Auto-connect on mount if enabled
  useEffect(() => {
    if (opts.autoConnect && sessionId) {
      connect(sessionId);
    }
    
    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, []); // Only run on mount/unmount
  
  // Reconnect if session ID changes
  useEffect(() => {
    if (isConnected && sessionId !== sessionIdRef.current) {
      console.log('[useWebSocket] Session ID changed, reconnecting');
      disconnect();
      if (sessionId) {
        connect(sessionId);
      }
    }
  }, [sessionId, isConnected, connect, disconnect]);
  
  return {
    // State
    isConnected,
    isConnecting,
    isReconnecting,
    lastMessage,
    error,
    reconnectAttempt,
    
    // Actions
    connect,
    disconnect,
    send,
    sendMessage,
    sendTyping,
    sendPing,
    
    // Utilities
    getReadyState,
    clearError,
  };
}

export default useWebSocket;
