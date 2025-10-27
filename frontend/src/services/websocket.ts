/**
 * WebSocket service for real-time communication
 */
import { WebSocketMessage } from '../types';

// Configuration
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

export type MessageHandler = (message: WebSocketMessage) => void;
export type ConnectionHandler = (connected: boolean) => void;

class WebSocketService {
  private socket: WebSocket | null = null;
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
    if (this.socket?.readyState === WebSocket.OPEN && this.sessionId === sessionId) {
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
      this.socket.send(JSON.stringify(message));
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
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.close();
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
           this.socket.readyState === WebSocket.OPEN;
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
