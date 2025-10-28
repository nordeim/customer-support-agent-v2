/**
 * API service for backend communication
 */
import axios, { AxiosInstance, AxiosError } from 'axios';
import toast from 'react-hot-toast';
import {
  ChatResponse,
  SessionResponse,
  Session,
  Message,
  Source,
  User,
} from '../types';

// Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || '';
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
