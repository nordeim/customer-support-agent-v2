/**
 * Hooks module exports
 * Central export point for all custom React hooks
 */

export { useChat } from './useChat';
export type { UseChatOptions } from './useChat';

export { useWebSocket } from './useWebSocket';
export type { 
  UseWebSocketOptions, 
  UseWebSocketReturn 
} from './useWebSocket';

export { useFileUpload } from './useFileUpload';
export type { 
  FileUploadOptions,
  UploadedFile,
  FileUploadError,
  FileUploadProgress,
  FileWithStatus,
  UseFileUploadReturn 
} from './useFileUpload';

// Re-export for convenience
export { default as useChat } from './useChat';
export { default as useWebSocket } from './useWebSocket';
export { default as useFileUpload } from './useFileUpload';
