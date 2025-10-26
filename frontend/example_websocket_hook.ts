// Using WebSocket Hook
import { useWebSocket } from '@/hooks';

function ChatComponent() {
  const { 
    isConnected, 
    sendMessage, 
    lastMessage 
  } = useWebSocket(sessionId, {
    autoConnect: true,
    onMessage: (msg) => console.log('Received:', msg)
  });
  
  // Use WebSocket functionality
}

// Using File Upload Hook
import { useFileUpload } from '@/hooks';

function UploadComponent() {
  const {
    files,
    uploadProgress,
    selectFiles,
    uploadFiles,
    errors
  } = useFileUpload({
    maxFiles: 5,
    maxSizeMB: 10,
    onSuccess: (file) => console.log('Uploaded:', file)
  });
  
  // Use file upload functionality
}
