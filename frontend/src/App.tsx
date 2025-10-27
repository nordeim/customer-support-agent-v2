/**
 * Main application component
 */
import { Toaster } from 'react-hot-toast';
import { ErrorBoundary } from 'react-error-boundary';
import ChatInterface from './components/ChatInterface';
import ErrorFallback from './components/ErrorFallback';

function App() {
  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onError={(error, errorInfo) => {
        console.error('Application error:', error, errorInfo);
      }}
      onReset={() => window.location.reload()}
    >
      <div className="h-screen overflow-hidden">
        <ChatInterface />
      </div>
      
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
            borderRadius: '8px',
          },
          success: {
            style: {
              background: '#10B981',
            },
            iconTheme: {
              primary: 'white',
              secondary: '#10B981',
            },
          },
          error: {
            style: {
              background: '#EF4444',
            },
            iconTheme: {
              primary: 'white',
              secondary: '#EF4444',
            },
          },
        }}
      />
    </ErrorBoundary>
  );
}

export default App;
