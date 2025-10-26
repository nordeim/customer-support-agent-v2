/**
 * Error fallback component for error boundaries
 */
import React from 'react';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  resetErrorBoundary,
}) => {
  return (
    <div className="flex h-screen items-center justify-center bg-gray-50">
      <div className="max-w-md text-center">
        <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-red-100">
          <ExclamationTriangleIcon className="h-8 w-8 text-red-600" />
        </div>
        
        <h1 className="mb-2 text-2xl font-semibold text-gray-900">
          Something went wrong
        </h1>
        <p className="mb-6 text-gray-600">
          We apologize for the inconvenience. The application encountered an unexpected error.
        </p>
        
        {import.meta.env.DEV && (
          <details className="mb-6 text-left">
            <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
              Error details
            </summary>
            <pre className="mt-2 overflow-auto rounded bg-gray-100 p-3 text-xs text-gray-800">
              {error.message}
              {error.stack}
            </pre>
          </details>
        )}
        
        <button
          onClick={resetErrorBoundary}
          className="rounded-lg bg-blue-600 px-6 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          Try again
        </button>
      </div>
    </div>
  );
};

export default ErrorFallback;
