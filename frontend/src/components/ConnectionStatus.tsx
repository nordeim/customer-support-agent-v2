/**
 * Connection status indicator
 */
import React from 'react';
import { ExclamationCircleIcon } from '@heroicons/react/24/solid';

interface ConnectionStatusProps {
  isConnected: boolean;
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ isConnected }) => {
  return (
    <div className="flex items-center text-xs">
      {isConnected ? (
        <>
          <div className="h-2 w-2 rounded-full bg-green-500 mr-2 animate-pulse" />
          <span className="text-green-700">Connected</span>
        </>
      ) : (
        <>
          <ExclamationCircleIcon className="h-3 w-3 text-amber-500 mr-2" />
          <span className="text-amber-700">Reconnecting...</span>
        </>
      )}
    </div>
  );
};

export default ConnectionStatus;
