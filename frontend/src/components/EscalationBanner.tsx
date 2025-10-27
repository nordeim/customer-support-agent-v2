/**
 * Escalation notification banner
 */
import React from 'react';
import { motion } from 'framer-motion';
import {
  UserGroupIcon,
  ClockIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { EscalationInfo } from '../types';

interface EscalationBannerProps {
  escalation: EscalationInfo;
  onClose?: () => void;
}

const EscalationBanner: React.FC<EscalationBannerProps> = ({
  escalation,
  onClose,
}) => {
  const getPriorityColor = () => {
    switch (escalation.priority) {
      case 'urgent':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'high':
        return 'bg-orange-50 border-orange-200 text-orange-800';
      default:
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={`border-b px-6 py-4 ${getPriorityColor()}`}
    >
      <div className="flex items-start">
        <UserGroupIcon className="h-5 w-5 mr-3 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <h3 className="text-sm font-medium">
            Connecting you with a human agent
          </h3>
          <p className="mt-1 text-sm opacity-90">
            Your issue has been escalated to our support team.
            {escalation.ticketId && (
              <span className="ml-1 font-mono">
                Ticket #{escalation.ticketId}
              </span>
            )}
          </p>
          
          {escalation.estimatedWaitTime && (
            <div className="mt-2 flex items-center text-xs">
              <ClockIcon className="h-4 w-4 mr-1" />
              Estimated wait: {escalation.estimatedWaitTime} minutes
            </div>
          )}
          
          {escalation.reason && escalation.reason.length > 0 && (
            <div className="mt-2 text-xs">
              Reason: {escalation.reason.join(', ')}
            </div>
          )}
        </div>
        
        {onClose && (
          <button
            onClick={onClose}
            className="ml-3 rounded p-1 hover:bg-white/50"
          >
            <XMarkIcon className="h-4 w-4" />
          </button>
        )}
      </div>
    </motion.div>
  );
};

export default EscalationBanner;
