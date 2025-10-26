/**
 * Welcome screen shown when no messages
 */
import React from 'react';
import {
  ChatBubbleLeftRightIcon,
  QuestionMarkCircleIcon,
  DocumentTextIcon,
  CreditCardIcon,
  TruckIcon,
} from '@heroicons/react/24/outline';

interface WelcomeScreenProps {
  onSampleQuestion: (question: string) => void;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onSampleQuestion }) => {
  const sampleQuestions = [
    {
      icon: QuestionMarkCircleIcon,
      text: "How do I reset my password?",
      category: "Account",
    },
    {
      icon: CreditCardIcon,
      text: "What is your refund policy?",
      category: "Billing",
    },
    {
      icon: TruckIcon,
      text: "How can I track my order?",
      category: "Orders",
    },
    {
      icon: DocumentTextIcon,
      text: "Where can I find my invoices?",
      category: "Documents",
    },
  ];

  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="max-w-2xl text-center">
        <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-blue-600">
          <ChatBubbleLeftRightIcon className="h-10 w-10 text-white" />
        </div>

        <h2 className="mb-3 text-2xl font-semibold text-gray-900">
          Welcome to Customer Support
        </h2>
        <p className="mb-8 text-gray-600">
          I'm here to help you 24/7. Ask me anything about your account,
          orders, billing, or any other questions you might have.
        </p>

        <div className="mb-6">
          <p className="mb-4 text-sm font-medium text-gray-700">
            Try asking one of these common questions:
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            {sampleQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => onSampleQuestion(question.text)}
                className="group flex items-start rounded-lg border border-gray-200 bg-white p-3 text-left hover:border-blue-300 hover:bg-blue-50 transition-all"
              >
                <question.icon className="mr-3 h-5 w-5 flex-shrink-0 text-gray-400 group-hover:text-blue-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900 group-hover:text-blue-700">
                    {question.text}
                  </p>
                  <p className="text-xs text-gray-500">{question.category}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-lg bg-blue-50 p-4">
          <p className="text-sm text-blue-700">
            💡 <span className="font-medium">Pro tip:</span> You can upload
            documents or images to get help with specific issues!
          </p>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;
