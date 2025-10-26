/**
 * Sources panel for displaying referenced sources
 */
import React from 'react';
import { XMarkIcon, DocumentTextIcon, StarIcon } from '@heroicons/react/24/outline';
import { Source } from '../types';

interface SourcesPanelProps {
  sources: Source[];
  onClose: () => void;
}

const SourcesPanel: React.FC<SourcesPanelProps> = ({ sources, onClose }) => {
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-6 py-4">
        <h2 className="text-lg font-semibold text-gray-900">Sources</h2>
        <button
          onClick={onClose}
          className="rounded-lg p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
        >
          <XMarkIcon className="h-5 w-5" />
        </button>
      </div>

      {/* Sources list */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-4">
          {sources.map((source, index) => (
            <SourceCard key={index} source={source} index={index} />
          ))}
        </div>
      </div>
    </div>
  );
};

const SourceCard: React.FC<{ source: Source; index: number }> = ({
  source,
  index,
}) => {
  const relevancePercent = Math.round(source.relevanceScore * 100);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 hover:border-blue-300 hover:shadow-md transition-all cursor-pointer">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center">
          <DocumentTextIcon className="h-5 w-5 text-gray-400 mr-2" />
          <span className="text-sm font-medium text-gray-900">
            Source #{index + 1}
          </span>
        </div>
        <div className="flex items-center">
          {/* Relevance indicator */}
          <div className="flex items-center">
            {[...Array(5)].map((_, i) => (
              <StarIcon
                key={i}
                className={`h-3 w-3 ${
                  i < Math.round(source.relevanceScore * 5)
                    ? 'text-yellow-400 fill-current'
                    : 'text-gray-300'
                }`}
              />
            ))}
          </div>
          <span className="ml-2 text-xs text-gray-500">
            {relevancePercent}% match
          </span>
        </div>
      </div>

      <p className="text-sm text-gray-700 line-clamp-4">{source.content}</p>

      {source.metadata && (
        <div className="mt-3 flex flex-wrap gap-2">
          {source.metadata.source && (
            <span className="inline-flex items-center rounded-full bg-blue-100 px-2 py-1 text-xs font-medium text-blue-700">
              {source.metadata.source}
            </span>
          )}
          {source.metadata.page && (
            <span className="inline-flex items-center rounded-full bg-gray-100 px-2 py-1 text-xs text-gray-600">
              Page {source.metadata.page}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default SourcesPanel;
