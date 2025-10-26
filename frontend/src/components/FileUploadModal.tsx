/**
 * File upload modal with drag and drop
 */
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import {
  XMarkIcon,
  CloudArrowUpIcon,
  DocumentIcon,
  PhotoIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

interface FileUploadModalProps {
  onSelect: (files: File[]) => void;
  onClose: () => void;
  maxFiles?: number;
  maxSizeMB?: number;
  acceptedFormats?: string[];
}

const FileUploadModal: React.FC<FileUploadModalProps> = ({
  onSelect,
  onClose,
  maxFiles = 5,
  maxSizeMB = 10,
  acceptedFormats = ['.pdf', '.doc', '.docx', '.txt', '.png', '.jpg'],
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: any[]) => {
      // Handle rejected files
      rejectedFiles.forEach((file) => {
        const errors = file.errors.map((e: any) => e.message).join(', ');
        toast.error(`${file.file.name}: ${errors}`);
      });

      // Add accepted files
      setSelectedFiles((prev) => {
        const newFiles = [...prev, ...acceptedFiles];
        if (newFiles.length > maxFiles) {
          toast.error(`Maximum ${maxFiles} files allowed`);
          return prev;
        }
        return newFiles;
      });
    },
    [maxFiles]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedFormats.reduce((acc, format) => {
      const mimeTypes: { [key: string]: string[] } = {
        '.pdf': ['application/pdf'],
        '.doc': ['application/msword'],
        '.docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        '.txt': ['text/plain'],
        '.png': ['image/png'],
        '.jpg': ['image/jpeg'],
        '.jpeg': ['image/jpeg'],
        '.csv': ['text/csv'],
        '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
      };
      
      if (mimeTypes[format]) {
        mimeTypes[format].forEach(mime => {
          acc[mime] = [format];
        });
      }
      return acc;
    }, {} as { [key: string]: string[] }),
    maxSize: maxSizeMB * 1024 * 1024,
    multiple: true,
  });

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = () => {
    if (selectedFiles.length === 0) {
      toast.error('Please select at least one file');
      return;
    }
    onSelect(selectedFiles);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('image/')) {
      return PhotoIcon;
    }
    return DocumentIcon;
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95 }}
        animate={{ scale: 1 }}
        exit={{ scale: 0.95 }}
        className="w-full max-w-lg rounded-lg bg-white shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b px-6 py-4">
          <h2 className="text-lg font-semibold text-gray-900">
            Upload Files
          </h2>
          <button
            onClick={onClose}
            className="rounded-lg p-1 text-gray-400 hover:bg-gray-100"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={`rounded-lg border-2 border-dashed p-8 text-center transition-colors cursor-pointer ${
              isDragActive
                ? 'border-blue-400 bg-blue-50'
                : 'border-gray-300 bg-gray-50 hover:border-gray-400'
            }`}
          >
            <input {...getInputProps()} />
            <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
            <p className="mt-2 text-sm text-gray-600">
              {isDragActive
                ? 'Drop files here...'
                : 'Drag & drop files here, or click to browse'}
            </p>
            <p className="mt-1 text-xs text-gray-500">
              Max {maxFiles} files, up to {maxSizeMB}MB each
            </p>
            <p className="mt-1 text-xs text-gray-500">
              Supported: {acceptedFormats.join(', ')}
            </p>
          </div>

          {/* Selected files */}
          {selectedFiles.length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-sm font-medium text-gray-700">
                Selected Files ({selectedFiles.length})
              </h3>
              <div className="space-y-2">
                {selectedFiles.map((file, index) => {
                  const Icon = getFileIcon(file);
                  return (
                    <div
                      key={index}
                      className="flex items-center justify-between rounded-lg border border-gray-200 bg-white p-3"
                    >
                      <div className="flex items-center">
                        <Icon className="h-5 w-5 text-gray-400 mr-3" />
                        <div>
                          <p className="text-sm font-medium text-gray-900">
                            {file.name}
                          </p>
                          <p className="text-xs text-gray-500">
                            {formatFileSize(file.size)}
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => removeFile(index)}
                        className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-red-600"
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end space-x-3 border-t px-6 py-4">
          <button
            onClick={onClose}
            className="rounded-lg px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={selectedFiles.length === 0}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            Upload {selectedFiles.length > 0 && `(${selectedFiles.length})`}
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default FileUploadModal;
