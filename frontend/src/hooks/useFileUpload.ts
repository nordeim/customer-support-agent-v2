/**
 * Custom hook for file upload management
 * Provides file validation, upload progress, and error handling
 */
import { useState, useCallback, useRef } from 'react';
import { api } from '../services/api';
import toast from 'react-hot-toast';

export interface FileUploadOptions {
  maxFiles?: number;
  maxSizeMB?: number;
  acceptedFormats?: string[];
  autoUpload?: boolean;
  multiple?: boolean;
  onSuccess?: (file: UploadedFile) => void;
  onError?: (error: FileUploadError) => void;
  onProgress?: (progress: FileUploadProgress) => void;
}

export interface UploadedFile {
  id: string;
  filename: string;
  size: number;
  type: string;
  url?: string;
  preview?: string;
  processed?: boolean;
  metadata?: Record<string, any>;
}

export interface FileUploadError {
  file: File;
  error: string;
  code: 'size' | 'type' | 'upload' | 'processing' | 'unknown';
}

export interface FileUploadProgress {
  file: File;
  progress: number;
  loaded: number;
  total: number;
}

export interface FileWithStatus {
  file: File;
  id: string;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
  result?: UploadedFile;
}

export interface UseFileUploadReturn {
  // State
  files: FileWithStatus[];
  isUploading: boolean;
  uploadProgress: number;
  errors: FileUploadError[];
  
  // Actions
  selectFiles: (files: File[]) => void;
  uploadFile: (file: File, sessionId: string) => Promise<UploadedFile | null>;
  uploadFiles: (files: File[], sessionId: string) => Promise<UploadedFile[]>;
  removeFile: (id: string) => void;
  clearFiles: () => void;
  clearErrors: () => void;
  retryFile: (id: string, sessionId: string) => Promise<void>;
  
  // Validation
  validateFile: (file: File) => { valid: boolean; error?: string };
  validateFiles: (files: File[]) => { valid: File[]; invalid: FileUploadError[] };
  
  // Utilities
  formatFileSize: (bytes: number) => string;
  getFileIcon: (file: File) => string;
  getFilePreview: (file: File) => Promise<string | null>;
}

const DEFAULT_OPTIONS: FileUploadOptions = {
  maxFiles: 5,
  maxSizeMB: 10,
  acceptedFormats: [
    '.pdf', '.doc', '.docx', '.txt', '.md',
    '.csv', '.xlsx', '.xls', '.json', '.xml',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp',
  ],
  autoUpload: false,
  multiple: true,
};

export function useFileUpload(options: FileUploadOptions = {}): UseFileUploadReturn {
  // Merge options with defaults
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  // State
  const [files, setFiles] = useState<FileWithStatus[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [errors, setErrors] = useState<FileUploadError[]>([]);
  
  // Refs
  const uploadQueue = useRef<AbortController[]>([]);
  const fileIdCounter = useRef(0);
  
  /**
   * Generate unique file ID
   */
  const generateFileId = useCallback((): string => {
    fileIdCounter.current += 1;
    return `file_${Date.now()}_${fileIdCounter.current}`;
  }, []);
  
  /**
   * Format file size for display
   */
  const formatFileSize = useCallback((bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  }, []);
  
  /**
   * Get file icon based on type
   */
  const getFileIcon = useCallback((file: File): string => {
    const type = file.type.toLowerCase();
    
    if (type.startsWith('image/')) return '🖼️';
    if (type.startsWith('video/')) return '🎥';
    if (type.startsWith('audio/')) return '🎵';
    if (type.includes('pdf')) return '📄';
    if (type.includes('word') || type.includes('document')) return '📝';
    if (type.includes('sheet') || type.includes('excel')) return '📊';
    if (type.includes('presentation') || type.includes('powerpoint')) return '📑';
    if (type.includes('zip') || type.includes('compress')) return '🗜️';
    if (type.includes('text') || type.includes('plain')) return '📃';
    if (type.includes('json') || type.includes('xml')) return '📋';
    
    return '📎';
  }, []);
  
  /**
   * Get file preview (for images)
   */
  const getFilePreview = useCallback(async (file: File): Promise<string | null> => {
    if (!file.type.startsWith('image/')) {
      return null;
    }
    
    return new Promise((resolve) => {
      const reader = new FileReader();
      
      reader.onloadend = () => {
        resolve(reader.result as string);
      };
      
      reader.onerror = () => {
        resolve(null);
      };
      
      reader.readAsDataURL(file);
    });
  }, []);
  
  /**
   * Validate single file
   */
  const validateFile = useCallback((file: File): { valid: boolean; error?: string } => {
    // Check file type
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (opts.acceptedFormats && !opts.acceptedFormats.includes(extension)) {
      return {
        valid: false,
        error: `File type ${extension} is not supported. Accepted: ${opts.acceptedFormats.join(', ')}`,
      };
    }
    
    // Check file size
    const maxSize = (opts.maxSizeMB || 10) * 1024 * 1024;
    if (file.size > maxSize) {
      return {
        valid: false,
        error: `File size (${formatFileSize(file.size)}) exceeds maximum of ${opts.maxSizeMB}MB`,
      };
    }
    
    // Check file name
    if (file.name.length > 255) {
      return {
        valid: false,
        error: 'File name is too long (max 255 characters)',
      };
    }
    
    return { valid: true };
  }, [opts.acceptedFormats, opts.maxSizeMB, formatFileSize]);
  
  /**
   * Validate multiple files
   */
  const validateFiles = useCallback((filesToValidate: File[]): {
    valid: File[];
    invalid: FileUploadError[];
  } => {
    const valid: File[] = [];
    const invalid: FileUploadError[] = [];
    
    // Check total file count
    if (opts.maxFiles && files.length + filesToValidate.length > opts.maxFiles) {
      const allowed = Math.max(0, opts.maxFiles - files.length);
      toast.error(`Maximum ${opts.maxFiles} files allowed. You can add ${allowed} more.`);
      filesToValidate = filesToValidate.slice(0, allowed);
    }
    
    // Validate each file
    for (const file of filesToValidate) {
      const validation = validateFile(file);
      
      if (validation.valid) {
        valid.push(file);
      } else {
        invalid.push({
          file,
          error: validation.error || 'Invalid file',
          code: validation.error?.includes('size') ? 'size' : 'type',
        });
      }
    }
    
    return { valid, invalid };
  }, [opts.maxFiles, files.length, validateFile]);
  
  /**
   * Select files for upload
   */
  const selectFiles = useCallback((newFiles: File[]) => {
    const { valid, invalid } = validateFiles(newFiles);
    
    // Add errors for invalid files
    if (invalid.length > 0) {
      setErrors((prev) => [...prev, ...invalid]);
      
      // Show toast for each invalid file
      invalid.forEach((err) => {
        toast.error(`${err.file.name}: ${err.error}`);
      });
    }
    
    // Add valid files
    if (valid.length > 0) {
      const fileStatuses: FileWithStatus[] = valid.map((file) => ({
        file,
        id: generateFileId(),
        status: 'pending',
        progress: 0,
      }));
      
      setFiles((prev) => [...prev, ...fileStatuses]);
      
      // Show success toast
      toast.success(`${valid.length} file(s) selected`);
    }
  }, [validateFiles, generateFileId]);
  
  /**
   * Upload single file
   */
  const uploadFile = useCallback(async (
    file: File,
    sessionId: string
  ): Promise<UploadedFile | null> => {
    // Validate file
    const validation = validateFile(file);
    if (!validation.valid) {
      const error: FileUploadError = {
        file,
        error: validation.error || 'Invalid file',
        code: 'type',
      };
      
      setErrors((prev) => [...prev, error]);
      opts.onError?.(error);
      
      return null;
    }
    
    // Create abort controller
    const abortController = new AbortController();
    uploadQueue.current.push(abortController);
    
    try {
      // Upload file
      const result = await api.uploadFile(
        file,
        sessionId,
        (progress) => {
          // Update progress
          setFiles((prev) =>
            prev.map((f) =>
              f.file === file
                ? { ...f, progress, status: 'uploading' as const }
                : f
            )
          );
          
          // Notify progress
          opts.onProgress?.({
            file,
            progress,
            loaded: (file.size * progress) / 100,
            total: file.size,
          });
        }
      );
      
      // Create uploaded file object
      const uploadedFile: UploadedFile = {
        id: generateFileId(),
        filename: result.filename,
        size: file.size,
        type: file.type,
        preview: result.preview,
        processed: result.processed,
      };
      
      // Update file status
      setFiles((prev) =>
        prev.map((f) =>
          f.file === file
            ? { ...f, status: 'completed' as const, result: uploadedFile }
            : f
        )
      );
      
      // Notify success
      opts.onSuccess?.(uploadedFile);
      toast.success(`${file.name} uploaded successfully`);
      
      return uploadedFile;
      
    } catch (error: any) {
      // Handle error
      const fileError: FileUploadError = {
        file,
        error: error.response?.data?.message || error.message || 'Upload failed',
        code: 'upload',
      };
      
      setFiles((prev) =>
        prev.map((f) =>
          f.file === file
            ? { ...f, status: 'error' as const, error: fileError.error }
            : f
        )
      );
      
      setErrors((prev) => [...prev, fileError]);
      opts.onError?.(fileError);
      
      toast.error(`Failed to upload ${file.name}`);
      
      return null;
      
    } finally {
      // Remove abort controller
      const index = uploadQueue.current.indexOf(abortController);
      if (index > -1) {
        uploadQueue.current.splice(index, 1);
      }
    }
  }, [validateFile, opts, generateFileId]);
  
  /**
   * Upload multiple files
   */
  const uploadFiles = useCallback(async (
    filesToUpload: File[],
    sessionId: string
  ): Promise<UploadedFile[]> => {
    setIsUploading(true);
    
    const results: UploadedFile[] = [];
    
    try {
      // Upload files sequentially to avoid overwhelming the server
      for (const file of filesToUpload) {
        const result = await uploadFile(file, sessionId);
        if (result) {
          results.push(result);
        }
      }
      
      return results;
      
    } finally {
      setIsUploading(false);
    }
  }, [uploadFile]);
  
  /**
   * Remove file from list
   */
  const removeFile = useCallback((id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
    setErrors((prev) => prev.filter((e) => e.file.name !== id));
  }, []);
  
  /**
   * Clear all files
   */
  const clearFiles = useCallback(() => {
    // Abort any ongoing uploads
    uploadQueue.current.forEach((controller) => controller.abort());
    uploadQueue.current = [];
    
    setFiles([]);
    setErrors([]);
    setIsUploading(false);
  }, []);
  
  /**
   * Clear errors
   */
  const clearErrors = useCallback(() => {
    setErrors([]);
  }, []);
  
  /**
   * Retry failed upload
   */
  const retryFile = useCallback(async (id: string, sessionId: string) => {
    const fileStatus = files.find((f) => f.id === id);
    if (!fileStatus || fileStatus.status !== 'error') {
      return;
    }
    
    // Reset status
    setFiles((prev) =>
      prev.map((f) =>
        f.id === id
          ? { ...f, status: 'pending' as const, progress: 0, error: undefined }
          : f
      )
    );
    
    // Remove error
    setErrors((prev) => prev.filter((e) => e.file !== fileStatus.file));
    
    // Retry upload
    await uploadFile(fileStatus.file, sessionId);
  }, [files, uploadFile]);
  
  // Calculate overall upload progress
  const uploadProgress = files.length > 0
    ? files.reduce((sum, f) => sum + f.progress, 0) / files.length
    : 0;
  
  return {
    // State
    files,
    isUploading,
    uploadProgress,
    errors,
    
    // Actions
    selectFiles,
    uploadFile,
    uploadFiles,
    removeFile,
    clearFiles,
    clearErrors,
    retryFile,
    
    // Validation
    validateFile,
    validateFiles,
    
    // Utilities
    formatFileSize,
    getFileIcon,
    getFilePreview,
  };
}

export default useFileUpload;
