"use client";

import { useState, useCallback, useRef, type DragEvent, type ChangeEvent } from "react";
import { compressImage } from "@/lib/api";
import { UPLOAD_MAX_WIDTH, COMPRESS_QUALITY } from "@/lib/constants";

interface UseFileUploadOptions {
  onFile: (base64: string) => void | Promise<void>;
}

export function useFileUpload({ onFile }: UseFileUploadOptions) {
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) {
        return; // silently ignore non-images; caller can handle error if desired
      }

      setIsProcessing(true);

      try {
        const raw = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = () => reject(new Error("Failed to read file"));
          reader.readAsDataURL(file);
        });

        // Always compress uploads — photos from phones can be 5-15MB
        const compressed = await compressImage(raw, UPLOAD_MAX_WIDTH, COMPRESS_QUALITY);
        setUploadPreview(compressed);
        await onFile(compressed);
      } catch {
        // Error will be handled by the caller's analyze error state
      } finally {
        setIsProcessing(false);
      }
    },
    [onFile],
  );

  const clearPreview = useCallback(() => setUploadPreview(null), []);

  const dropProps = {
    onDragOver: (e: DragEvent) => {
      e.preventDefault();
      setDragOver(true);
    },
    onDragLeave: () => setDragOver(false),
    onDrop: (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
  };

  const inputProps = {
    ref: inputRef,
    type: "file" as const,
    accept: "image/*",
    className: "hidden",
    onChange: (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
  };

  const triggerFileSelect = useCallback(() => {
    inputRef.current?.click();
  }, []);

  return {
    uploadPreview,
    dragOver,
    isProcessing,
    handleFile,
    clearPreview,
    dropProps,
    inputProps,
    triggerFileSelect,
  };
}
