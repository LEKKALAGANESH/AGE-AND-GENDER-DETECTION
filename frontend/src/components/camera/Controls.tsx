"use client";

import { type ChangeEvent } from "react";
import {
  Camera as CameraIcon,
  Upload,
  Video,
  VideoOff,
  Loader2,
} from "lucide-react";
import clsx from "clsx";
import type { CameraStatus } from "@/lib/types";

interface ControlsProps {
  status: CameraStatus;
  isStreaming: boolean;
  isUploading: boolean;
  onStartCamera: () => void;
  onStopCamera: () => void;
  onCapture: () => void;
  onToggleStream: () => void;
  onFileSelect: (e: ChangeEvent<HTMLInputElement>) => void;
}

export function Controls({
  status,
  isStreaming,
  isUploading,
  onStartCamera,
  onStopCamera,
  onCapture,
  onToggleStream,
  onFileSelect,
}: ControlsProps) {
  const isActive = status === "active" || status === "streaming";

  return (
    <div className="flex items-center justify-center gap-3">
      {!isActive ? (
        <>
          <button
            onClick={onStartCamera}
            disabled={status === "starting"}
            aria-label="Start camera"
            className="flex items-center gap-2 px-5 py-2.5 bg-white text-black rounded-lg font-medium text-sm hover:bg-zinc-200 transition-colors disabled:opacity-50"
          >
            {status === "starting" ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Video className="w-4 h-4" />
            )}
            {status === "starting" ? "Starting..." : "Start Camera"}
          </button>
          <label
            className={clsx(
              "flex items-center gap-2 px-5 py-2.5 bg-zinc-800 text-white rounded-lg font-medium text-sm hover:bg-zinc-700 transition-colors cursor-pointer",
              isUploading && "opacity-50 pointer-events-none",
            )}
            aria-label="Upload image"
          >
            {isUploading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Upload className="w-4 h-4" />
            )}
            {isUploading ? "Processing..." : "Upload Image"}
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={onFileSelect}
              disabled={isUploading}
            />
          </label>
        </>
      ) : (
        <>
          <button
            onClick={onCapture}
            disabled={isStreaming}
            aria-label="Capture single frame for analysis"
            className="flex items-center gap-2 px-5 py-2.5 bg-white text-black rounded-lg font-medium text-sm hover:bg-zinc-200 transition-colors disabled:opacity-50"
          >
            <CameraIcon className="w-4 h-4" />
            Capture
          </button>
          <button
            onClick={onToggleStream}
            aria-label={isStreaming ? "Stop live detection" : "Start live detection"}
            className={clsx(
              "flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium text-sm transition-colors",
              isStreaming
                ? "bg-red-600 text-white hover:bg-red-700"
                : "bg-emerald-600 text-white hover:bg-emerald-700",
            )}
          >
            {isStreaming ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Video className="w-4 h-4" />
            )}
            {isStreaming ? "Stop Live" : "Live Detect"}
          </button>
          <button
            onClick={onStopCamera}
            aria-label="Stop camera"
            className="flex items-center gap-2 px-5 py-2.5 bg-zinc-800 text-white rounded-lg font-medium text-sm hover:bg-zinc-700 transition-colors"
          >
            <VideoOff className="w-4 h-4" />
            Stop
          </button>
        </>
      )}
    </div>
  );
}
