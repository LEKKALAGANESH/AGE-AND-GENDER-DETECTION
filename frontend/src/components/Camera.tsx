"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Camera as CameraIcon, Loader2 } from "lucide-react";
import clsx from "clsx";

import { useCamera } from "@/hooks/useCamera";
import { useAnalyze } from "@/hooks/useAnalyze";
import { useFileUpload } from "@/hooks/useFileUpload";
import { useTemporalSmoothing } from "@/hooks/useTemporalSmoothing";
import { drawOverlay } from "@/lib/canvas";

import { DropZone } from "@/components/camera/DropZone";
import { Controls } from "@/components/camera/Controls";
import { ResultsPanel } from "@/components/camera/ResultsPanel";
import { DebugPanel } from "@/components/camera/DebugPanel";

export function Camera() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const busyRef = useRef(false);
  const streamingRef = useRef(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { videoRef, status, setStatus, startCamera, stopCamera } = useCamera();
  const { results, error, isLoading, analyze, clearError, setResults } = useAnalyze();
  const { smoothResults, reset: resetSmoothing } = useTemporalSmoothing();

  const [streaming, setStreaming] = useState(false);
  const [debug, setDebug] = useState(false);

  // Keep ref in sync with state so the recursive setTimeout reads current value
  useEffect(() => {
    streamingRef.current = streaming;
  }, [streaming]);

  // ── Handle file upload: compress then analyze ──
  const handleUploadAnalyze = useCallback(
    async (base64: string) => {
      await analyze(base64);
    },
    [analyze],
  );

  const {
    uploadPreview,
    dragOver,
    isProcessing: isUploading,
    clearPreview,
    dropProps,
    inputProps,
    triggerFileSelect,
  } = useFileUpload({ onFile: handleUploadAnalyze });

  const isActive = status === "active" || status === "streaming";

  // ── Capture one frame and analyze ──
  const captureFrame = useCallback(async () => {
    if (busyRef.current || !videoRef.current || !canvasRef.current) return;
    busyRef.current = true;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      busyRef.current = false;
      return;
    }
    ctx.drawImage(video, 0, 0);

    const base64 = canvas.toDataURL("image/jpeg", 0.8);
    const data = await analyze(base64);
    if (data && overlayRef.current) {
      // Apply temporal smoothing only during live streaming
      const displayResults = streaming ? smoothResults(data.results) : data.results;
      const displayData = { ...data, results: displayResults };
      setResults(displayData);
      drawOverlay(overlayRef.current, displayData, video.videoWidth, video.videoHeight, streaming);
    }
    busyRef.current = false;
  }, [analyze, videoRef, streaming, smoothResults, setResults]);

  // ── Recursive setTimeout for streaming (backpressure-safe) ──
  // Waits for previous frame to complete before scheduling next.
  // MIN_FRAME_GAP_MS prevents hammering the server even on fast networks.
  const MIN_FRAME_GAP_MS = 200; // ~5fps max — balances responsiveness vs server load
  const scheduleNextFrame = useCallback(() => {
    if (!streamingRef.current) return;
    timeoutRef.current = setTimeout(async () => {
      if (!streamingRef.current) return;
      await captureFrame();
      // Only schedule next after previous completes — natural backpressure
      scheduleNextFrame();
    }, MIN_FRAME_GAP_MS);
  }, [captureFrame]);

  // ── Toggle continuous live detection ──
  const toggleStream = useCallback(() => {
    if (streaming) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      setStreaming(false);
      setStatus("active");
      resetSmoothing();
    } else {
      setStreaming(true);
      setStatus("streaming");
      streamingRef.current = true;
      scheduleNextFrame();
    }
  }, [streaming, scheduleNextFrame, setStatus, resetSmoothing]);

  // ── Start camera (clear upload state first) ──
  const handleStartCamera = useCallback(() => {
    clearPreview();
    clearError();
    startCamera();
  }, [clearPreview, clearError, startCamera]);

  // ── Stop camera (also stop streaming) ──
  const handleStopCamera = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setStreaming(false);
    resetSmoothing();
    stopCamera();
    // Clear overlay
    const ctx = overlayRef.current?.getContext("2d");
    if (ctx && overlayRef.current) {
      ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    }
  }, [stopCamera, resetSmoothing]);

  // ── Cleanup streaming timeout on unmount ──
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="w-full max-w-2xl space-y-4">
      {/* Video / Upload area */}
      <DropZone dragOver={dragOver} dropProps={dropProps} onTriggerFileSelect={triggerFileSelect}>
        {/* Hidden file input for keyboard-triggered uploads */}
        <input {...inputProps} />

        {/* Webcam video (hidden when showing upload) */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={clsx("w-full h-full object-cover", (!isActive || uploadPreview) && "hidden")}
        />

        {/* Upload preview */}
        {uploadPreview && !isActive && (
          <img src={uploadPreview} alt="Uploaded preview" className="w-full h-full object-contain" />
        )}

        {/* Overlay canvas for bounding boxes */}
        <canvas
          ref={overlayRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          aria-hidden="true"
        />

        {/* Hidden capture canvas */}
        <canvas ref={canvasRef} className="hidden" aria-hidden="true" />

        {/* Placeholder when nothing is active */}
        {!isActive && !uploadPreview && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-zinc-500">
            {status === "starting" ? (
              <>
                <Loader2 className="w-12 h-12 animate-spin" />
                <p className="text-sm">Starting camera...</p>
              </>
            ) : (
              <>
                <CameraIcon className="w-12 h-12" />
                <p className="text-sm">Start camera or drop an image</p>
              </>
            )}
          </div>
        )}

        {/* Loading overlay during upload analysis */}
        {(isUploading || (isLoading && !isActive && uploadPreview)) && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <Loader2 className="w-8 h-8 animate-spin text-white" />
          </div>
        )}
      </DropZone>

      {/* Controls */}
      <Controls
        status={status}
        isStreaming={streaming}
        isUploading={isUploading || (isLoading && !isActive)}
        onStartCamera={handleStartCamera}
        onStopCamera={handleStopCamera}
        onCapture={captureFrame}
        onToggleStream={toggleStream}
        onFileSelect={inputProps.onChange}
      />

      {/* Debug toggle (dev only) */}
      {process.env.NODE_ENV !== "production" && (
        <>
          <div className="flex justify-center">
            <button
              onClick={() => setDebug((d) => !d)}
              aria-label={debug ? "Hide debug panel" : "Show debug panel"}
              className="text-[10px] text-zinc-600 hover:text-zinc-400 transition-colors"
            >
              {debug ? "Hide" : "Show"} Debug
            </button>
          </div>
          {debug && <DebugPanel />}
        </>
      )}

      {/* Error */}
      {error && (
        <div className="text-red-400 text-sm text-center bg-red-500/10 rounded-lg p-3" role="alert">
          {error}
        </div>
      )}

      {/* Results */}
      <ResultsPanel results={results} isStreaming={streaming} />
    </div>
  );
}
