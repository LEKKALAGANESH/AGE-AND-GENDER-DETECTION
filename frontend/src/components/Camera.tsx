"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Camera as CameraIcon, Upload, Video, VideoOff, Loader2 } from "lucide-react";
import clsx from "clsx";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "/api/analyze";
const HEALTH_URL = API_URL.replace(/\/analyze$/, "/health");
const MAX_PAYLOAD_BYTES = 4 * 1024 * 1024; // 4MB safety margin (Vercel limit: 4.5MB)

/** Compress an image to fit within Vercel's payload limit. */
function compressImage(
  base64: string,
  maxWidth = 800,
  quality = 0.7,
): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const scale = img.width > maxWidth ? maxWidth / img.width : 1;
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);

      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, w, h);
      resolve(canvas.toDataURL("image/jpeg", quality));
    };
    img.src = base64;
  });
}

interface FaceResult {
  age: number;
  gender: string;
  confidence: number;
  region: [number, number, number, number];
}

interface AnalyzeResponse {
  results: FaceResult[];
  face_count: number;
  processing_time_ms: number;
}

export function Camera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const busyRef = useRef(false);

  const [active, setActive] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [results, setResults] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [debug, setDebug] = useState(false);
  const [debugMsg, setDebugMsg] = useState<string | null>(null);

  // ── Start webcam ──
  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setActive(true);
      }
    } catch {
      setError("Camera access denied.");
    }
  }, []);

  // ── Stop webcam ──
  const stopCamera = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
    setStreaming(false);
    // Clear overlay
    const ctx = overlayRef.current?.getContext("2d");
    if (ctx && overlayRef.current) ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
  }, []);

  // ── Analyze a single base64 image ──
  const analyze = useCallback(async (base64: string): Promise<AnalyzeResponse | null> => {
    try {
      // Compress if payload would exceed Vercel's limit
      let image = base64;
      const rawSize = new Blob([JSON.stringify({ image })]).size;
      if (rawSize > MAX_PAYLOAD_BYTES) {
        image = await compressImage(base64, 800, 0.6);
      }

      const body = JSON.stringify({ image });
      if (body.length > MAX_PAYLOAD_BYTES) {
        throw new Error(`Image too large (${(body.length / 1024 / 1024).toFixed(1)}MB). Try a smaller image.`);
      }

      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return await res.json();
    } catch (e) {
      setError(e instanceof Error ? e.message : "API error");
      return null;
    }
  }, []);

  // ── Draw detection overlays ──
  const drawOverlay = useCallback((data: AnalyzeResponse, width: number, height: number) => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    data.results.forEach((face) => {
      const [rx, ry, rw, rh] = face.region;

      // Support both normalized (0-1) and pixel coordinates
      const isNormalized = rx <= 1 && ry <= 1 && rw <= 1 && rh <= 1;
      const x = isNormalized ? rx * width : rx;
      const y = isNormalized ? ry * height : ry;
      const w = isNormalized ? rw * width : rw;
      const h = isNormalized ? rh * height : rh;

      const color = face.gender === "Male" ? "#3b82f6" : "#ec4899";

      // Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // Label background
      const label = `${face.gender}, ${face.age} yrs`;
      ctx.font = "bold 12px system-ui, sans-serif";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = `${color}cc`;
      ctx.fillRect(x, y - 22, tw + 12, 22);

      // Label text
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 6, y - 6);
    });
  }, []);

  // ── Capture one frame and analyze ──
  const captureFrame = useCallback(async () => {
    if (busyRef.current || !videoRef.current || !canvasRef.current) return;
    busyRef.current = true;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) { busyRef.current = false; return; }
    ctx.drawImage(video, 0, 0);

    const base64 = canvas.toDataURL("image/jpeg", 0.8);
    const data = await analyze(base64);
    if (data) {
      setResults(data);
      drawOverlay(data, video.videoWidth, video.videoHeight);
    }
    busyRef.current = false;
  }, [analyze, drawOverlay]);

  // ── Toggle continuous streaming at ~10fps ──
  const toggleStream = useCallback(() => {
    if (streaming) {
      if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
      setStreaming(false);
    } else {
      setStreaming(true);
      intervalRef.current = setInterval(captureFrame, 100);
    }
  }, [streaming, captureFrame]);

  // ── Handle file drop/upload ──
  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) { setError("Not an image file."); return; }
    setError(null);

    const reader = new FileReader();
    reader.onload = async () => {
      const raw = reader.result as string;
      // Always compress uploads — photos from phones can be 5-15MB
      const base64 = await compressImage(raw, 1024, 0.7);
      setUploadPreview(base64);
      const data = await analyze(base64);
      if (data) {
        setResults(data);
      }
    };
    reader.readAsDataURL(file);
  }, [analyze]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  }, []);

  return (
    <div className="w-full max-w-2xl space-y-4">
      {/* ── Video / Upload area ── */}
      <div
        className={clsx(
          "relative aspect-video rounded-xl overflow-hidden border-2 border-dashed transition-colors",
          dragOver ? "border-blue-500 bg-blue-500/5" : "border-zinc-800 bg-zinc-900"
        )}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
      >
        {/* Webcam video (hidden when showing upload) */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={clsx("w-full h-full object-cover", (!active || uploadPreview) && "hidden")}
        />

        {/* Upload preview */}
        {uploadPreview && !active && (
          <img src={uploadPreview} alt="Uploaded" className="w-full h-full object-contain" />
        )}

        {/* Overlay canvas for bounding boxes */}
        <canvas
          ref={overlayRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Hidden capture canvas */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Placeholder when nothing is active */}
        {!active && !uploadPreview && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-zinc-500">
            <CameraIcon className="w-12 h-12" />
            <p className="text-sm">Start camera or drop an image</p>
          </div>
        )}
      </div>

      {/* ── Controls ── */}
      <div className="flex items-center justify-center gap-3">
        {!active ? (
          <>
            <button
              onClick={() => { setUploadPreview(null); startCamera(); }}
              className="flex items-center gap-2 px-5 py-2.5 bg-white text-black rounded-lg font-medium text-sm hover:bg-zinc-200 transition-colors"
            >
              <Video className="w-4 h-4" />
              Start Camera
            </button>
            <label className="flex items-center gap-2 px-5 py-2.5 bg-zinc-800 text-white rounded-lg font-medium text-sm hover:bg-zinc-700 transition-colors cursor-pointer">
              <Upload className="w-4 h-4" />
              Upload Image
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFile(file);
                }}
              />
            </label>
          </>
        ) : (
          <>
            <button
              onClick={captureFrame}
              disabled={streaming}
              className="flex items-center gap-2 px-5 py-2.5 bg-white text-black rounded-lg font-medium text-sm hover:bg-zinc-200 transition-colors disabled:opacity-50"
            >
              <CameraIcon className="w-4 h-4" />
              Capture
            </button>
            <button
              onClick={toggleStream}
              className={clsx(
                "flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium text-sm transition-colors",
                streaming
                  ? "bg-red-600 text-white hover:bg-red-700"
                  : "bg-emerald-600 text-white hover:bg-emerald-700"
              )}
            >
              {streaming ? <Loader2 className="w-4 h-4 animate-spin" /> : <Video className="w-4 h-4" />}
              {streaming ? "Stop Stream" : "Stream 10fps"}
            </button>
            <button
              onClick={stopCamera}
              className="flex items-center gap-2 px-5 py-2.5 bg-zinc-800 text-white rounded-lg font-medium text-sm hover:bg-zinc-700 transition-colors"
            >
              <VideoOff className="w-4 h-4" />
              Stop
            </button>
          </>
        )}
      </div>

      {/* ── Debug toggle ── */}
      <div className="flex justify-center">
        <button
          onClick={() => setDebug((d) => !d)}
          className="text-[10px] text-zinc-600 hover:text-zinc-400 transition-colors"
        >
          {debug ? "Hide" : "Show"} Debug
        </button>
      </div>
      {debug && (
        <div className="space-y-2 p-3 rounded-lg bg-zinc-900 border border-zinc-800 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="text-zinc-500">API:</span>
            <span className="text-zinc-300">{API_URL}</span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={async () => {
                setDebugMsg("Pinging...");
                try {
                  const t0 = performance.now();
                  const res = await fetch(HEALTH_URL);
                  const ms = Math.round(performance.now() - t0);
                  const data = await res.json();
                  setDebugMsg(`${res.status} OK — ${ms}ms — ${JSON.stringify(data)}`);
                } catch (e) {
                  setDebugMsg(`FAIL: ${e instanceof Error ? e.message : "Network error"}`);
                }
              }}
              className="px-3 py-1 bg-zinc-800 rounded hover:bg-zinc-700 text-zinc-300"
            >
              Ping /api/health
            </button>
            <button
              onClick={async () => {
                setDebugMsg("Sending 10x10 test...");
                const c = document.createElement("canvas");
                c.width = 10; c.height = 10;
                const ctx = c.getContext("2d")!;
                ctx.fillStyle = "#888";
                ctx.fillRect(0, 0, 10, 10);
                const tiny = c.toDataURL("image/jpeg", 0.5);
                const t0 = performance.now();
                const data = await analyze(tiny);
                const ms = Math.round(performance.now() - t0);
                if (data) {
                  setDebugMsg(`OK — ${ms}ms — ${data.face_count} faces — ${data.processing_time_ms}ms server`);
                  setError(null);
                } else {
                  setDebugMsg(`POST failed after ${ms}ms — check error above`);
                }
              }}
              className="px-3 py-1 bg-zinc-800 rounded hover:bg-zinc-700 text-zinc-300"
            >
              Test 10x10 POST
            </button>
          </div>
          {debugMsg && (
            <div className={clsx(
              "p-2 rounded",
              debugMsg.startsWith("FAIL") || debugMsg.includes("failed")
                ? "bg-red-500/10 text-red-400"
                : "bg-emerald-500/10 text-emerald-400"
            )}>
              {debugMsg}
            </div>
          )}
        </div>
      )}

      {/* ── Error ── */}
      {error && (
        <div className="text-red-400 text-sm text-center bg-red-500/10 rounded-lg p-3">
          {error}
        </div>
      )}

      {/* ── Results ── */}
      {results && results.face_count > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-zinc-500 text-center">
            {results.face_count} face{results.face_count > 1 ? "s" : ""} &middot; {results.processing_time_ms.toFixed(0)}ms
          </p>
          <div className="grid gap-2">
            {results.results.map((face, i) => (
              <div
                key={i}
                className={clsx(
                  "flex items-center justify-between p-3 rounded-lg border",
                  face.gender === "Male"
                    ? "border-blue-500/30 bg-blue-500/5"
                    : "border-pink-500/30 bg-pink-500/5"
                )}
              >
                <div className="flex items-center gap-4">
                  <span className={clsx(
                    "text-lg font-bold",
                    face.gender === "Male" ? "text-blue-400" : "text-pink-400"
                  )}>
                    {face.gender}
                  </span>
                  <span className="text-zinc-300 font-medium">{face.age} yrs</span>
                </div>
                <span className="text-xs text-zinc-500">{(face.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {results && results.face_count === 0 && (
        <p className="text-zinc-500 text-sm text-center">No faces detected</p>
      )}
    </div>
  );
}
