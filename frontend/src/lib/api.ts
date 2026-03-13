import {
  API_URL,
  MAX_PAYLOAD_BYTES,
  COMPRESS_MAX_WIDTH,
  COMPRESS_QUALITY_AGGRESSIVE,
  FETCH_TIMEOUT_MS,
  COMPRESS_TIMEOUT_MS,
} from "./constants";
import type { AnalyzeResponse } from "./types";

/**
 * Compress a base64 image by resizing and re-encoding as JPEG.
 * Rejects on bad images and has a 10s timeout.
 */
export function compressImage(
  base64: string,
  maxWidth = COMPRESS_MAX_WIDTH,
  quality = 0.7,
): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error("Image compression timed out"));
    }, COMPRESS_TIMEOUT_MS);

    const img = new Image();

    img.onload = () => {
      clearTimeout(timeout);
      try {
        const scale = img.width > maxWidth ? maxWidth / img.width : 1;
        const w = Math.round(img.width * scale);
        const h = Math.round(img.height * scale);

        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          reject(new Error("Failed to get canvas context for compression"));
          return;
        }
        ctx.drawImage(img, 0, 0, w, h);
        resolve(canvas.toDataURL("image/jpeg", quality));
      } catch (e) {
        reject(e instanceof Error ? e : new Error("Compression failed"));
      }
    };

    img.onerror = () => {
      clearTimeout(timeout);
      reject(new Error("Failed to load image for compression"));
    };

    img.src = base64;
  });
}

/**
 * Send a base64 image to the analyze API.
 * Uses AbortController for timeout. Returns null on failure.
 */
export async function analyzeImage(
  base64: string,
  signal?: AbortSignal,
): Promise<{ data: AnalyzeResponse | null; error: string | null }> {
  try {
    // Compress if payload would exceed Vercel's limit
    let image = base64;
    const rawSize = new Blob([JSON.stringify({ image })]).size;
    if (rawSize > MAX_PAYLOAD_BYTES) {
      image = await compressImage(base64, COMPRESS_MAX_WIDTH, COMPRESS_QUALITY_AGGRESSIVE);
    }

    const body = JSON.stringify({ image });
    if (body.length > MAX_PAYLOAD_BYTES) {
      return {
        data: null,
        error: `Image too large (${(body.length / 1024 / 1024).toFixed(1)}MB). Try a smaller image.`,
      };
    }

    // Create a timeout abort controller, chain with external signal if provided
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), FETCH_TIMEOUT_MS);

    // If an external signal is provided, abort our controller when it fires
    const onExternalAbort = () => timeoutController.abort();
    signal?.addEventListener("abort", onExternalAbort);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        signal: timeoutController.signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        return { data: null, error: err.detail || `HTTP ${res.status}` };
      }

      const data: AnalyzeResponse = await res.json();
      return { data, error: null };
    } finally {
      clearTimeout(timeoutId);
      signal?.removeEventListener("abort", onExternalAbort);
    }
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      return { data: null, error: "Request timed out or was cancelled" };
    }
    return {
      data: null,
      error: e instanceof Error ? e.message : "API error",
    };
  }
}
