"use client";

import { useState, useCallback } from "react";
import clsx from "clsx";
import { API_URL, HEALTH_URL } from "@/lib/constants";
import { analyzeImage } from "@/lib/api";

export function DebugPanel() {
  const [debugMsg, setDebugMsg] = useState<string | null>(null);

  const handlePing = useCallback(async () => {
    setDebugMsg("Pinging...");
    try {
      const t0 = performance.now();
      const res = await fetch(HEALTH_URL);
      const ms = Math.round(performance.now() - t0);
      const data = await res.json();
      setDebugMsg(`${res.status} OK \u2014 ${ms}ms \u2014 ${JSON.stringify(data)}`);
    } catch (e) {
      setDebugMsg(`FAIL: ${e instanceof Error ? e.message : "Network error"}`);
    }
  }, []);

  const handleTestPost = useCallback(async () => {
    setDebugMsg("Sending 10x10 test...");
    const c = document.createElement("canvas");
    c.width = 10;
    c.height = 10;
    const ctx = c.getContext("2d");
    if (!ctx) {
      setDebugMsg("FAIL: could not get canvas context");
      return;
    }
    ctx.fillStyle = "#888";
    ctx.fillRect(0, 0, 10, 10);
    const tiny = c.toDataURL("image/jpeg", 0.5);
    const t0 = performance.now();
    const { data, error } = await analyzeImage(tiny);
    const ms = Math.round(performance.now() - t0);
    if (data) {
      setDebugMsg(
        `OK \u2014 ${ms}ms \u2014 ${data.face_count} faces \u2014 ${data.processing_time_ms}ms server`,
      );
    } else {
      setDebugMsg(`POST failed after ${ms}ms \u2014 ${error || "unknown error"}`);
    }
  }, []);

  return (
    <div className="space-y-2 p-3 rounded-lg bg-zinc-900 border border-zinc-800 text-xs font-mono">
      <div className="flex items-center gap-2">
        <span className="text-zinc-500">API:</span>
        <span className="text-zinc-300">{API_URL}</span>
      </div>
      <div className="flex gap-2">
        <button
          onClick={handlePing}
          aria-label="Ping health endpoint"
          className="px-3 py-1 bg-zinc-800 rounded hover:bg-zinc-700 text-zinc-300"
        >
          Ping /api/health
        </button>
        <button
          onClick={handleTestPost}
          aria-label="Test with 10x10 image"
          className="px-3 py-1 bg-zinc-800 rounded hover:bg-zinc-700 text-zinc-300"
        >
          Test 10x10 POST
        </button>
      </div>
      {debugMsg && (
        <div
          className={clsx(
            "p-2 rounded",
            debugMsg.startsWith("FAIL") || debugMsg.includes("failed")
              ? "bg-red-500/10 text-red-400"
              : "bg-emerald-500/10 text-emerald-400",
          )}
        >
          {debugMsg}
        </div>
      )}
    </div>
  );
}
