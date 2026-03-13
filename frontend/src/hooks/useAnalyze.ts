"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { analyzeImage } from "@/lib/api";
import type { AnalyzeResponse } from "@/lib/types";

export function useAnalyze() {
  const [results, setResults] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const analyze = useCallback(async (base64: string): Promise<AnalyzeResponse | null> => {
    // Cancel any previous in-flight request
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    const { data, error: apiError } = await analyzeImage(base64, controller.signal);

    // Don't update state if this request was aborted (superseded by another)
    if (controller.signal.aborted) {
      return null;
    }

    if (apiError) {
      setError(apiError);
      setResults(null);
    } else if (data) {
      setError(null);
      setResults(data);
    }

    setIsLoading(false);
    return data;
  }, []);

  const clearError = useCallback(() => setError(null), []);

  // Cancel in-flight requests on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  return { results, setResults, error, isLoading, analyze, clearError };
}
