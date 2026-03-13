"use client";

import { useRef, useCallback, useState } from "react";
import type { FaceResult } from "@/lib/types";

/** Internal tracked-face representation with smoothing state. */
interface SmoothedFace {
  id: number;
  age: number;
  age_min: number;
  age_max: number;
  gender: string;
  gender_confidence: number;
  confidence: number;
  region: [number, number, number, number];
  framesSeen: number;
  framesMissed: number;
  genderVotes: string[];
}

/** EMA blending factor: 30% new frame, 70% history. */
const ALPHA = 0.3;

/** Remove a tracked face after this many consecutive missed frames. */
const STALE_THRESHOLD = 10;

/** Minimum IoU to consider two bounding boxes the same face. */
const IOU_THRESHOLD = 0.3;

/** Number of gender votes to keep for majority voting. */
const GENDER_VOTE_BUFFER = 8;

/** Minimum votes needed to flip gender (6 out of 8 = 75% supermajority). */
const GENDER_FLIP_THRESHOLD = 6;

/**
 * Compute Intersection over Union for two [x, y, w, h] bounding boxes
 * (works with both normalized 0-1 and pixel coordinates).
 */
function computeIoU(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  const [ax, ay, aw, ah] = a;
  const [bx, by, bw, bh] = b;

  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = aw * ah + bw * bh - intersection;

  return union > 0 ? intersection / union : 0;
}

let nextFaceId = 1;

/**
 * Hook that provides temporal smoothing for live face-detection streams.
 *
 * When streaming at ~10 fps, raw per-frame predictions can jitter
 * (e.g., age jumping between 25 and 35). This hook tracks faces across
 * frames using IoU matching and applies an exponential moving average
 * so displayed values change gradually.
 */
export function useTemporalSmoothing() {
  const trackedRef = useRef<SmoothedFace[]>([]);
  const [isSmoothing, setIsSmoothing] = useState(false);

  /**
   * Match incoming detections to existing tracked faces via IoU,
   * apply EMA smoothing, and return the smoothed FaceResult array.
   */
  const smoothResults = useCallback((results: FaceResult[]): FaceResult[] => {
    const tracked = trackedRef.current;
    const updatedTracked: SmoothedFace[] = [];

    // --- Step 1: Greedy IoU matching (new detections -> tracked faces) ---
    // Build all (detection, tracked) IoU pairs, sort descending, greedily assign.
    const pairs: { di: number; ti: number; iou: number }[] = [];
    for (let di = 0; di < results.length; di++) {
      for (let ti = 0; ti < tracked.length; ti++) {
        const iou = computeIoU(results[di].region, tracked[ti].region);
        if (iou > IOU_THRESHOLD) {
          pairs.push({ di, ti, iou });
        }
      }
    }
    pairs.sort((a, b) => b.iou - a.iou);

    const matchedDetections = new Set<number>();
    const matchedTracked = new Set<number>();
    const matches: { di: number; ti: number }[] = [];

    for (const { di, ti } of pairs) {
      if (matchedDetections.has(di) || matchedTracked.has(ti)) continue;
      matchedDetections.add(di);
      matchedTracked.add(ti);
      matches.push({ di, ti });
    }

    // --- Step 2: Update matched tracked faces with EMA ---
    for (const { di, ti } of matches) {
      const det = results[di];
      const prev = tracked[ti];

      const genderConf = det.gender_confidence ?? det.confidence;
      const ageMin = det.age_min ?? det.age;
      const ageMax = det.age_max ?? det.age;

      const smoothedAge = Math.round(ALPHA * det.age + (1 - ALPHA) * prev.age);
      const smoothedAgeMin = Math.round(ALPHA * ageMin + (1 - ALPHA) * prev.age_min);
      const smoothedAgeMax = Math.round(ALPHA * ageMax + (1 - ALPHA) * prev.age_max);
      const smoothedGenderConf = ALPHA * genderConf + (1 - ALPHA) * prev.gender_confidence;
      const smoothedConfidence = ALPHA * det.confidence + (1 - ALPHA) * prev.confidence;

      // Smooth region too to prevent box jitter
      const smoothedRegion: [number, number, number, number] = [
        ALPHA * det.region[0] + (1 - ALPHA) * prev.region[0],
        ALPHA * det.region[1] + (1 - ALPHA) * prev.region[1],
        ALPHA * det.region[2] + (1 - ALPHA) * prev.region[2],
        ALPHA * det.region[3] + (1 - ALPHA) * prev.region[3],
      ];

      // Gender majority voting: require 6/8 frames to agree before flipping
      const votes = [...prev.genderVotes, det.gender].slice(-GENDER_VOTE_BUFFER);
      const maleVotes = votes.filter(v => v === "Male").length;
      const femaleVotes = votes.length - maleVotes;

      let gender: string;
      if (maleVotes >= GENDER_FLIP_THRESHOLD) {
        gender = "Male";
      } else if (femaleVotes >= GENDER_FLIP_THRESHOLD) {
        gender = "Female";
      } else {
        // Not enough votes to flip — keep previous gender
        gender = prev.gender;
      }

      updatedTracked.push({
        id: prev.id,
        age: smoothedAge,
        age_min: smoothedAgeMin,
        age_max: smoothedAgeMax,
        gender,
        gender_confidence: smoothedGenderConf,
        confidence: smoothedConfidence,
        region: smoothedRegion,
        framesSeen: prev.framesSeen + 1,
        framesMissed: 0,
        genderVotes: votes,
      });
    }

    // --- Step 3: Create new tracked faces for unmatched detections ---
    for (let di = 0; di < results.length; di++) {
      if (matchedDetections.has(di)) continue;
      const det = results[di];
      updatedTracked.push({
        id: nextFaceId++,
        age: det.age,
        age_min: det.age_min ?? det.age,
        age_max: det.age_max ?? det.age,
        gender: det.gender,
        gender_confidence: det.gender_confidence ?? det.confidence,
        confidence: det.confidence,
        region: [...det.region],
        framesSeen: 1,
        framesMissed: 0,
        genderVotes: [det.gender],
      });
    }

    // --- Step 4: Carry forward unmatched tracked faces (increment missed count) ---
    for (let ti = 0; ti < tracked.length; ti++) {
      if (matchedTracked.has(ti)) continue;
      const t = tracked[ti];
      const missed = t.framesMissed + 1;
      if (missed < STALE_THRESHOLD) {
        updatedTracked.push({ ...t, framesMissed: missed });
      }
      // else: face is stale, drop it
    }

    trackedRef.current = updatedTracked;
    setIsSmoothing(updatedTracked.length > 0);

    // --- Step 5: Return only faces seen this frame (not stale carry-forwards) ---
    // We return faces that were either matched or newly created this frame.
    const activeCount = matchedDetections.size + (results.length - matchedDetections.size);
    const activeFaces = updatedTracked.slice(0, activeCount);

    return activeFaces.map((f) => ({
      age: f.age,
      age_min: f.age_min,
      age_max: f.age_max,
      gender: f.gender,
      gender_confidence: f.gender_confidence,
      confidence: f.confidence,
      region: f.region,
    }));
  }, []);

  /** Clear all tracking history. Call when stopping the stream. */
  const reset = useCallback(() => {
    trackedRef.current = [];
    setIsSmoothing(false);
  }, []);

  return { smoothResults, reset, isSmoothing };
}
