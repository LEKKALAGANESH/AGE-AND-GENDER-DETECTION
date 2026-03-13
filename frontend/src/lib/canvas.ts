import type { AnalyzeResponse } from "./types";
import { getAgeRange, getGenderConfidence } from "./types";

/** Draw corner brackets at the four corners of a bounding box (sci-fi style). */
function drawCornerBrackets(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
  bracketLen: number,
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 4;

  // Top-left
  ctx.beginPath();
  ctx.moveTo(x, y + bracketLen);
  ctx.lineTo(x, y);
  ctx.lineTo(x + bracketLen, y);
  ctx.stroke();

  // Top-right
  ctx.beginPath();
  ctx.moveTo(x + w - bracketLen, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + bracketLen);
  ctx.stroke();

  // Bottom-left
  ctx.beginPath();
  ctx.moveTo(x, y + h - bracketLen);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x + bracketLen, y + h);
  ctx.stroke();

  // Bottom-right
  ctx.beginPath();
  ctx.moveTo(x + w - bracketLen, y + h);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w, y + h - bracketLen);
  ctx.stroke();

  ctx.shadowBlur = 0;
}

/**
 * Draw a scanning laser line that bounces up and down inside the bounding box.
 * Uses performance.now() for smooth time-based animation (~2s per cycle).
 * The line has a glow effect achieved by drawing multiple passes with
 * decreasing opacity and increasing width.
 */
function drawScanLine(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
): void {
  // ~2 second full up-down cycle using a triangle wave
  const period = 2000; // ms
  const t = (performance.now() % period) / period; // 0..1
  // Triangle wave: goes 0→1→0 over one period for smooth bounce
  const bounce = t < 0.5 ? t * 2 : 2 - t * 2;
  const margin = 4;
  const scanY = y + margin + bounce * (h - 2 * margin);

  // Build horizontal gradient that fades at edges
  const gradient = ctx.createLinearGradient(x, scanY, x + w, scanY);
  gradient.addColorStop(0, "transparent");
  gradient.addColorStop(0.15, color);
  gradient.addColorStop(0.5, color);
  gradient.addColorStop(0.85, color);
  gradient.addColorStop(1, "transparent");

  // Pass 1: wide soft glow
  ctx.save();
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 8;
  ctx.globalAlpha = 0.1;
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.moveTo(x + 2, scanY);
  ctx.lineTo(x + w - 2, scanY);
  ctx.stroke();

  // Pass 2: medium glow
  ctx.lineWidth = 4;
  ctx.globalAlpha = 0.25;
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(x + 2, scanY);
  ctx.lineTo(x + w - 2, scanY);
  ctx.stroke();

  // Pass 3: sharp core line
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.8;
  ctx.shadowBlur = 4;
  ctx.beginPath();
  ctx.moveTo(x + 2, scanY);
  ctx.lineTo(x + w - 2, scanY);
  ctx.stroke();

  ctx.restore();
}

/**
 * Draw a vertical dual bar chart (Male/Female confidence) to the right of the bounding box.
 * Male bar is blue, Female bar is pink. Heights represent probabilities.
 */
function drawConfidenceBars(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  gender: string,
  genderConf: number,
): void {
  const isMale = gender === "Male";
  const maleProb = isMale ? genderConf : 1 - genderConf;
  const femaleProb = isMale ? 1 - genderConf : genderConf;

  const chartWidth = 30;
  const chartX = x + w + 4;
  const chartY = y;
  const chartH = h;
  const barWidth = 10;
  const barGap = 4;
  const labelHeight = 12;
  const barAreaH = chartH - labelHeight;

  // Background panel
  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
  ctx.beginPath();
  ctx.roundRect(chartX, chartY, chartWidth, chartH, 3);
  ctx.fill();

  // Male bar (left)
  const mBarX = chartX + (chartWidth / 2 - barGap / 2 - barWidth);
  const mBarH = Math.max(2, barAreaH * maleProb);
  const mBarY = chartY + barAreaH - mBarH;
  ctx.fillStyle = "#3b82f6";
  ctx.globalAlpha = 0.85;
  ctx.fillRect(mBarX, mBarY, barWidth, mBarH);

  // Female bar (right)
  const fBarX = chartX + chartWidth / 2 + barGap / 2;
  const fBarH = Math.max(2, barAreaH * femaleProb);
  const fBarY = chartY + barAreaH - fBarH;
  ctx.fillStyle = "#ec4899";
  ctx.fillRect(fBarX, fBarY, barWidth, fBarH);

  // Labels below bars
  ctx.globalAlpha = 1;
  ctx.font = "bold 8px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillStyle = "#3b82f6";
  ctx.fillText("M", mBarX + barWidth / 2, chartY + chartH - 2);
  ctx.fillStyle = "#ec4899";
  ctx.fillText("F", fBarX + barWidth / 2, chartY + chartH - 2);

  ctx.restore();
}

/**
 * Draw a live timestamp in the bottom-left corner of the overlay.
 * Shows HH:MM:SS.mmm in a semi-transparent background pill.
 * Only displayed during active streaming to prove the feed is live.
 */
function drawTimestamp(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
): void {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  const ms = String(now.getMilliseconds()).padStart(3, "0");
  const timestamp = `${hh}:${mm}:${ss}.${ms}`;

  ctx.save();
  ctx.font = "10px 'Courier New', monospace";
  const tw = ctx.measureText(timestamp).width;
  const pillW = tw + 12;
  const pillH = 18;
  const pillX = 6;
  const pillY = height - pillH - 6;

  // Background pill
  ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
  ctx.beginPath();
  ctx.roundRect(pillX, pillY, pillW, pillH, 4);
  ctx.fill();

  // Text
  ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
  ctx.fillText(timestamp, pillX + 6, pillY + 13);
  ctx.restore();
}

/**
 * Draw the processing time badge in the top-right corner.
 * Color-coded: green (<200ms), yellow (200-500ms), red (>500ms).
 */
function drawProcessingBadge(
  ctx: CanvasRenderingContext2D,
  width: number,
  processingTimeMs: number,
): void {
  const label = `\u26A1 ${processingTimeMs.toFixed(0)}ms`;
  let badgeColor: string;
  if (processingTimeMs < 200) {
    badgeColor = "#10b981"; // green
  } else if (processingTimeMs < 500) {
    badgeColor = "#eab308"; // yellow
  } else {
    badgeColor = "#ef4444"; // red
  }

  ctx.save();
  ctx.font = "bold 11px 'Courier New', monospace";
  const tw = ctx.measureText(label).width;
  const pillW = tw + 14;
  const pillH = 20;
  const pillX = width - pillW - 6;
  const pillY = 6;

  // Background pill
  ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
  ctx.beginPath();
  ctx.roundRect(pillX, pillY, pillW, pillH, 4);
  ctx.fill();

  // Text in badge color
  ctx.fillStyle = badgeColor;
  ctx.fillText(label, pillX + 7, pillY + 14);
  ctx.restore();
}

/** Draw a small emotion label below the bounding box. */
function drawEmotionIndicator(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  emotion: string,
  emotionConf: number,
): void {
  const label = `${emotion} ${(emotionConf * 100).toFixed(0)}%`;
  ctx.save();
  ctx.font = "10px system-ui, sans-serif";
  const tw = ctx.measureText(label).width;
  const pillW = tw + 10;
  const pillH = 16;
  const pillX = x + (w - pillW) / 2;
  const pillY = y + h + 4;

  // Background pill
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.beginPath();
  ctx.roundRect(pillX, pillY, pillW, pillH, 3);
  ctx.fill();

  // Text
  ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
  ctx.fillText(label, pillX + 5, pillY + 12);
  ctx.restore();
}

/**
 * Draw a color-coded focus ring around the bounding box indicating prediction reliability.
 * Green = neutral expression + high confidence (reliable)
 * Yellow = expressive face (may affect accuracy)
 * Red = low confidence (unreliable)
 */
function drawFocusRing(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  genderConf: number,
  emotion?: string,
  emotionConf?: number,
): void {
  let ringColor: string;
  const EXPRESSIVE = new Set(["happiness", "surprise", "contempt"]);

  if (genderConf < 0.6) {
    ringColor = "rgba(239, 68, 68, 0.5)"; // red — low confidence
  } else if (emotion && EXPRESSIVE.has(emotion) && (emotionConf ?? 0) > 0.5) {
    ringColor = "rgba(234, 179, 8, 0.4)"; // yellow — expressive
  } else {
    ringColor = "rgba(16, 185, 129, 0.35)"; // green — reliable
  }

  ctx.save();
  ctx.strokeStyle = ringColor;
  ctx.lineWidth = 3;
  ctx.setLineDash([6, 3]);
  ctx.strokeRect(x - 2, y - 2, w + 4, h + 4);
  ctx.setLineDash([]);
  ctx.restore();
}

/** Draw bounding boxes, labels, scanning effects, confidence bars, and HUD on the overlay canvas. */
export function drawOverlay(
  overlayCanvas: HTMLCanvasElement,
  data: AnalyzeResponse,
  width: number,
  height: number,
  isStreaming?: boolean,
): void {
  const ctx = overlayCanvas.getContext("2d");
  if (!ctx) return;

  overlayCanvas.width = width;
  overlayCanvas.height = height;
  ctx.clearRect(0, 0, width, height);

  data.results.forEach((face) => {
    const [rx, ry, rw, rh] = face.region;

    // Support both normalized (0-1) and pixel coordinates
    const isNormalized = rx <= 1 && ry <= 1 && rw <= 1 && rh <= 1;
    const x = isNormalized ? rx * width : rx;
    const y = isNormalized ? ry * height : ry;
    const w = isNormalized ? rw * width : rw;
    const h = isNormalized ? rh * height : rh;

    const isMale = face.gender === "Male";
    const color = isMale ? "#3b82f6" : "#ec4899";
    const genderIcon = isMale ? "\u2642" : "\u2640";
    const { min: ageMin, max: ageMax } = getAgeRange(face);
    const genderConf = getGenderConfidence(face);

    // ── Corner brackets (sci-fi style) ──
    const bracketLen = Math.min(w, h, 40) * 0.35;
    drawCornerBrackets(ctx, x, y, w, h, color, bracketLen);

    // ── Bounding box (slightly transparent so brackets stand out) ──
    ctx.strokeStyle = `${color}88`;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);

    // ── Scanning laser line (only during streaming) ──
    if (isStreaming) {
      drawScanLine(ctx, x, y, w, h, color);
    }

    // ── Age deviation label: "♂ Male, 24 ±3 yrs" ──
    const deviation = Math.ceil((ageMax - ageMin) / 2);
    const label = `${genderIcon} ${face.gender}, ${face.age} \u00B1${deviation} yrs`;
    ctx.font = "bold 12px system-ui, sans-serif";
    const tw = ctx.measureText(label).width;

    // Label background
    ctx.fillStyle = `${color}cc`;
    ctx.fillRect(x, y - 22, tw + 12, 22);

    // Label text
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 6, y - 6);

    // ── Dual confidence bar chart (right of bounding box) ──
    drawConfidenceBars(ctx, x, y, w, h, face.gender, genderConf);

    // ── Focus ring (prediction reliability indicator) ──
    drawFocusRing(ctx, x, y, w, h, genderConf, face.emotion, face.emotion_confidence);

    // ── Emotion indicator (below bounding box) ──
    if (face.emotion && face.emotion_confidence != null) {
      drawEmotionIndicator(ctx, x, y, w, h, face.emotion, face.emotion_confidence);
    }
  });

  // ── HUD overlays (only during streaming with results) ──
  if (isStreaming && data.results.length > 0) {
    drawTimestamp(ctx, width, height);
    drawProcessingBadge(ctx, width, data.processing_time_ms);
  }
}
