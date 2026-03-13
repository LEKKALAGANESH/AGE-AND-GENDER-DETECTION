import clsx from "clsx";
import type { AnalyzeResponse } from "@/lib/types";
import { getAgeRange, getGenderConfidence } from "@/lib/types";

interface ResultsPanelProps {
  results: AnalyzeResponse | null;
  isStreaming?: boolean;
}

/** Color-coded badge for processing time. */
function ProcessingTimeBadge({
  timeMs,
  isStreaming,
}: {
  timeMs: number;
  isStreaming: boolean;
}) {
  const rounded = Math.round(timeMs);
  let badgeColor: string;
  let label: string;

  if (rounded < 200) {
    badgeColor = "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
    label = `${rounded}ms`;
  } else if (rounded < 500) {
    badgeColor = "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
    label = `${rounded}ms`;
  } else {
    badgeColor = "bg-red-500/20 text-red-400 border-red-500/30";
    label = `${rounded}ms`;
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className={clsx(
          "inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono border",
          badgeColor,
        )}
      >
        {label}
      </span>
      {isStreaming && (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-bold bg-red-500/20 text-red-400 border border-red-500/30">
          <span className="animate-pulse-dot inline-block w-1.5 h-1.5 rounded-full bg-red-400" />
          LIVE
        </span>
      )}
    </span>
  );
}

/** Dual-bar gender confidence visualization. */
function GenderConfidenceBars({
  genderConf,
  isMale,
}: {
  genderConf: number;
  isMale: boolean;
}) {
  const malePct = isMale ? genderConf * 100 : (1 - genderConf) * 100;
  const femalePct = isMale ? (1 - genderConf) * 100 : genderConf * 100;

  return (
    <div className="w-full max-w-[220px]">
      {/* Labels row */}
      <div className="flex justify-between text-[10px] mb-0.5">
        <span className={clsx("font-mono", isMale ? "text-blue-400 font-bold" : "text-blue-400/50")}>
          {malePct.toFixed(1)}%
        </span>
        <span className={clsx("font-mono", !isMale ? "text-pink-400 font-bold" : "text-pink-400/50")}>
          {femalePct.toFixed(1)}%
        </span>
      </div>
      {/* Bars row */}
      <div className="flex h-2 gap-0.5 rounded-full overflow-hidden">
        {/* Male bar: fills left-to-right */}
        <div className="flex-1 bg-zinc-800 rounded-l-full overflow-hidden">
          <div
            className={clsx(
              "h-full rounded-l-full transition-all duration-300",
              isMale ? "bg-blue-500" : "bg-blue-500/25",
            )}
            style={{ width: `${malePct}%` }}
          />
        </div>
        {/* Female bar: fills right-to-left */}
        <div className="flex-1 bg-zinc-800 rounded-r-full overflow-hidden flex justify-end">
          <div
            className={clsx(
              "h-full rounded-r-full transition-all duration-300",
              !isMale ? "bg-pink-500" : "bg-pink-500/25",
            )}
            style={{ width: `${femalePct}%` }}
          />
        </div>
      </div>
      {/* Bar labels */}
      <div className="flex justify-between text-[9px] mt-0.5 text-zinc-600">
        <span>Male</span>
        <span>Female</span>
      </div>
    </div>
  );
}

/** Age range visualization with a range bar and predicted-age marker. */
function AgeRangeBar({
  age,
  ageMin,
  ageMax,
  color,
}: {
  age: number;
  ageMin: number;
  ageMax: number;
  color: string;
}) {
  // The range bar spans from ageMin to ageMax, with the predicted age marked
  const range = ageMax - ageMin;
  const agePct = range > 0 ? ((age - ageMin) / range) * 100 : 50;
  const deviation = Math.round(range / 2);

  return (
    <div className="flex flex-col items-center gap-0.5 min-w-[80px]">
      {/* Predicted age */}
      <span className="text-zinc-200 font-bold text-lg leading-tight tabular-nums">
        {age}
      </span>
      {/* Range bar */}
      <div className="relative w-full h-1.5 rounded-full bg-zinc-700/60 overflow-visible">
        {/* Filled range */}
        <div
          className="absolute top-0 h-full rounded-full opacity-40"
          style={{
            left: "0%",
            width: "100%",
            background: `linear-gradient(90deg, transparent, ${color}, transparent)`,
          }}
        />
        {/* Predicted age marker dot */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full border-2 border-white shadow-lg"
          style={{
            left: `${agePct}%`,
            transform: `translate(-50%, -50%)`,
            backgroundColor: color,
            boxShadow: `0 0 6px ${color}`,
          }}
        />
      </div>
      {/* Range labels */}
      <div className="flex justify-between w-full text-[9px] text-zinc-500 tabular-nums">
        <span>{ageMin}</span>
        <span className="text-zinc-400">&plusmn;{deviation} yrs</span>
        <span>{ageMax}</span>
      </div>
    </div>
  );
}

export function ResultsPanel({ results, isStreaming = false }: ResultsPanelProps) {
  if (!results) return null;

  if (results.face_count === 0) {
    return <p className="text-zinc-500 text-sm text-center">No faces detected</p>;
  }

  return (
    <div className="space-y-2">
      {/* Summary line with processing badge */}
      <div className="flex items-center justify-center gap-2 text-xs text-zinc-500">
        <span>
          {results.face_count} face{results.face_count > 1 ? "s" : ""}
        </span>
        <span>&middot;</span>
        <ProcessingTimeBadge timeMs={results.processing_time_ms} isStreaming={isStreaming} />
      </div>

      <div className="grid gap-2">
        {results.results.map((face, i) => {
          const isMale = face.gender === "Male";
          const { min: ageMin, max: ageMax } = getAgeRange(face);
          const genderConf = getGenderConfidence(face);
          const color = isMale ? "#3b82f6" : "#ec4899";

          return (
            <div
              key={i}
              className={clsx(
                "p-3 rounded-lg border space-y-3",
                isMale
                  ? "border-blue-500/30 bg-blue-500/5"
                  : "border-pink-500/30 bg-pink-500/5",
              )}
            >
              {/* Top row: gender label + face confidence */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {/* Pulsing dot for dominant gender */}
                  <span
                    className={clsx(
                      "inline-block w-2 h-2 rounded-full",
                      isStreaming && "animate-pulse-dot",
                      isMale ? "bg-blue-400" : "bg-pink-400",
                    )}
                  />
                  <span
                    className={clsx(
                      "text-lg font-bold leading-tight",
                      isMale ? "text-blue-400" : "text-pink-400",
                    )}
                  >
                    {face.gender}
                  </span>
                </div>
                <span className="text-xs text-zinc-500">
                  face {(face.confidence * 100).toFixed(1)}%
                </span>
              </div>

              {/* Gender dual-bar chart */}
              <GenderConfidenceBars genderConf={genderConf} isMale={isMale} />

              {/* Age range visualization */}
              <AgeRangeBar age={face.age} ageMin={ageMin} ageMax={ageMax} color={color} />

              {/* Emotion indicator */}
              {face.emotion && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-400">Expression</span>
                  <span className="text-zinc-300 capitalize">
                    {face.emotion}
                    {face.emotion_confidence != null && (
                      <span className="text-zinc-500 ml-1">
                        {(face.emotion_confidence * 100).toFixed(0)}%
                      </span>
                    )}
                  </span>
                </div>
              )}

              {/* Accuracy warning for expressive faces */}
              {face.emotion &&
                ["happiness", "surprise", "contempt"].includes(face.emotion) &&
                (face.emotion_confidence ?? 0) > 0.5 && (
                  <div className="text-[10px] text-amber-400/80 bg-amber-500/10 rounded px-2 py-1 text-center">
                    Expression may affect gender accuracy
                  </div>
                )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
