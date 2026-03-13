"use client";

import { useCallback, type ReactNode, type KeyboardEvent, type DragEvent } from "react";
import clsx from "clsx";

interface DropZoneProps {
  children: ReactNode;
  dragOver: boolean;
  dropProps: {
    onDragOver: (e: DragEvent) => void;
    onDragLeave: () => void;
    onDrop: (e: DragEvent) => void;
  };
  onTriggerFileSelect: () => void;
}

export function DropZone({ children, dragOver, dropProps, onTriggerFileSelect }: DropZoneProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onTriggerFileSelect();
      }
    },
    [onTriggerFileSelect],
  );

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Drop zone for image upload. Press Enter or Space to select a file."
      className={clsx(
        "relative aspect-video rounded-xl overflow-hidden border-2 border-dashed transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500",
        dragOver ? "border-blue-500 bg-blue-500/5" : "border-zinc-800 bg-zinc-900",
      )}
      onKeyDown={handleKeyDown}
      {...dropProps}
    >
      {children}
      {dragOver && (
        <div className="absolute inset-0 flex items-center justify-center bg-blue-500/10 pointer-events-none">
          <p className="text-blue-400 font-medium text-sm">Drop image here</p>
        </div>
      )}
    </div>
  );
}
