export default function Loading() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-2xl space-y-4">
        {/* Title skeleton */}
        <div className="flex flex-col items-center gap-2 mb-8">
          <div className="h-9 w-48 bg-zinc-800 rounded-lg animate-pulse" />
          <div className="h-4 w-64 bg-zinc-800/60 rounded animate-pulse" />
        </div>

        {/* Camera view area skeleton */}
        <div className="aspect-video w-full rounded-xl bg-zinc-900 border-2 border-dashed border-zinc-800 animate-pulse flex items-center justify-center">
          <div className="w-12 h-12 rounded-full bg-zinc-800" />
        </div>

        {/* Control buttons skeleton */}
        <div className="flex items-center justify-center gap-3">
          <div className="h-10 w-36 bg-zinc-800 rounded-lg animate-pulse" />
          <div className="h-10 w-36 bg-zinc-800 rounded-lg animate-pulse" />
        </div>
      </div>
    </div>
  );
}
