import { Suspense } from "react";
import { Camera } from "@/components/Camera";

function CameraFallback() {
  return (
    <div className="w-full max-w-2xl">
      <div className="aspect-video w-full rounded-xl bg-zinc-900 border-2 border-dashed border-zinc-800 animate-pulse flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin" />
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-4">
      <h1 className="text-3xl font-bold mb-2 tracking-tight">Lite-Vision</h1>
      <p className="text-zinc-500 text-sm mb-8">Real-time age & gender detection</p>
      <Suspense fallback={<CameraFallback />}>
        <Camera />
      </Suspense>
    </main>
  );
}
