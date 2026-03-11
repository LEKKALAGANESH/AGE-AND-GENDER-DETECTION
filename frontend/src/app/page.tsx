import { Camera } from "@/components/Camera";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-4">
      <h1 className="text-3xl font-bold mb-2 tracking-tight">Lite-Vision</h1>
      <p className="text-zinc-500 text-sm mb-8">Real-time age & gender detection</p>
      <Camera />
    </main>
  );
}
