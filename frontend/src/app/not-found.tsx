import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <div className="max-w-md w-full text-center space-y-6">
        <h1 className="text-7xl font-bold tracking-tighter text-zinc-700">
          404
        </h1>
        <h2 className="text-xl font-semibold tracking-tight">
          Page not found
        </h2>
        <p className="text-zinc-400 text-sm">
          The page you are looking for does not exist or has been moved.
        </p>
        <Link
          href="/"
          className="inline-block px-5 py-2.5 bg-white text-black rounded-lg font-medium text-sm hover:bg-zinc-200 transition-colors"
        >
          Back to home
        </Link>
      </div>
    </div>
  );
}
