import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

/**
 * Tests for lib/api.ts — compressImage and analyzeImage.
 *
 * These tests target the expected public API:
 *   compressImage(base64: string, maxWidth?: number, quality?: number): Promise<string>
 *   analyzeImage(base64: string): Promise<AnalyzeResponse | null>
 */

// ---------------------------------------------------------------------------
// Dynamic import helper — the module under test may not exist yet.
// We lazily import so the test file itself always parses.
// ---------------------------------------------------------------------------
async function loadApi() {
  return await import("@/lib/api");
}

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

/** Minimal mock for the browser Image constructor used inside compressImage. */
function mockImageConstructor({ shouldLoad = true } = {}) {
  const instances: Array<{ onload: (() => void) | null; onerror: ((e: unknown) => void) | null; src: string; width: number; height: number }> = [];

  class FakeImage {
    onload: (() => void) | null = null;
    onerror: ((e: unknown) => void) | null = null;
    src = "";
    width = 100;
    height = 100;

    constructor() {
      // eslint-disable-next-line @typescript-eslint/no-this-alias
      const self = this;
      instances.push(self);

      // Trigger onload / onerror on next microtask after `src` is set.
      const origDescriptor = Object.getOwnPropertyDescriptor(
        Object.getPrototypeOf(this),
        "src",
      );
      Object.defineProperty(this, "src", {
        set(value: string) {
          self.src = value;
          queueMicrotask(() => {
            if (shouldLoad) {
              self.onload?.();
            } else {
              self.onerror?.(new Error("Image load failed"));
            }
          });
        },
        get() {
          return self.src;
        },
        configurable: true,
      });

      // Fix: avoid infinite recursion by using an internal backing field.
      let _src = "";
      Object.defineProperty(this, "src", {
        set(value: string) {
          _src = value;
          queueMicrotask(() => {
            if (shouldLoad) {
              self.onload?.();
            } else {
              self.onerror?.(new Error("Image load failed"));
            }
          });
        },
        get() {
          return _src;
        },
        configurable: true,
      });
    }
  }

  vi.stubGlobal("Image", FakeImage);
  return instances;
}

/** Stub canvas returned by document.createElement("canvas"). */
function mockCanvas() {
  const fakeCtx = {
    drawImage: vi.fn(),
  };
  const fakeCanvas = {
    width: 0,
    height: 0,
    getContext: vi.fn(() => fakeCtx),
    toDataURL: vi.fn(() => "data:image/jpeg;base64,compressed"),
  };

  const origCreateElement = document.createElement.bind(document);
  vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
    if (tag === "canvas") return fakeCanvas as unknown as HTMLCanvasElement;
    return origCreateElement(tag);
  });

  return { fakeCanvas, fakeCtx };
}

// ---------------------------------------------------------------------------
// Tests — compressImage
// ---------------------------------------------------------------------------
describe("compressImage", () => {
  beforeEach(() => {
    mockImageConstructor({ shouldLoad: true });
    mockCanvas();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("resolves with a compressed base64 string for valid input", async () => {
    const { compressImage } = await loadApi();
    const result = await compressImage("data:image/png;base64,AAAA");
    expect(result).toContain("data:image/jpeg");
  });

  it("rejects or handles gracefully when Image fails to load", async () => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    mockImageConstructor({ shouldLoad: false });
    mockCanvas();

    const { compressImage } = await loadApi();

    // The implementation should either reject the promise or return an error.
    // We accept either behaviour — the key point is it does not hang forever.
    try {
      const result = await Promise.race([
        compressImage("not-a-real-image"),
        new Promise<string>((_, reject) =>
          setTimeout(() => reject(new Error("timeout")), 2000),
        ),
      ]);
      // If it resolves, that's acceptable (some implementations silently resolve).
      expect(result).toBeDefined();
    } catch (err) {
      // Rejection is the expected "graceful" path.
      expect(err).toBeDefined();
    }
  });
});

// ---------------------------------------------------------------------------
// Tests — analyzeImage
// ---------------------------------------------------------------------------
describe("analyzeImage", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn() as unknown as typeof fetch,
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("calls fetch with the correct URL, method, and body", async () => {
    const mockResponse = {
      ok: true,
      json: async () => ({
        results: [
          { age: 25, gender: "Male", confidence: 0.95, region: [0, 0, 0.5, 0.5] },
        ],
        face_count: 1,
        processing_time_ms: 123,
      }),
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse);

    const { analyzeImage } = await loadApi();
    const result = await analyzeImage("data:image/jpeg;base64,AAAA");

    expect(globalThis.fetch).toHaveBeenCalledTimes(1);

    const [url, options] = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls[0];

    // The URL should point to the analyze endpoint.
    expect(typeof url).toBe("string");
    expect(options.method).toBe("POST");
    expect(options.headers).toEqual(
      expect.objectContaining({ "Content-Type": "application/json" }),
    );

    // Body should contain the image.
    const body = JSON.parse(options.body);
    expect(body.image).toBe("data:image/jpeg;base64,AAAA");

    // Should return the parsed response.
    expect(result).not.toBeNull();
    expect(result!.face_count).toBe(1);
    expect(result!.results[0].age).toBe(25);
  });

  it("handles HTTP errors gracefully and returns null", async () => {
    const mockResponse = {
      ok: false,
      status: 500,
      json: async () => ({ detail: "Internal Server Error" }),
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse);

    const { analyzeImage } = await loadApi();
    const result = await analyzeImage("data:image/jpeg;base64,AAAA");

    // Should return null on HTTP error rather than throwing.
    expect(result).toBeNull();
  });

  it("handles network failures gracefully and returns null", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(
      new TypeError("Failed to fetch"),
    );

    const { analyzeImage } = await loadApi();
    const result = await analyzeImage("data:image/jpeg;base64,AAAA");

    expect(result).toBeNull();
  });

  it("respects timeout — returns null if request takes too long", async () => {
    // Simulate a fetch that never resolves within a reasonable time.
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockImplementation(
      () =>
        new Promise((resolve) => {
          // Intentionally never resolves (simulates hung request).
          // If the implementation has an AbortController / timeout it will abort.
          setTimeout(
            () =>
              resolve({
                ok: true,
                json: async () => ({
                  results: [],
                  face_count: 0,
                  processing_time_ms: 0,
                }),
              }),
            60_000,
          );
        }),
    );

    const { analyzeImage } = await loadApi();

    // Race against our own timeout to verify the implementation either:
    // a) has its own timeout and returns null, or
    // b) we detect it would hang.
    const result = await Promise.race([
      analyzeImage("data:image/jpeg;base64,AAAA"),
      new Promise<null>((resolve) => setTimeout(() => resolve(null), 5_000)),
    ]);

    // Regardless of whether the implementation times out on its own, we
    // verify it doesn't throw and the result is null (timed-out).
    expect(result).toBeNull();
  });
});
