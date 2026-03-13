import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Camera } from "@/components/Camera";

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

/** Provide a minimal mock for navigator.mediaDevices.getUserMedia. */
function mockGetUserMedia() {
  const mockStream = {
    getTracks: () => [{ stop: vi.fn() }],
  } as unknown as MediaStream;

  const getUserMedia = vi.fn().mockResolvedValue(mockStream);

  Object.defineProperty(navigator, "mediaDevices", {
    value: { getUserMedia },
    writable: true,
    configurable: true,
  });

  return { getUserMedia, mockStream };
}

/** Stub HTMLMediaElement.play() which jsdom does not implement. */
function stubVideoPlay() {
  // jsdom doesn't implement play()
  Object.defineProperty(HTMLMediaElement.prototype, "play", {
    value: vi.fn().mockResolvedValue(undefined),
    writable: true,
    configurable: true,
  });
}

/** Stub canvas getContext to avoid jsdom limitations. */
function stubCanvas() {
  const fakeCtx = {
    drawImage: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    fillText: vi.fn(),
    measureText: vi.fn(() => ({ width: 50 })),
    font: "",
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
  };

  HTMLCanvasElement.prototype.getContext = vi.fn(() => fakeCtx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.toDataURL = vi.fn(() => "data:image/jpeg;base64,stub");

  return fakeCtx;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Camera component", () => {
  let getUserMediaMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    const mocks = mockGetUserMedia();
    getUserMediaMock = mocks.getUserMedia;
    stubVideoPlay();
    stubCanvas();

    // Stub fetch globally so the component doesn't make real HTTP calls.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ results: [], face_count: 0, processing_time_ms: 0 }),
      }),
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders without crashing", () => {
    const { container } = render(<Camera />);
    expect(container).toBeTruthy();
  });

  it('shows "Start Camera" button in idle state', () => {
    render(<Camera />);
    const startBtn = screen.getByText("Start Camera");
    expect(startBtn).toBeInTheDocument();
  });

  it('shows "Upload Image" button in idle state', () => {
    render(<Camera />);
    const uploadBtn = screen.getByText("Upload Image");
    expect(uploadBtn).toBeInTheDocument();
  });

  it('displays both "Start Camera" and "Upload Image" buttons simultaneously', () => {
    render(<Camera />);
    expect(screen.getByText("Start Camera")).toBeInTheDocument();
    expect(screen.getByText("Upload Image")).toBeInTheDocument();
  });

  it("attempts to access getUserMedia when Start Camera is clicked", async () => {
    render(<Camera />);

    const startBtn = screen.getByText("Start Camera");
    fireEvent.click(startBtn);

    // getUserMedia should have been called (may be async).
    // Allow microtasks to flush.
    await vi.waitFor(() => {
      expect(getUserMediaMock).toHaveBeenCalledTimes(1);
    });

    // Verify it requested video (not audio).
    const constraints = getUserMediaMock.mock.calls[0][0];
    expect(constraints.video).toBeTruthy();
    expect(constraints.audio).toBe(false);
  });

  it("shows placeholder text when no camera is active and no image uploaded", () => {
    render(<Camera />);
    expect(
      screen.getByText(/start camera or drop an image/i),
    ).toBeInTheDocument();
  });
});
