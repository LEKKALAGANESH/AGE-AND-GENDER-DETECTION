"""
Lite-Vision Load Testing Script
================================
Standalone load tester for the /api/health and /api/analyze endpoints.

Usage:
    python load_test.py
    python load_test.py --workers 10 --duration 60
    python load_test.py --base-url http://my-server:8000 --rps 50

Dependencies: stdlib only (no pip install needed).
"""

import argparse
import base64
import json
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configurable defaults (overridden by CLI args)
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000"
CONCURRENT_WORKERS = 5
REQUESTS_PER_WORKER = 20
TOTAL_DURATION_SECONDS = 30

# ---------------------------------------------------------------------------
# Minimal valid JPEG (1x1 red pixel)
# ---------------------------------------------------------------------------

def _build_minimal_jpeg() -> bytes:
    """Return the raw bytes of a minimal valid JPEG image (1x1 pixel).

    This is a hand-crafted binary blob so we avoid any dependency on PIL/Pillow.
    The image is a 1x1 red pixel encoded as a baseline JPEG.
    """
    # Minimal valid JPEG: 1x1 pixel, red (#FF0000), baseline DCT, YCbCr.
    # Generated once from Pillow and captured as a constant.
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.\x27 \",.+\x1c\x1c(7),01444\x1f\x27"
        b"9=82<.342\xff\xdb\x00C\x01\t\t\t\x0c\x0b\x0c\x18\r\r\x182!\x1c"
        b"!22222222222222222222222222222222222222222222222222\xff\xc0\x00\x11"
        b"\x08\x00\x01\x00\x01\x03\x01\"\x00\x02\x11\x01\x03\x11\x01\xff"
        b"\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n"
        b"\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05"
        b"\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A"
        b"\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0"
        b"$3br\x82\t\n\x16\x17\x18\x19\x1a%&\x27()*456789:CDEFGHIJSTUVWXY"
        b"Zcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94"
        b"\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa"
        b"\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7"
        b"\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3"
        b"\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8"
        b"\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01"
        b"\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06"
        b"\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04"
        b"\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04"
        b"\x05!1\x06\x12AQ\x07aq\x13\"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#"
        b"3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a&\x27()*56789"
        b":CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88"
        b"\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5"
        b"\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2"
        b"\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8"
        b"\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5"
        b"\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03"
        b"\x11\x00?\x00\xfb\xd2\x8a(\x03\xff\xd9"
    )


# Pre-build the base64-encoded test image payload once
_JPEG_BYTES = _build_minimal_jpeg()
_IMAGE_B64 = base64.b64encode(_JPEG_BYTES).decode("ascii")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    endpoint: str
    status_code: int
    latency_ms: float
    success: bool
    error: str = ""


@dataclass
class EndpointStats:
    name: str
    total: int = 0
    success: int = 0
    failed: int = 0
    latencies: list = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.latencies) if self.latencies else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def p50(self) -> float:
        return _percentile(self.latencies, 50) if self.latencies else 0.0

    @property
    def p95(self) -> float:
        return _percentile(self.latencies, 95) if self.latencies else 0.0

    @property
    def p99(self) -> float:
        return _percentile(self.latencies, 99) if self.latencies else 0.0


def _percentile(data: list[float], pct: float) -> float:
    """Compute the given percentile from a sorted copy of *data*."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: float = 30.0) -> RequestResult:
    """Send a POST request with a JSON body and return a RequestResult."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()  # consume body
            latency = (time.perf_counter() - t0) * 1000
            return RequestResult(
                endpoint=url, status_code=resp.status,
                latency_ms=latency, success=(200 <= resp.status < 300),
            )
    except urllib.error.HTTPError as exc:
        latency = (time.perf_counter() - t0) * 1000
        return RequestResult(
            endpoint=url, status_code=exc.code,
            latency_ms=latency, success=False, error=str(exc),
        )
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return RequestResult(
            endpoint=url, status_code=0,
            latency_ms=latency, success=False, error=str(exc),
        )


def _get(url: str, timeout: float = 30.0) -> RequestResult:
    """Send a GET request and return a RequestResult."""
    req = urllib.request.Request(url, method="GET")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            latency = (time.perf_counter() - t0) * 1000
            return RequestResult(
                endpoint=url, status_code=resp.status,
                latency_ms=latency, success=(200 <= resp.status < 300),
            )
    except urllib.error.HTTPError as exc:
        latency = (time.perf_counter() - t0) * 1000
        return RequestResult(
            endpoint=url, status_code=exc.code,
            latency_ms=latency, success=False, error=str(exc),
        )
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return RequestResult(
            endpoint=url, status_code=0,
            latency_ms=latency, success=False, error=str(exc),
        )


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def health_worker(
    base_url: str,
    requests_per_worker: int,
    deadline: float,
) -> list[RequestResult]:
    """Repeatedly hit GET /api/health until quota or deadline is reached."""
    url = f"{base_url}/api/health"
    results: list[RequestResult] = []
    for _ in range(requests_per_worker):
        if time.time() >= deadline:
            break
        results.append(_get(url))
    return results


def analyze_worker(
    base_url: str,
    requests_per_worker: int,
    deadline: float,
) -> list[RequestResult]:
    """Repeatedly hit POST /api/analyze until quota or deadline is reached."""
    url = f"{base_url}/api/analyze"
    payload = {"image": f"data:image/jpeg;base64,{_IMAGE_B64}"}
    results: list[RequestResult] = []
    for _ in range(requests_per_worker):
        if time.time() >= deadline:
            break
        results.append(_post_json(url, payload))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(val: float) -> str:
    """Format a float to 2 decimal places."""
    return f"{val:.2f}"


def print_report(
    health_stats: EndpointStats,
    analyze_stats: EndpointStats,
    wall_seconds: float,
) -> None:
    """Print a clean summary table to stdout."""
    divider = "+" + "-" * 24 + "+" + "-" * 18 + "+" + "-" * 18 + "+"
    header = (
        f"| {'Metric':<22} | {'GET /api/health':>16} | {'POST /api/analyze':>16} |"
    )

    print()
    print("=" * 64)
    print("  LOAD TEST RESULTS")
    print("=" * 64)
    print()
    print(divider)
    print(header)
    print(divider)

    rows = [
        ("Total Requests", str(health_stats.total), str(analyze_stats.total)),
        ("Successful", str(health_stats.success), str(analyze_stats.success)),
        ("Failed", str(health_stats.failed), str(analyze_stats.failed)),
        ("Avg Latency (ms)", _fmt(health_stats.avg_ms), _fmt(analyze_stats.avg_ms)),
        ("Min Latency (ms)", _fmt(health_stats.min_ms), _fmt(analyze_stats.min_ms)),
        ("Max Latency (ms)", _fmt(health_stats.max_ms), _fmt(analyze_stats.max_ms)),
        ("P50 Latency (ms)", _fmt(health_stats.p50), _fmt(analyze_stats.p50)),
        ("P95 Latency (ms)", _fmt(health_stats.p95), _fmt(analyze_stats.p95)),
        ("P99 Latency (ms)", _fmt(health_stats.p99), _fmt(analyze_stats.p99)),
    ]

    for label, h_val, a_val in rows:
        print(f"| {label:<22} | {h_val:>16} | {a_val:>16} |")

    print(divider)
    print()

    total_requests = health_stats.total + analyze_stats.total
    rps = total_requests / wall_seconds if wall_seconds > 0 else 0.0
    print(f"  Wall time:           {wall_seconds:.2f} s")
    print(f"  Total requests:      {total_requests}")
    print(f"  Throughput:          {rps:.2f} req/s")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_load_test(
    base_url: str,
    workers: int,
    requests_per_worker: int,
    duration: int,
) -> None:
    """Execute the load test and print results."""

    print(f"  Target:              {base_url}")
    print(f"  Workers:             {workers} per endpoint ({workers * 2} total)")
    print(f"  Requests/worker:     {requests_per_worker}")
    print(f"  Duration cap:        {duration} s")
    print()

    # --- Connectivity check ---------------------------------------------------
    print("  Checking server connectivity ... ", end="", flush=True)
    probe = _get(f"{base_url}/api/health", timeout=5.0)
    if not probe.success:
        print("FAILED")
        print(f"  Could not reach {base_url}/api/health  (status={probe.status_code})")
        if probe.error:
            print(f"  Error: {probe.error}")
        print("\n  Ensure the server is running and try again.")
        sys.exit(1)
    print(f"OK  ({probe.latency_ms:.0f} ms)")
    print()

    # --- Run -----------------------------------------------------------------
    deadline = time.time() + duration
    wall_start = time.perf_counter()

    all_results: list[RequestResult] = []

    with ThreadPoolExecutor(max_workers=workers * 2) as pool:
        futures = []

        # Submit health workers
        for _ in range(workers):
            futures.append(
                pool.submit(health_worker, base_url, requests_per_worker, deadline)
            )

        # Submit analyze workers
        for _ in range(workers):
            futures.append(
                pool.submit(analyze_worker, base_url, requests_per_worker, deadline)
            )

        completed = 0
        total_futures = len(futures)
        for future in as_completed(futures):
            completed += 1
            results = future.result()
            all_results.extend(results)
            print(
                f"\r  Progress: {completed}/{total_futures} workers done "
                f"({len(all_results)} requests so far)",
                end="", flush=True,
            )

    wall_seconds = time.perf_counter() - wall_start
    print()

    # --- Aggregate stats -----------------------------------------------------
    health_url = f"{base_url}/api/health"
    analyze_url = f"{base_url}/api/analyze"

    health_stats = EndpointStats(name="GET /api/health")
    analyze_stats = EndpointStats(name="POST /api/analyze")

    for r in all_results:
        if r.endpoint == health_url:
            stats = health_stats
        elif r.endpoint == analyze_url:
            stats = analyze_stats
        else:
            continue

        stats.total += 1
        if r.success:
            stats.success += 1
        else:
            stats.failed += 1
        stats.latencies.append(r.latency_ms)

    print_report(health_stats, analyze_stats, wall_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load test for the Lite-Vision API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python load_test.py\n"
            "  python load_test.py --workers 10 --duration 60\n"
            "  python load_test.py --base-url https://my-api.com --requests 50\n"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help=f"Base URL of the API server (default: {BASE_URL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CONCURRENT_WORKERS,
        help=f"Number of concurrent workers per endpoint (default: {CONCURRENT_WORKERS})",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=REQUESTS_PER_WORKER,
        dest="requests_per_worker",
        help=f"Requests each worker sends (default: {REQUESTS_PER_WORKER})",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=TOTAL_DURATION_SECONDS,
        help=f"Maximum test duration in seconds (default: {TOTAL_DURATION_SECONDS})",
    )

    args = parser.parse_args()

    print()
    print("=" * 64)
    print("  LITE-VISION LOAD TEST")
    print("=" * 64)
    print()

    run_load_test(
        base_url=args.base_url.rstrip("/"),
        workers=args.workers,
        requests_per_worker=args.requests_per_worker,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
