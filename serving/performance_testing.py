#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the Silero VAD HTTP service.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/vad", help="VAD endpoint URL")
    parser.add_argument("--audio", default="tests/data/test.wav", help="Audio file sent as request body")
    parser.add_argument("--audio-url", default="", help="Download audio to --audio if the file does not exist")
    parser.add_argument("--concurrency", default="1,4,16,64,256", help="Comma-separated concurrency levels")
    parser.add_argument("--requests", type=int, default=64, help="Requests per concurrency level")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    return parser.parse_args()


def ensure_audio(path: Path, audio_url: str) -> None:
    if path.exists():
        return
    if not audio_url:
        raise FileNotFoundError(f"Audio file not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(audio_url, timeout=60) as response:
        path.write_bytes(response.read())


def post_once(url: str, body: bytes, timeout: float):
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    return elapsed, payload


def percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[index]


def benchmark_level(url: str, body: bytes, concurrency: int, requests: int, timeout: float):
    started = time.perf_counter()
    latencies = []
    errors = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(post_once, url, body, timeout) for _ in range(requests)]
        for future in concurrent.futures.as_completed(futures):
            try:
                latency, _ = future.result()
                latencies.append(latency)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
                errors += 1

    wall_seconds = time.perf_counter() - started
    ok = len(latencies)
    return {
        "concurrency": concurrency,
        "requests": requests,
        "ok": ok,
        "errors": errors,
        "avg_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "throughput_rps": ok / wall_seconds if wall_seconds > 0 else 0.0,
    }


def main():
    args = parse_args()
    audio_path = Path(args.audio)
    ensure_audio(audio_path, args.audio_url)
    body = audio_path.read_bytes()
    levels = [int(item.strip()) for item in args.concurrency.split(",") if item.strip()]

    print("| concurrency | ok/total | errors | avg latency (s) | p50 (s) | p95 (s) | throughput (req/s) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for level in levels:
        result = benchmark_level(args.url, body, level, args.requests, args.timeout)
        print(
            f"| {result['concurrency']} | {result['ok']}/{result['requests']} | {result['errors']} | "
            f"{result['avg_latency_s']:.3f} | {result['p50_latency_s']:.3f} | "
            f"{result['p95_latency_s']:.3f} | {result['throughput_rps']:.2f} |"
        )


if __name__ == "__main__":
    main()
