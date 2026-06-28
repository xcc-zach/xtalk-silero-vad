# Silero VAD One-Command Serving

This README documents the deployment wrapper and HTTP client usage. The original project README is preserved as [README.original.md](README.original.md).

## Requirements

- Linux or macOS
- Python 3.8+
- CPU serving is supported with multiple concurrent workers
- GPU concurrency requires CUDA PyTorch and a working NVIDIA driver

## Install

```bash
chmod +x install.sh start.sh
bash install.sh
```

`install.sh` disables proxy settings while downloading pip packages: it unsets `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and their lowercase variants, and runs `pip --isolated` so user pip proxy configuration is ignored. If you need a package mirror, set `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, or `PIP_TRUSTED_HOST`; the script passes those values to pip explicitly.

The script installs CPU PyTorch wheels by default, so a machine with `nvidia-smi` does not silently download large CUDA wheels. CPU concurrency needs no extra setup. To enable GPU concurrency, install CUDA PyTorch explicitly:

```bash
INSTALL_GPU_TORCH=1 bash install.sh
```

## Start

```bash
bash start.sh --port 8000 --model-workers 4
```

CPU concurrency uses ONNX Runtime by default. Each worker owns one independent model instance:

```bash
bash start.sh --port 8000 --device cpu --onnx --model-workers 8
```

GPU concurrency uses TorchScript. Each worker owns one GPU model instance; install CUDA PyTorch first with `INSTALL_GPU_TORCH=1 bash install.sh`:

```bash
bash start.sh --port 8000 --device cuda --torch --model-workers 4
```

Options:

- `--host`: defaults to `0.0.0.0`
- `--port`: defaults to `8000`
- `--model-workers`: number of model instances, defaults to `min(CPU cores, 4)`; this controls both CPU and GPU concurrency
- `--device`: `cpu`, `cuda`, `cuda:<index>`, or `auto`
- `--cpu`: same as `--device cpu`
- `--cuda`/`--gpu`: same as `--device cuda --torch`
- `--onnx`: default, use ONNX Runtime CPU inference
- `--torch`: use TorchScript inference; required for GPU mode

## Client Requests

Health check:

```bash
curl http://localhost:8000/health
```

Send a WAV file:

```bash
curl -s -X POST "http://localhost:8000/v1/vad?sample_rate=16000&return_seconds=true" \
  -H "Content-Type: audio/wav" \
  --data-binary @tests/data/test.wav
```

Send raw mono `s16le` PCM:

```bash
curl -s -X POST "http://localhost:8000/v1/vad?sample_rate=16000&encoding=pcm_s16le&channels=1" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.s16le
```

Example response:

```json
{
  "segments": [{"start": 0.3, "end": 1.7}],
  "sample_rate": 16000,
  "duration_seconds": 2.0,
  "speech_seconds": 1.4,
  "processing_seconds": 0.01,
  "worker_id": 0
}
```

## Benchmark

Start the service first, then run:

```bash
source .venv/bin/activate
python serving/performance_testing.py --concurrency 1,4,16,64,256 --requests 64
```

The default audio file is `tests/data/test.wav`. To use audio from the Hugging Face `xcczach/sample-data` dataset, download it locally and pass `--audio path/to/audio.wav`.

Validation environment: Linux 5.15.0-141-generic x86_64, Intel Xeon Gold 6530, Python 3.13.9, torch 2.9.1+cpu, ONNX Runtime 1.27.0, --model-workers 1, sample tests/data/test.wav, 8 requests per concurrency level.

| concurrency | ok/total | errors | avg latency (s) | p50 (s) | p95 (s) | throughput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8/8 | 0 | 0.394 | 0.383 | 0.442 | 2.54 |
| 4 | 8/8 | 0 | 1.238 | 1.519 | 1.527 | 2.63 |
| 16 | 8/8 | 0 | 1.748 | 1.941 | 3.083 | 2.59 |
| 64 | 8/8 | 0 | 1.723 | 1.912 | 3.041 | 2.63 |
| 256 | 8/8 | 0 | 1.721 | 1.907 | 3.043 | 2.63 |

GPU validation environment: Linux 5.15.0-141-generic x86_64, NVIDIA GeForce RTX 4090, driver 570.124.06, Python 3.13.9, torch 2.9.1+cu128, CUDA 12.8, ONNX Runtime 1.27.0, start arguments --device cuda:0 --torch --model-workers 4, sample tests/data/test.wav, 64 requests per concurrency level. GPU0 memory usage before benchmark was about 26170 MiB/49140 MiB; the service ran on GPU0.

| concurrency | ok/total | errors | avg latency (s) | p50 (s) | p95 (s) | throughput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64/64 | 0 | 0.811 | 0.803 | 0.913 | 1.23 |
| 4 | 64/64 | 0 | 2.027 | 2.017 | 2.166 | 1.97 |
| 16 | 64/64 | 0 | 7.384 | 8.127 | 8.274 | 1.96 |
| 64 | 64/64 | 0 | 17.391 | 18.368 | 32.308 | 1.96 |
| 256 | 64/64 | 0 | 17.312 | 18.243 | 32.192 | 1.97 |
