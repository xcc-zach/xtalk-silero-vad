import asyncio
import concurrent.futures
import io
import os
import tempfile
import wave
import time
from functools import partial
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio


app = FastAPI(title="Silero VAD Service", version="1.0.0")

_workers: List["VadWorker"] = []
_worker_queue: asyncio.Queue = None
_executor: concurrent.futures.ThreadPoolExecutor = None
_use_onnx = True
_device = "cpu"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    return max(min_value, min(max_value, parsed))


class DeviceModel:
    def __init__(self, model, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def reset_states(self):
        result = self.model.reset_states()
        self.model.to(self.device)
        return result

    def __call__(self, x: torch.Tensor, sampling_rate: int):
        output = self.model(x.to(self.device, non_blocking=True), sampling_rate)
        return output.detach().cpu()


class VadWorker:
    def __init__(self, worker_id: int, use_onnx: bool, device: str):
        self.worker_id = worker_id
        self.device = device
        if use_onnx:
            self.model = load_silero_vad(onnx=True)
        else:
            model = load_silero_vad(onnx=False)
            self.model = DeviceModel(model, torch.device(device))

    def detect(self, audio: torch.Tensor, params: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        segments = get_speech_timestamps(audio, self.model, **params)
        elapsed = time.perf_counter() - started
        sample_rate = params["sampling_rate"]
        duration_seconds = float(audio.numel()) / float(sample_rate)

        if params.get("return_seconds"):
            speech_seconds = sum(float(item["end"] - item["start"]) for item in segments)
        else:
            speech_seconds = sum(float(item["end"] - item["start"]) for item in segments) / float(sample_rate)

        return {
            "segments": segments,
            "sample_rate": sample_rate,
            "duration_seconds": round(duration_seconds, 6),
            "speech_seconds": round(speech_seconds, 6),
            "processing_seconds": round(elapsed, 6),
            "worker_id": self.worker_id,
            "device": self.device,
        }


@app.on_event("startup")
async def startup() -> None:
    global _device, _executor, _use_onnx, _worker_queue, _workers

    torch.set_num_threads(_env_int("TORCH_NUM_THREADS", 1, 1, 64))
    cpu_count = os.cpu_count() or 1
    worker_count = _env_int("VAD_MODEL_WORKERS", min(cpu_count, 4), 1, 64)
    _use_onnx = _env_bool("VAD_USE_ONNX", True)
    requested_device = os.getenv("VAD_DEVICE", "cpu").strip().lower()
    if requested_device == "auto":
        _device = "cuda" if (not _use_onnx and torch.cuda.is_available()) else "cpu"
    else:
        _device = requested_device

    if _device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("VAD_DEVICE=cuda was requested, but torch.cuda.is_available() is false")
        if _use_onnx:
            raise RuntimeError("GPU serving uses TorchScript. Start with --device cuda --torch.")
    elif _device != "cpu":
        raise RuntimeError("VAD_DEVICE must be cpu, cuda, cuda:<index>, or auto")

    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
    _workers = [VadWorker(index, _use_onnx, _device) for index in range(worker_count)]
    _worker_queue = asyncio.Queue(maxsize=worker_count)
    for worker in _workers:
        _worker_queue.put_nowait(worker)


@app.on_event("shutdown")
async def shutdown() -> None:
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_workers": len(_workers),
        "onnx": _use_onnx,
        "device": _device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }


@app.get("/")
async def root() -> Dict[str, str]:
    return {"service": "silero-vad", "health": "/health", "vad": "/v1/vad"}


def _decode_pcm(body: bytes, sample_rate: int, encoding: str, channels: int) -> torch.Tensor:
    if channels < 1:
        raise ValueError("channels must be >= 1")

    normalized = encoding.lower().replace("-", "_")
    if normalized in {"pcm_s16le", "s16le", "int16"}:
        if len(body) % 2:
            raise ValueError("pcm_s16le payload length must be divisible by 2")
        audio = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    elif normalized in {"pcm_f32le", "f32le", "float32"}:
        if len(body) % 4:
            raise ValueError("pcm_f32le payload length must be divisible by 4")
        audio = np.frombuffer(body, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported PCM encoding: {encoding}")

    if audio.size == 0:
        raise ValueError("empty audio payload")
    if audio.size % channels:
        raise ValueError("payload sample count is not divisible by channels")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    tensor = torch.from_numpy(audio.copy()).float()
    if sample_rate not in {8000, 16000} and sample_rate % 16000 != 0:
        raise ValueError("sample_rate must be 8000, 16000, or a multiple of 16000")
    return tensor


def _resample_audio(audio: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return audio
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    target_length = max(1, round(audio.numel() * float(target_rate) / float(source_rate)))
    return torch.nn.functional.interpolate(
        audio.view(1, 1, -1), size=target_length, mode="linear", align_corners=False
    ).view(-1)


def _decode_wav(body: bytes, target_sample_rate: int) -> torch.Tensor:
    with wave.open(io.BytesIO(body), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels < 1:
        raise ValueError("WAV channel count must be >= 1")
    if audio.size % channels:
        raise ValueError("WAV frame data is not divisible by channel count")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    tensor = torch.from_numpy(audio.copy()).float()
    return _resample_audio(tensor, source_rate, target_sample_rate)


def _decode_file(body: bytes, sample_rate: int, content_type: str) -> torch.Tensor:
    suffix = ".wav"
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    if content_type in {"audio/wav", "audio/x-wav", "audio/wave"}:
        try:
            return _decode_wav(body, sample_rate)
        except wave.Error as exc:
            raise ValueError(f"invalid WAV payload: {exc}") from exc
    if content_type == "audio/mpeg":
        suffix = ".mp3"
    elif content_type == "audio/ogg":
        suffix = ".opus"
    elif content_type in {"audio/flac", "audio/x-flac"}:
        suffix = ".flac"

    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(body)
        handle.flush()
        return read_audio(handle.name, sampling_rate=sample_rate).float()


def _decode_audio(body: bytes, sample_rate: int, encoding: str, channels: int, content_type: str) -> torch.Tensor:
    if not body:
        raise ValueError("empty request body")

    normalized = encoding.lower().replace("-", "_")
    if normalized == "auto":
        media_type = (content_type or "").split(";", 1)[0].strip().lower()
        if media_type in {"application/octet-stream", "audio/pcm", "audio/raw"}:
            normalized = "pcm_s16le"
        else:
            return _decode_file(body, sample_rate, content_type)

    if normalized.startswith("pcm") or normalized in {"s16le", "f32le", "int16", "float32"}:
        return _decode_pcm(body, sample_rate, normalized, channels)
    return _decode_file(body, sample_rate, content_type)


@app.post("/v1/vad")
async def vad(
    request: Request,
    sample_rate: int = Query(16000, description="Input sample rate. Use 8000 or 16000 for PCM."),
    encoding: str = Query("auto", description="auto, pcm_s16le, or pcm_f32le"),
    channels: int = Query(1, ge=1, le=8),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    min_speech_duration_ms: int = Query(250, ge=0),
    min_silence_duration_ms: int = Query(100, ge=0),
    speech_pad_ms: int = Query(30, ge=0),
    return_seconds: bool = Query(True),
) -> Dict[str, Any]:
    if _executor is None or _worker_queue is None:
        raise HTTPException(status_code=503, detail="service is still starting")

    body = await request.body()
    loop = asyncio.get_running_loop()

    try:
        audio = await loop.run_in_executor(
            _executor,
            partial(_decode_audio, body, sample_rate, encoding, channels, request.headers.get("content-type", "")),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    params = {
        "sampling_rate": sample_rate,
        "threshold": threshold,
        "min_speech_duration_ms": min_speech_duration_ms,
        "min_silence_duration_ms": min_silence_duration_ms,
        "speech_pad_ms": speech_pad_ms,
        "return_seconds": return_seconds,
    }

    worker = await _worker_queue.get()
    try:
        return await loop.run_in_executor(_executor, partial(worker.detect, audio, params))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _worker_queue.put_nowait(worker)
