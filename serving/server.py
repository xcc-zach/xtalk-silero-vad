import asyncio
import base64
import binascii
import concurrent.futures
import io
import json
import os
import tempfile
import wave
import time
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

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


class StreamingVadSession:
    def __init__(
        self,
        model,
        sampling_rate: int,
        frame_samples: int,
        positive_speech_threshold: float,
        negative_speech_threshold: float,
        redemption_frames: int,
    ):
        if sampling_rate not in {8000, 16000}:
            raise ValueError("streaming sample_rate must be 8000 or 16000")
        expected_frame_samples = 512 if sampling_rate == 16000 else 256
        if frame_samples != expected_frame_samples:
            raise ValueError(f"frame_samples must be {expected_frame_samples} for sample_rate={sampling_rate}")

        self.model = model
        self.sampling_rate = sampling_rate
        self.frame_samples = frame_samples
        self.positive_speech_threshold = positive_speech_threshold
        self.negative_speech_threshold = negative_speech_threshold
        self.redemption_frames = redemption_frames
        self.pending = torch.empty(0, dtype=torch.float32)
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self.pending = torch.empty(0, dtype=torch.float32)
        self.seq = 0
        self.in_speech = False
        self.redemption_counter = 0

    def _timestamp_ms(self) -> int:
        return int(round(self.seq * self.frame_samples / self.sampling_rate * 1000))

    def _process_frame(self, frame: torch.Tensor) -> Dict[str, Any]:
        speech_prob = float(self.model(frame, self.sampling_rate).item())
        not_speech_prob = 1.0 - speech_prob

        if not self.in_speech:
            if speech_prob >= self.positive_speech_threshold:
                self.in_speech = True
                self.redemption_counter = 0
        else:
            if speech_prob < self.negative_speech_threshold:
                self.redemption_counter += 1
                if self.redemption_counter >= self.redemption_frames:
                    self.in_speech = False
                    self.redemption_counter = 0
            else:
                self.redemption_counter = 0

        response = {
            "type": "frame",
            "seq": self.seq,
            "timestamp_ms": self._timestamp_ms(),
            "speech_prob": round(speech_prob, 6),
            "not_speech_prob": round(not_speech_prob, 6),
            "is_speech": bool(self.in_speech),
        }
        self.seq += 1
        return response

    def process_audio(self, audio: torch.Tensor) -> List[Dict[str, Any]]:
        if audio.numel() == 0:
            return []

        self.pending = torch.cat([self.pending, audio.float()])
        messages: List[Dict[str, Any]] = []

        while self.pending.numel() >= self.frame_samples:
            frame = self.pending[: self.frame_samples]
            self.pending = self.pending[self.frame_samples :]
            messages.append(self._process_frame(frame))

        return messages

    def flush(self) -> List[Dict[str, Any]]:
        if self.pending.numel() == 0:
            return []

        frame = torch.nn.functional.pad(self.pending, (0, self.frame_samples - self.pending.numel()))
        self.pending = torch.empty(0, dtype=torch.float32)
        return [self._process_frame(frame)]


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
    return {"service": "silero-vad", "health": "/health", "vad": "/v1/vad", "vad_stream": "/ws/vad"}


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


def _stream_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    sample_rate = int(payload.get("sample_rate", 16000))
    frame_samples = int(payload.get("frame_samples", 512 if sample_rate == 16000 else 256))
    encoding = str(payload.get("encoding", "pcm_s16le"))
    channels = int(payload.get("channels", 1))
    positive_speech_threshold = float(payload.get("positive_speech_threshold", 0.8))
    negative_speech_threshold = float(payload.get("negative_speech_threshold", 0.2))
    redemption_frames = int(payload.get("redemption_frames", 16))

    if sample_rate not in {8000, 16000}:
        raise ValueError("sample_rate must be 8000 or 16000 for streaming")
    expected_frame_samples = 512 if sample_rate == 16000 else 256
    if frame_samples != expected_frame_samples:
        raise ValueError(f"frame_samples must be {expected_frame_samples} for sample_rate={sample_rate}")
    if encoding.lower().replace("-", "_") != "pcm_s16le":
        raise ValueError("encoding must be pcm_s16le")
    if channels != 1:
        raise ValueError("channels must be 1")
    if positive_speech_threshold < 0.0 or positive_speech_threshold > 1.0:
        raise ValueError("positive_speech_threshold must be between 0.0 and 1.0")
    if negative_speech_threshold < 0.0 or negative_speech_threshold > 1.0:
        raise ValueError("negative_speech_threshold must be between 0.0 and 1.0")
    if negative_speech_threshold > positive_speech_threshold:
        raise ValueError("negative_speech_threshold must be <= positive_speech_threshold")
    if redemption_frames < 1:
        raise ValueError("redemption_frames must be >= 1")

    return {
        "sample_rate": sample_rate,
        "frame_samples": frame_samples,
        "encoding": encoding,
        "channels": channels,
        "positive_speech_threshold": positive_speech_threshold,
        "negative_speech_threshold": negative_speech_threshold,
        "redemption_frames": redemption_frames,
    }


async def _send_ws_error(websocket: WebSocket, code: str, message: str) -> None:
    await websocket.send_json({"type": "error", "code": code, "message": message})


async def _send_frame_responses(websocket: WebSocket, responses: List[Dict[str, Any]]) -> None:
    for response in responses:
        await websocket.send_json(response)


@app.websocket("/ws/vad")
async def vad_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    if _executor is None or _worker_queue is None:
        await _send_ws_error(websocket, "service_unavailable", "service is still starting")
        await websocket.close(code=1013)
        return

    loop = asyncio.get_running_loop()
    worker: Optional[VadWorker] = None
    session: Optional[StreamingVadSession] = None
    config: Optional[Dict[str, Any]] = None

    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                break

            text = message.get("text")
            data = message.get("bytes")

            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    await _send_ws_error(websocket, "invalid_json", f"invalid JSON message: {exc.msg}")
                    continue

                command = payload.get("type")
                if command == "start":
                    try:
                        config = _stream_config(payload)
                    except Exception as exc:
                        await _send_ws_error(websocket, "invalid_start", str(exc))
                        continue

                    if worker is None:
                        worker = await _worker_queue.get()

                    session = StreamingVadSession(
                        worker.model,
                        config["sample_rate"],
                        config["frame_samples"],
                        config["positive_speech_threshold"],
                        config["negative_speech_threshold"],
                        config["redemption_frames"],
                    )
                    await websocket.send_json(
                        {
                            "type": "start_ack",
                            "sample_rate": config["sample_rate"],
                            "frame_samples": session.frame_samples,
                        }
                    )
                elif command == "audio":
                    if session is None or config is None:
                        await _send_ws_error(websocket, "not_started", "send start before audio")
                        continue
                    try:
                        audio_b64 = payload["audio"]
                        if not isinstance(audio_b64, str):
                            raise ValueError("audio must be a base64 string")
                        audio_bytes = base64.b64decode(audio_b64, validate=True)
                        audio = _decode_pcm(audio_bytes, config["sample_rate"], config["encoding"], config["channels"])
                        responses = await loop.run_in_executor(_executor, partial(session.process_audio, audio))
                    except KeyError:
                        await _send_ws_error(websocket, "invalid_frame", "missing audio")
                        continue
                    except (binascii.Error, ValueError) as exc:
                        await _send_ws_error(websocket, "invalid_frame", str(exc))
                        continue
                    except Exception as exc:
                        await _send_ws_error(websocket, "internal_error", str(exc))
                        continue
                    await _send_frame_responses(websocket, responses)
                elif command == "reset":
                    if session is None:
                        await _send_ws_error(websocket, "not_started", "stream has not been started")
                    else:
                        await loop.run_in_executor(_executor, session.reset)
                        await websocket.send_json({"type": "reset_ack"})
                elif command == "flush":
                    if session is None:
                        await _send_ws_error(websocket, "not_started", "stream has not been started")
                    else:
                        responses = await loop.run_in_executor(_executor, session.flush)
                        await _send_frame_responses(websocket, responses)
                        await websocket.send_json({"type": "flush_ack"})
                elif command == "close":
                    await websocket.close()
                    break
                else:
                    await _send_ws_error(websocket, "unsupported_message", "unsupported message type")
                continue

            if data is None:
                continue

            if session is None or config is None:
                await _send_ws_error(websocket, "not_started", "send start before binary audio")
                continue

            try:
                audio = _decode_pcm(data, config["sample_rate"], config["encoding"], config["channels"])
                responses = await loop.run_in_executor(_executor, partial(session.process_audio, audio))
            except ValueError as exc:
                await _send_ws_error(websocket, "invalid_frame", str(exc))
                continue
            except Exception as exc:
                await _send_ws_error(websocket, "internal_error", str(exc))
                continue

            await _send_frame_responses(websocket, responses)
    except WebSocketDisconnect:
        pass
    finally:
        if worker is not None:
            _worker_queue.put_nowait(worker)
