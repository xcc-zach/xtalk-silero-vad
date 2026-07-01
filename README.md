# Silero VAD 一键部署服务

本 README 为服务一键部署和客户端使用说明。原始项目说明见 [README.original.md](README.original.md)。

## 环境要求

- Linux 或 macOS
- Python 3.8+
- CPU 可运行，支持多 worker 并发
- GPU 并发需要 CUDA 版 PyTorch 和可用 NVIDIA 驱动

## 安装

```bash
chmod +x install.sh start.sh
bash install.sh
```

`install.sh` 在下载 pip 包时会关闭代理：脚本会清空 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 等大小写代理环境变量，并使用 `pip --isolated` 避免读取 pip 用户配置中的代理。若需要指定镜像源，可继续使用 `PIP_INDEX_URL`、`PIP_EXTRA_INDEX_URL` 和 `PIP_TRUSTED_HOST`，脚本会把这些值显式传给 pip。

脚本默认安装 CPU 版 PyTorch，避免在有 `nvidia-smi` 的机器上隐式下载很大的 CUDA wheel。CPU 并发不需要额外配置。若要启用 GPU 并发，需要安装 CUDA 版 PyTorch：

```bash
INSTALL_GPU_TORCH=1 bash install.sh
```

## 服务启动

```bash
bash start.sh --port 8000 --model-workers 4
```

CPU 并发默认使用 ONNX Runtime，每个 worker 持有一个独立模型实例：

```bash
bash start.sh --port 8000 --device cpu --onnx --model-workers 8
```

GPU 并发使用 TorchScript，每个 worker 持有一个 GPU 模型实例；启动前需要用 `INSTALL_GPU_TORCH=1 bash install.sh` 安装 CUDA 版 PyTorch：

```bash
bash start.sh --port 8000 --device cuda --torch --model-workers 4
```

可选参数：

- `--host`：默认 `0.0.0.0`
- `--port`：默认 `8000`
- `--model-workers`：模型实例数量，默认 `min(CPU 核数, 4)`；CPU/GPU 并发都由这个参数控制
- `--device`：`cpu`、`cuda`、`cuda:<index>` 或 `auto`
- `--cpu`：等价于 `--device cpu`
- `--cuda`/`--gpu`：等价于 `--device cuda --torch`
- `--onnx`：默认，使用 ONNX Runtime CPU 推理
- `--torch`：使用 TorchScript 推理，GPU 模式必须使用该后端

## 客户端请求

健康检查：

```bash
curl http://localhost:8000/health
```

发送 WAV 文件：

```bash
curl -s -X POST "http://localhost:8000/v1/vad?sample_rate=16000&return_seconds=true" \
  -H "Content-Type: audio/wav" \
  --data-binary @tests/data/test.wav
```

发送裸 PCM，格式为单声道 `s16le`：

```bash
curl -s -X POST "http://localhost:8000/v1/vad?sample_rate=16000&encoding=pcm_s16le&channels=1" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.s16le
```

响应示例：

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

流式 VAD 使用 WebSocket，一条连接对应一个连续音频流：

```text
ws://localhost:8000/ws/vad
```

连接后先发送 `start` 控制消息，再持续发送二进制 PCM16 mono little-endian 音频。binary 消息可以任意大小，服务端会按 `frame_samples` 切帧；16 kHz 推荐 512 samples。

```json
{"type":"start","sample_rate":16000,"frame_samples":512,"encoding":"pcm_s16le","channels":1,"positive_speech_threshold":0.8,"negative_speech_threshold":0.2,"redemption_frames":16}
```

也支持 JSON base64 音频，但推荐使用 binary：

```json
{"type":"audio","seq":12,"audio":"<base64 pcm_s16le>"}
```

客户端控制消息：

```json
{"type":"reset"}
{"type":"flush"}
{"type":"close"}
```

服务端事件：

```json
{"type":"start_ack","sample_rate":16000,"frame_samples":512}
{"type":"frame","seq":42,"timestamp_ms":1344,"speech_prob":0.91,"not_speech_prob":0.09,"is_speech":true}
{"type":"reset_ack"}
{"type":"flush_ack"}
{"type":"error","code":"invalid_frame","message":"audio must be pcm_s16le mono"}
```

`flush` 会把缓冲区里不足一帧的残留音频补零并返回该帧结果，然后返回 `flush_ack`；它不会重置模型状态。要结束当前流并重新开始，请发送 `reset`。

## 压测

先启动服务，然后运行：

```bash
source .venv/bin/activate
python serving/performance_testing.py --concurrency 1,4,16,64,256 --requests 64
```

默认使用仓库内 `tests/data/test.wav`。如需使用 Hugging Face `xcczach/sample-data` 中的音频，可先下载到本地，再通过 `--audio path/to/audio.wav` 指定。

验证环境：Linux 5.15.0-141-generic x86_64，Intel Xeon Gold 6530，Python 3.13.9，torch 2.9.1+cpu，ONNX Runtime 1.27.0，--model-workers 1，样本 tests/data/test.wav，每档 8 个请求。

| 并发 | 成功/总数 | 错误 | 平均延时(s) | p50(s) | p95(s) | 吞吐(req/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8/8 | 0 | 0.394 | 0.383 | 0.442 | 2.54 |
| 4 | 8/8 | 0 | 1.238 | 1.519 | 1.527 | 2.63 |
| 16 | 8/8 | 0 | 1.748 | 1.941 | 3.083 | 2.59 |
| 64 | 8/8 | 0 | 1.723 | 1.912 | 3.041 | 2.63 |
| 256 | 8/8 | 0 | 1.721 | 1.907 | 3.043 | 2.63 |

GPU 验证环境：Linux 5.15.0-141-generic x86_64，NVIDIA GeForce RTX 4090，driver 570.124.06，Python 3.13.9，torch 2.9.1+cu128，CUDA 12.8，ONNX Runtime 1.27.0，启动参数 --device cuda:0 --torch --model-workers 4，样本 tests/data/test.wav，每档 64 个请求。压测前 GPU0 显存占用约 26170 MiB/49140 MiB，压测期间服务使用 GPU0。

| 并发 | 成功/总数 | 错误 | 平均延时(s) | p50(s) | p95(s) | 吞吐(req/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64/64 | 0 | 0.811 | 0.803 | 0.913 | 1.23 |
| 4 | 64/64 | 0 | 2.027 | 2.017 | 2.166 | 1.97 |
| 16 | 64/64 | 0 | 7.384 | 8.127 | 8.274 | 1.96 |
| 64 | 64/64 | 0 | 17.391 | 18.368 | 32.308 | 1.96 |
| 256 | 64/64 | 0 | 17.312 | 18.243 | 32.192 | 1.97 |
