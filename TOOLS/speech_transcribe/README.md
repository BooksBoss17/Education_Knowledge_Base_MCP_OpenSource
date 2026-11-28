# 本地语音转写

通过 MCP 工具 `speech_transcribe` 转写本地音频或视频的第一条音轨。语言可省略以自动检测，或传入官方语言名（例如 Chinese）。默认同时运行 Qwen3-ForcedAligner-0.6B，提供相对于音轨开头的秒数。不支持对齐的语种保留转写并在片段标记 unsupported_language。传 timestamps=false 仅运行 Qwen3-ASR-1.7B。

1. 调用 `{"action":"start","source":"Original_Backup/INSTRUCTIONAL_VIDEOS/示例/video.mp4"}`。相对路径从知识库工作区解析，也可用已配置输入根中的绝对路径。
2. 将返回的 job_id 传入 `{"action":"status","job_id":"…"}`。长任务可断开后重新连接。同一任务可带 task_run_id 接入统一监控和收尾，取消使用 runtime_jobs。
3. 成功后调用 `{"action":"read","job_id":"…","offset":0,"limit":1}` 读取片段正文和字词时间戳。按 next_offset 翻页。完整 transcript.txt、segments.jsonl 与 summary.json 在本工作区 Output/speech_transcribe/<job_id>/。

模型位于 MCP 的 MODELS 目录。语音工作进程使用 .local/runtimes/qwen3-speech-006 下的独立 Windows Python 3.11 环境，通过本地 PyAV 解码媒体，按剩余显存为两个模型分别选择 CUDA 或 CPU。作业结束后卸载模型。依赖版本见 requirements.lock，可在新机器为该解释器用 uv pip install --python 指定路径并通过 -r requirements.lock 安装。权重版本与哈希见各自 MODEL_MANIFEST.json。

## 可替换的语音后端

默认配置在 `TOOLS/speech_transcribe/providers.json`。要调整本机，复制为 `.local/speech-api.json` 后编辑，下次提交即读取。任务指纹绑定生效角色的配置、HTTP 模型 revision 与本地模型清单。排队期间生效配置改变时，该次作业会报错，可用新配置重新提交。已有完成作业可继续 status/read。

- `timing=aligner` 先调用 ASR，再把同一窗口的音频、原文和语言交给 aligner。两个角色可独立选用 managed_http（工具启动本机服务）或 http（连接已部署服务）。默认使用 Qwen3-ASR-1.7B 和 Qwen3-ForcedAligner-0.6B。
- `timing=native` 只调用 asr，由它在一个响应中给出正文和时间，granularity 取 segment、word、character。原生时间戳描述本次转写。
- `timing=none` 默认只返回正文。显式请求 timestamps=true 会提示不支持。其他模式传 timestamps=false 也只运行转录。仅有文本的结果仍含解码窗口起止时间。

### 本机 API 生命周期

默认两个 Qwen 角色均使用 HTTP POST，内置服务以 16 kHz 单声道 float32 原样传递解码采样，外部 speech-api-v1 端点使用下文的 PCM16 WAV。第一次请求某个角色时，本机服务加载对应模型；同一作业内的音频窗口复用它。工具仅为实际启用的 managed_http 角色启动一个子进程，绑定 127.0.0.1 的动态端口，以作业专用令牌认证，作业收尾或受管进程树终止时释放模型。timestamps=false 仅启用识别角色。已有配置的 qwen 类型同样解析为 managed_http。

要跨作业复用已加载的模型，将相应角色设为 http，指向由你管理的常驻服务。可以让一个服务提供两个端点，也可以分别配置端点。模型名称、版本和时间戳能力均在配置文件内选择。summary.json 中 http_calls 分角色记录成功的推理请求次数。

单模型配置示例（端口由部署者选择）：

```json
{
  "schema": "speech-providers-v1",
  "timing": "native",
  "asr": {
    "type": "http", "endpoint": "http://127.0.0.1:8075/transcribe",
    "model": "my-asr", "revision": "v1", "granularity": "word"
  }
}
```

HTTP 适配器向配置的地址 POST JSON。请求包含 schema=speech-api-v1、role（transcribe 或 align）、model、revision、language、granularity 和 audio_wav_base64（单声道 16 kHz PCM16 WAV）。align 请求另有 text。转录响应示例为 `{"text":"你好","language":"Chinese","timestamps":[{"text":"你好","start":0.1,"end":0.6}]}`；align 响应只需 timestamps。纯文本请求省略返回的 timestamps。时间以本窗口开头为零点，单位秒，按先后排列且区间不重叠。静音可返回空文本和空数组；不支持对齐的语言可显式返回 alignment=unsupported_language 和空数组。API 服务更新模型或预处理后同步更新 revision。已有其他协议的服务可用网关转换此契约。默认接受回环地址，向外部服务传音频须在对应角色显式设置 allow_remote=true 并使用 HTTPS；令牌通过 token_env 指定的运行服务环境变量提供。

结果保持原有文件布局。字词写入 timestamps，原生片段写入 timed_segments。摘要给出 timing_mode、timestamp_granularity、后端模型和 route_fingerprint。audio_to_media 记录以解码音轨计时的分段起点和对应播放起点（秒），同一段内加上已过时长；播放起点为 null 时须回源确认。配置 HTTP-only 可复用本机语音环境，或在 .local/runtimes/speech-api-v1 准备同平台 Python 3.11 与 av、numpy 供解码使用。更换运行解释器后刷新空闲管理器。

## GitHub 公开发行的可选安装

Windows Python 3.11 下，已经下载上游模型的用户可运行 python scripts/setup_video.py --speech-model-root <目录>，安装工具会核对模型清单和依赖。

只使用 HTTP 识别或对齐服务时，从 config/speech-api.example.json 复制并编辑服务地址和模型版本后，运行 python scripts/setup_video.py --speech-api-config <文件>。该路径安装轻量媒体解码环境，并保存生效配置。更新安装后重新连接；共享管理器处于活跃状态时需等待作业结束，按现有运行管理流程重新启动。
