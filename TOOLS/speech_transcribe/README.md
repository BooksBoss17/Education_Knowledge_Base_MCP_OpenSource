# 本地语音转写

通过 MCP 工具 `speech_transcribe` 转写本地音频或视频的第一条音轨。语言可省略以自动检测，或传入官方语言名（例如 Chinese）。默认同时运行 Qwen3-ForcedAligner-0.6B，提供相对于音轨开头的秒数。不支持对齐的语种保留转写并在片段标记 unsupported_language。传 timestamps=false 仅运行 Qwen3-ASR-1.7B。

1. 调用 `{"action":"start","source":"Original_Backup/INSTRUCTIONAL_VIDEOS/示例/video.mp4"}`。相对路径从知识库工作区解析，也可用已配置输入根中的绝对路径。
2. 将返回的 job_id 传入 `{"action":"status","job_id":"…"}`。长任务可断开后重新连接。同一任务可带 task_run_id 接入统一监控和收尾，取消使用 runtime_jobs。
3. 成功后调用 `{"action":"read","job_id":"…","offset":0,"limit":1}` 读取片段正文和字词时间戳。按 next_offset 翻页。完整 transcript.txt、segments.jsonl 与 summary.json 在本工作区 Output/speech_transcribe/<job_id>/。

模型位于 MCP 的 MODELS 目录。语音工作进程使用 .local/runtimes/qwen3-speech-006 下的独立 Windows Python 3.11 环境，通过本地 PyAV 解码媒体，按剩余显存为两个模型分别选择 CUDA 或 CPU。作业结束后卸载模型。依赖版本见 requirements.lock，可在新机器为该解释器用 uv pip install --python 指定路径并通过 -r requirements.lock 安装。权重版本与哈希见各自 MODEL_MANIFEST.json。
