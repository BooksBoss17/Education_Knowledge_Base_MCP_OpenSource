# 视频检索、下载与转写

可选组件支持 Windows CPython 3.11。先按仓库首页安装 MCP，再按实际使用需求从仓库根目录执行以下命令。

- B 站标题及作者检索使用本机 Microsoft Edge。安装 Edge、Node.js 20 或更新版和 npm 后运行 `python scripts/setup_video.py --browser`。随包提供 Playwright MCP 的锁定依赖。
- 下载供游客访问的视频并提取关键帧，需要可信发行源的 Windows FFmpeg 和 aria2 执行文件。运行 `python scripts/setup_video.py --ffmpeg <ffmpeg.exe> --aria2 <aria2c.exe>`，程序安装到本机 .local 目录并记录 SHA-256。选帧命令的 --ffmpeg 可指定 .local/video-bin/ffmpeg.exe。
- 语音转写及字词对齐使用 [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 与 [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)。请在两份官方仓库按 [随包模型清单](../speech_transcribe/model-manifests) 的 upstream_revision 下载列出的文件，放入同一父目录下的对应模型名子目录，再运行 `python scripts/setup_video.py --speech-model-root <模型父目录>`。程序核对每个文件的 SHA-256 并安装独立的语音运行环境。操作依赖锁包含 Windows CUDA PyTorch 轮子。重连 MCP 后调用 speech_transcribe。

`bilibili_search` 可按标题查视频、按 UP 主查投稿。收到站点限制时会保留取得的结果并标记未取全。`bilibili_download` 使用 aria2 和 FFmpeg，可核对 BV 与分 P、查看下载状态和核验成品。桌面程序、账户登录和模型权重属于本机独立载荷。视频入库使用[教学视频导入 Skill](../../SKILLS/instructional-video-import/SKILL.md)。
