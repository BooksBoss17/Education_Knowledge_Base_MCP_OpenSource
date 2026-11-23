---
name: instructional-video-import
description: 将教学视频录入教育知识库。接收本地视频、B 站 BV 号、标题或 UP 主名称，归档原视频与时间戳稿件，选取关键帧并整理带来源时间的课堂笔记。用于明确要求导入或入库的任务。
---

# 教学视频导入

从主会话取得知识库工作区与 task_run_id。单独执行时按 education-import-orchestrator 的工作流使用 runtime_jobs(begin/inspect/end) 登记和收尾。每次 MCP 业务调用绑定该 ID。以一个视频的一条分 P 为一个导入单元。

1. **取得视频。** 本地文件直接使用用户指定的原件。有 BV 号或 B 站视频 URL 时保留 BV 和分 P（含 URL 的 p），用 bilibili_download 的 inspect 核对标题、UP 主和分 P。仅有标题、昵称或 UID 时，先用 bilibili_search 核对候选身份并取得 BV 号。重名或多个合理候选交给用户选择。仅给作者名时先列候选投稿，确认范围后再下载；明确指定整批时按 UID 分页、BV 去重并对账。412 或未取全时保存断点，报告已取得的范围。用 bilibili_download(start/status/verify) 在当前工作区下载并核验；保存实际 video 路径及 job_id。BiliDown 成功回包的作业号在 data.job_id，状态在 data.state。下载成功时取 data.result.video；speech_transcribe 的 job_id 和 state 在顶层。
2. **转写。** 调用 speech_transcribe(start, source=本地实际路径, timestamps=true)，保存 job_id，用 status 轮询到 SUCCEEDED。失败或中断先查原 job。按需用 read 分页。ASR 稿件采用第一音轨时间，课堂笔记和截图采用视频播放时间。遇到音轨延迟起播、长前导静音、剪辑跳时或音画明显不同步时，回看原视频，注明时间偏移并按实际播放时间选帧。无语音的区段可以明确标记；缺少字词对齐时仅使用实有的片段时间。
3. **归档。** 使用本 skill 的 scripts/video_import.py（Python 3.11+）。运行 `prepare --workspace <知识库> --source <视频> --job-id <转写作业> --title <标题>`。程序核对源文件与转写摘要 SHA-256，返回 archive、draft、package_id 等实际路径。B 站已归档的视频原地配稿；本地视频复制并校验。原视频和 transcript.md 位于 Original_Backup/INSTRUCTIONAL_VIDEOS 下同一视频目录。时间戳原稿保持 ASR 原文。
4. **选帧与写作。** 阅读整份转写，按讲解内容选择板书完成、公式出现、装置变化、示例结论等关键时刻。用 `frames --workspace <知识库> --id <package_id> --seconds 12.5 38 --ffmpeg <ffmpeg.exe>` 提取到暂存目录。可用本机 MCP 的 TOOLS/bilidown/bin/ffmpeg.exe。实际打开截图核对清晰度、事件与时间；按需另选时间，覆盖无语音但有教学信息的画面。每批最多查看四张。对明显 ASR 错词，通过音画复核后在笔记中表述；无法判断时交主会话解决后再发布。
5. **发布。** Agent 写 UTF-8 JSON 计划，顶层为 sections 数组。每项有 start（秒）、heading、body（Markdown 正文）、frames 数组。每个引用是 {"file":"脚本返回的 JPG 文件名","caption":"画面中实际可见的教学证据"}。从导入开头到结尾按知识内容分节，写清问题、条件、讲解过程与结论，区分原课与整理者补充。以视频实际讲授内容为准，节选应注明覆盖范围。未讲解的试题只记录课堂呈现的信息；图中无法辨清的公式下标或数字保留原图，概括可确认的内容，不能按理论反推后当作原文。每幅采纳的图都要审阅。运行 `publish --workspace <知识库> --id <package_id> --plan <计划.json>`。程序在 KNOWLEDGE_BASE/INSTRUCTIONAL_VIDEOS 下为该视频发布独立目录，含 lesson.md 与 frames/，只带入笔记引用的截图。回读成品、核验相对链接，给出原件、稿件和笔记位置。已有不同版本时保留现场并报告冲突；需要修订时先保存已有版本供核查，不删除成品来绕过冲突检查。
