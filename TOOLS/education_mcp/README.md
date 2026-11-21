# Education Knowledge Base MCP 0.7

正式转换使用 BeMarkdown 本地 API。支持 PDF、DOCX 与单帧图片，结果绑定原文件哈希。先完成 bemarkdown_vision 图片验证，再提交转换并轮询同一作业。通过 bemarkdown_source 查看 PDF 原页、Word 嵌图和独立原图。转换报告和 handoff_summary 记录待核内容。使用 bemarkdown_review 时提供当前 SHA、唯一替换和已观察的来源依据。

教材导入调用 textbook_import，按 [现行教材说明](TEXTBOOK_IMPORT.md)完成准备、逐项原页复核、预览、批准和发布。

安装和迁移按 [BeMarkdown 说明](../bemarkdown/README.md)配置环境。使用 launch.py 并显式设置 MCP 根、业务工作区和允许的输入根。新连接使用当前接口。

## 全程资源监控

在任务的 MCP 连接调用 `runtime_monitor(action="start", interval_seconds=1)`，保存返回的 id。相同工作区的连接共享一份采样；重复 start 返回已有记录。默认最多运行 24 小时，duration_seconds 可配置。需要跨暂停分段统计时先 stop，续接后再 start。已经有其他任务共享采集时，由发起采集的一方负责停止。

- status 查看采集进程状态、最后采样时刻与 SQLite 路径。summary 查看整段资源与逐工具/动作的响应时间。calls、samples 通过 offset 和 limit 读取逐次调用、系统采样及模型工作进程。
- models 传成功转换的 job_id，可再传 capture_id、offset、limit。逐请求返回角色、实际 model_id、request_id、批大小、客户端响应时间、可用的服务端推理耗时、Torch 显存分配/保留峰值，以及请求期间采到的资源。stop 后仍可查历史。
- GPU 指标为本地 NVML GPU 0 的整卡利用率和已用显存，包含并行负载。主机 CPU 为全机百分比，物理内存为 total 减 available。进程 CPU 用一个逻辑核为 100%，内存以字节计。MCP 进程记录调用首尾 RSS，已识别的本地模型 worker 及其子进程合计采样 RSS 和 CPU（共享内存可能重复计入 RSS）。远端模型的资源以服务端提供的指标为准。calls 同时返回调用首尾的整卡和主机快照；短调用可能处于两次周期采样之间，缺测项为 null。工具耗时为服务端处理时间，异步提交和后台模型请求分别计时。
- 数据库位于业务工作区 tmp/runtime-monitor/captures.sqlite3。采集持续到 stop、时限或采集进程退出。返回的时间戳是 Unix 秒。按原始采样时间评估间隔；设备不提供 NVML 数据时仍可查看 CPU 和内存。整个宿主对话和第三方 MCP 的工具调用不在本服务的调用记录内。
- Windows 可用 GPU 性能计数器时，calls 和 models 还返回进程或模型进程树在本机各 GPU 引擎中的最忙引擎利用率和专用显存采样峰值；samples 保留按 PID 的对应记录。GPU 引擎和进程专用显存可能跨本机适配器，与 NVML 的 GPU 0 整卡数据分别标记。

## 按任务统一收尾

runtime_jobs 支持 begin、inspect、end、resume。主会话预先生成并保存 task_run_id（32 位小写十六进制），begin 启用持久登记和工作区采集。在每次业务调用中传该 ID，子 agent 使用相同 ID。end 的 outcome 为 complete、pause 或 cancel，返回 CLOSING 时后台管理器继续排空；inspect 返回任务状态、作业、在途调用数量与终态回执。pause 等待在途工作自然完成，resume 开启新的采样段。取消独占作业通过管理器已有的作业取消路径执行。多个任务共用的工作区采集由最后一个持有者释放；开始任务之前已存在的外部采集归其原发起者。宿主进程和原生子 agent 由其宿主管理。单次业务调用可独立运行；整任务使用 task_run_id 登记。
