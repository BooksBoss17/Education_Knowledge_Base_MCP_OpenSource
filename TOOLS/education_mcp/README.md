# Education Knowledge Base MCP 0.6

转换入口使用 BeMarkdown 0.4 本地 API 线路。保留 PDF、DOCX、独立图片、持久任务查询、原文/资源查看、精确修订和教材分章能力。转换时不自动运行旧图片分类；显式筛选接口和旧转换包的审阅兼容仍保留。

首次连接先完成 `bemarkdown_vision` 的六符号图片验证。调用 `bemarkdown_convert` 后轮询同一个 job；断开连接不会取消持久作业。`runtime_jobs` 可在当前工作区查队列、认领和显式取消，提交前先检查是否已有任务。

使用 `bemarkdown_read(kind="original"/"report"/"handoff_summary")` 查看结果。新线路没有三模型分歧候选，零候选不表示没有识别错误。通过 `bemarkdown_source` 查看原 PDF 页、原图片或 Word 的嵌入图片；新线路使用原图直接转录，不生成 Office 分页副本，`source_view="pagination"` 仅用于确有该证据的旧包。

`bemarkdown_review` 基于当前 SHA、唯一文本片段及具体来源依据安全修订。扫描表格、空识别及不支持的 Word 图片外观保留资源并明确待核；包校验成功不等于内容准确率认证。

教材导入使用 `textbook_import` 和随包提供的教材 Skill。可用已完成转录包建立候选，查看原页与待审项，审查后按版本执行 preview、approve 和 publish。旧的 `textbook_organize` 入口继续可用。[教材导入 Skill](../../SKILLS/textbook-import/SKILL.md)

安装或迁移时先按 `../bemarkdown/README.md` 激活环境，再运行 `launch.py --mcp-root <MCP目录> --workspace <业务工作区> --input-root <允许的输入目录>`。启动器依据安装收据选择环境，不依赖开发目录。部署更新后新连接使用当前版本；已有连接的内存信息可能仍显示旧适配器版本。

## 全程资源监控

在任务的 MCP 连接调用 `runtime_monitor(action="start", interval_seconds=1)`，保存返回的 id。相同工作区的连接共享一份采样；重复 start 返回已有记录。默认最多运行 24 小时，duration_seconds 可配置。需要跨暂停分段统计时先 stop，续接后再 start。已经有其他任务共享采集时，由发起采集的一方负责停止。

- status 查看采集进程状态、最后采样时刻与 SQLite 路径。summary 查看整段资源与逐工具/动作的响应时间。calls、samples 通过 offset 和 limit 读取逐次调用、系统采样及模型工作进程。
- models 传成功转换的 job_id，可再传 capture_id、offset、limit。逐请求返回角色、实际 model_id、request_id、批大小、客户端响应时间、可用的服务端推理耗时、Torch 显存分配/保留峰值，以及请求期间采到的资源。stop 后仍可查历史。
- GPU 指标为本地 NVML GPU 0 的整卡利用率和已用显存，包含并行负载。主机 CPU 为全机百分比，物理内存为 total 减 available。进程 CPU 用一个逻辑核为 100%，内存以字节计。MCP 进程记录调用首尾 RSS，已识别的本地模型 worker 及其子进程合计采样 RSS 和 CPU（共享内存可能重复计入 RSS）。远端模型的资源以服务端提供的指标为准。calls 同时返回调用首尾的整卡和主机快照；短调用可能处于两次周期采样之间，缺测项为 null。工具耗时为服务端处理时间，异步提交和后台模型请求分别计时。
- 数据库位于业务工作区 tmp/runtime-monitor/captures.sqlite3。采集持续到 stop、时限或采集进程退出。返回的时间戳是 Unix 秒。按原始采样时间评估间隔；设备不提供 NVML 数据时仍可查看 CPU 和内存。整个宿主对话和第三方 MCP 的工具调用不在本服务的调用记录内。
- Windows 可用 GPU 性能计数器时，calls 和 models 还返回进程或模型进程树在本机各 GPU 引擎中的最忙引擎利用率和专用显存采样峰值；samples 保留按 PID 的对应记录。GPU 引擎和进程专用显存可能跨本机适配器，与 NVML 的 GPU 0 整卡数据分别标记。

## 按任务统一收尾

runtime_jobs 支持 begin、inspect、end、resume。主会话预先生成并保存 task_run_id（32 位小写十六进制），begin 启用持久登记和工作区采集。在每次业务调用中传该 ID，子 agent 使用相同 ID。end 的 outcome 为 complete、pause 或 cancel，返回 CLOSING 时后台管理器继续排空；inspect 返回任务状态、作业、在途调用数量与终态回执。pause 等待在途工作自然完成，resume 开启新的采样段。取消独占作业通过管理器已有的作业取消路径执行。多个任务共用的工作区采集由最后一个持有者释放；开始任务之前已存在的外部采集归其原发起者。宿主进程和原生子 agent 由其宿主管理。未传任务 ID 的旧调用仍走原有路径。
