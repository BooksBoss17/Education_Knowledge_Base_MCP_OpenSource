# MCP 运行管理与资源预算

教育 MCP 的 `bemarkdown_convert` 和 BeMarkdown 原生入口 `TOOLS/bemarkdown/convert.py` 使用共享持久队列。同输入、版本和配置的活动请求复用任务；跨客户端重任务按当前资源配置准入，断开连接不代表取消。

`runtime_jobs` 支持 list/status/adopt/cancel。恢复时先查原任务；取消仅针对指定任务及其拥有的进程。错误或中断保留证据，不自动重新执行已成功的推理。

当前转换使用版面、公式、表格、文字四个 API。注册表提供 `education-convert` 和 `native-convert` 两个转换程序。API-only 安装使用外部模型服务，由使用者配置模型的资源预算。
