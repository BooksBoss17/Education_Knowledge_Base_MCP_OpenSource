# Education Knowledge Base MCP 0.6

转换入口使用 BeMarkdown 0.4 本地 API 线路。保留 PDF、DOCX、独立图片、持久任务查询、原文/资源查看、精确修订和教材分章能力。转换时不自动运行旧图片分类；显式筛选接口和旧转换包的审阅兼容仍保留。

首次连接先完成 `bemarkdown_vision` 的六符号图片验证。调用 `bemarkdown_convert` 后轮询同一个 job；断开连接不会取消持久作业。`runtime_jobs` 可在当前工作区查队列、认领和显式取消，提交前先检查是否已有任务。

使用 `bemarkdown_read(kind="original"/"report"/"handoff_summary")` 查看结果。新线路没有三模型分歧候选，零候选不表示没有识别错误。通过 `bemarkdown_source` 查看原 PDF 页、原图片或 Word 的嵌入图片；新线路使用原图直接转录，不生成 Office 分页副本，`source_view="pagination"` 仅用于确有该证据的旧包。

`bemarkdown_review` 基于当前 SHA、唯一文本片段及具体来源依据安全修订。扫描表格、空识别及不支持的 Word 图片外观保留资源并明确待核；包校验成功不等于内容准确率认证。

教材导入说明随工具交付在 `TEXTBOOK_IMPORT.md`，接口会在入队前检查它是否存在。只有用户要求教材入库时才执行 publish/revise；测试和普通转换只生成中间产物。

安装或迁移时先按 `../bemarkdown/README.md` 激活环境，再运行 `launch.py --mcp-root <MCP目录> --workspace <业务工作区> --input-root <允许的输入目录>`。启动器依据安装收据选择环境，不依赖开发目录。部署更新后新连接使用当前版本；已有连接的内存信息可能仍显示旧适配器版本。
