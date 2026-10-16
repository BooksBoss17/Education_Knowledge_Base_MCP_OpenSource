# 受管转换与内置图片筛选

本版将已在本地安装验收的 BeMarkdown 转换、资源管理、提速选项和图片分类整合到公开包。公开安装适配继续使用 `.local/runtime-selection.json` 与独立的语义模型运行时登记，不包含开发机器路径、模型权重、教材样本、私有模板或开发验收日志。

## 安装与升级

1. 保留自己的工作区和 `.local` 配置，取得完整新版仓库；先运行 `setup.ps1 -VerifyOnly` 检查交付文件。
2. 按主 README 运行 `setup.ps1` 安装/更新转换环境。程序将新 wheel 与运行环境绑定，原有可用第三方依赖可复用；完成后重启 MCP 客户端。
3. **默认内置图片筛选需要 B200、Qwen3.5-9B 以及已登记的独立语义运行环境。** 普通 `setup.ps1` 不会自动完成这套额外运行环境的安装。按 [语义运行环境说明](SEMANTIC_RETENTION.md) 下载模型并执行 `scripts/install_semantic_retention.py --runtime-only --register-runtime ...`；已有登记会沿用公开版指针，不能假定内部机器上的固定 runtime ID 在另一台电脑存在。
4. 只有 OCR 环境时，可以从受管原生 CLI 显式传 `--no-image-screening`。默认 MCP 转换需要先准备完整筛选依赖。分类依赖或资源不足时保留原始包并报告实际状态，不能将未运行分类算作成功筛选。

当前包没有附带 `textbook-import` skill，与当前内部交付一致。普通 PDF/DOCX/图片转换可用；不要在缺少匹配编排包时请求 `material_type=textbook`。旧说明中的教材编排能力属于有配套 skill 的部署，不代表本包已附带该文件。

## 任务管理

`runtime_jobs` 提供持久任务查询、状态、取消和恢复入口。转换返回任务 ID 后查询原任务；客户端断开不代表模型进程已停，不要重复提交同一输入/参数/输出组合。独立语义 MCP 仍可调用，分类结果接口保持。

线程/并发、任务排队、RAM/系统提交余量及整卡显存受统一策略控制。重模型按阶段串接，新分类阶段重新检查资源。不要通过降低保护线、关闭其他程序或放大并发来跳过资源拒绝。具体预算由当前设备和已安装策略决定。

## 图片与待审交接

- OCR、公式和表格转录证据图：保留，跳过普通图片分类。
- 示意图：保留；带标签不意味着它应改成纯文字。
- 标题和正文文字图：先保留图片，生成待转录任务，验证转录后才可替换引用。
- 经冻结语义判定与用途观察共同确认可排除的杂图：只从派生包去除，原文件和原始转换包不变。
- 不确定、冲突和裁框风险：保留并按实际范围交给 agent。

转换返回 `image_screening`；handoff 增加 `image_screening_summary`、`image_screening_tasks`。`COMPLETE` 与资源/来源延期状态分开，分类通过不代表文字、阅读顺序或整页覆盖已验收。普通 DOCX 没有合格 PDF 上下文时保留图像并明确延期分类。

## 原生参数

`TOOLS/bemarkdown_optimized/convert.py` 支持 `--layout-batch-size 1..4`、`--batch-source`、`--repetition-guard`、`--no-image-screening`、`--screen-existing <已完成OCR包>`。后者必须保持源文件 SHA 与完整包指纹一致，输出使用新目录。详细说明：[提速与批处理](../TOOLS/bemarkdown_optimized/PERFORMANCE.md)、[图片筛选](../TOOLS/bemarkdown_optimized/IMAGE_SCREENING.md)、[运行管理](../TOOLS/runtime_manager/README.md)。

早停仅将异常重复标为未决，不把截断输出当成正确转录。小批窗口只复用约定阶段；新配置需单独记录执行指纹与评测范围。

## 质量与交付范围

本版保留原模型二进制和下载身份，新增图片用途观察尚未完成全量分类标准集认证。旧冻结 K/D/U 的质量数字仅适用于原数据、输入和策略，不能直接推广到新增整链。公开安装/接口检查与内容正确率是不同结论，速度和资源受硬件、缓存、模型准备和输入影响。

公开包只提供运行/安装说明、必要清单、源代码、许可和聚合能力边界。开发报告、原始预测、教材图文、金标、逐图裁决、主机配置及回滚证据均不在仓库中。
