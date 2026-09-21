# Education Knowledge Base MCP · 基底版本

面向教学知识库的本地 MCP。当前主要工具 **BeMarkdown** 将 DOCX、PDF 转成保留来源信息的中间 Markdown，供 agent 继续复核和整理。

2026-09-22 更新：[图形残余归属、图题关联与保守顶部正文裁切](docs/FIGURE_OWNERSHIP_20260922.md)。

自有代码采用 **AGPL-3.0-only**；第三方组件保留原许可证。完整条款见 [LICENSE](LICENSE)、[许可核查](docs/LICENSE_AUDIT.md) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。

## 当前可以做什么

- 自动初始化教学知识库工作区，检查目录结构，分页列举文件；按明确请求修复目录或调整框架。
- 转换原生 Word、扫描截图型 Word、原生 PDF 和扫描 PDF。
- 解析正文、公式与表格，将示意图保存成图片并在 Markdown 中引用。
- 提供异步转换任务、状态查询、日志、原文件证据和待复核候选项。
- 让 agent 根据原文证据执行精确替换，保留原始 Markdown 和修改记录；对三模型文字候选支持上下文语义修复（字数变化受正负 1 字限制并单独记录推断依据）。
- 支持显式声明的**教材导入**：先按 SHA-256 备份原件，转换后由 `textbook_organize` 完成图片复核、章节边界确认、装饰清理与标题转写，分章发布到 `KNOWLEDGE_BASE/TEXTBOOKS/`；随包提供 `textbook-import` skill 编排整个流程。
- 正文段落图片可调用局部转换复用同一模型与 GPU 队列，不重新转换整份文档。
- 共享登记 13 个模型，包括 Qwen3-Embedding-4B 和 WeMM-Embedding-4B。两套 Embedding 模型目前是可下载资源，检索、索引及对应 MCP 工具作为后续扩展。

输出定位是**中间产物**。按教学资料类型删减、拼接、整理成知识库条目的流程，可继续通过 Skills 或新工具实现。

## 视觉模型要求

本 MCP 仅支持可接收图片的视觉模型。首次使用（以及切换模型、重连或空闲超过 30 分钟后）必须先完成 `bemarkdown_vision` 挑战：服务返回一张含六个符号的图片，调用方实际读出符号并以 `action="verify"` 提交答案。验证通过前，转换、读取、复核与教材整理工具均返回 `VISION_MODEL_REQUIRED`。该检查验证的是实际图片可达性，不是可信的模型身份。

## 部署

当前完整技术路线验证于 **Windows x64、CPython 3.11、NVIDIA GPU**。建议独立显存大于 10 GiB；6–10 GiB 使用较保守调度。复杂页面仍可能触及显存限制。需要兼容 CUDA 13 系列 PyTorch wheel 的 NVIDIA 驱动；安装脚本获取用户态运行库，显卡驱动由操作系统管理员维护。

推荐预留 **100 GB 可用磁盘空间**用于模型、Python 环境、下载缓存和转换输出。完整模型约二十多 GB，实际占用还包括运行库。只有转换需求时可选择 `conversion`，跳过两套较大的 Embedding 资源。

1. 下载或克隆此文件夹，保持文件完整。GitHub 仓库不内置大型权重和 NVIDIA 运行库。
2. 在 Windows PowerShell 中进入此目录，运行：

```powershell
.\setup.ps1
```

脚本会先提示外部许可证，再检查／安装 Python 3.11、LibreOffice 和必要的 Visual C++ 运行库，下载并校验模型，创建隔离环境，安装固定依赖，运行完整 GPU 检查，初始化工作区并输出 MCP 配置。系统组件自动安装使用 Windows `winget`，可能需要用户允许 UAC。没有 winget 时可从官方站点安装这些前置组件后重试。

已阅读并接受外部条款、需要无人值守部署时：

```powershell
.\setup.ps1 -AcceptExternalLicenses
```

只安装当前转换所需的 11 个模型：

```powershell
.\setup.ps1 -AcceptExternalLicenses -Models conversion
```

可用 `-Workspace` 指定工作区，`-PythonExe` 指定现有 Python 3.11，`-RuntimeDirectory` 指定隔离环境，`-ModelCache` 指定已有共享 `MODELS` 目录。导入缓存也会检查每个文件的大小和 SHA-256。中断后重新执行可复用已完成的下载。

3. 安装完成显示 `READY_FULL` 后，将生成的 **`.local/mcp-client.json`** 内容加入 MCP 客户端。它已包含本机路径和所选 Python 环境，无需复制他人的路径配置。
4. 需要转换工作区以外的文件时，在配置的 `args` 中追加 `--input-root` 和该资料目录。可重复添加多个目录。

模型推理在本地完成。MCP 服务本身不绑定某个 agent 平台或云模型；复核所用 agent 由客户端配置。

## 目录

| 目录 | 用途 |
| --- | --- |
| `MODELS/` | 共享模型登记及权威清单；运行安装脚本后补齐模型文件 |
| `TOOLS/bemarkdown/` | BeMarkdown wheel、依赖锁定和运行清单 |
| `TOOLS/education_mcp/` | 标准 stdio MCP 适配层（0.4.0，11 个工具）、工作区管理 |
| `SOURCE/bemarkdown/` | 与发布 wheel 对应的可修改源码及构建配置 |
| `SKILLS/` | 工作流编排技能（含 `textbook-import` 教材导入流程） |
| `scripts/` | 自动安装、模型下载、校验等部署脚本 |
| `docs/` | 部署、许可、技术路线和扩展说明 |
| `licenses/` | 上游许可证据和组件清单 |

此仓库是一份可扩展基底。可以让 agent 设计新的 MCP 工具、共享模型接口或插件；具体方法见 [扩展指南](docs/EXTENDING.md)。

## MCP 接口

适配层版本 0.4.0，共 11 个工具。首次使用必须先通过 `bemarkdown_vision` 验证（见上文"视觉模型要求"）。

| 工具 | 用途 |
| --- | --- |
| `bemarkdown_vision` | 视觉模型门禁：领取六符号图片挑战并验证；未通过前其余工具锁定 |
| `knowledge_workspace` | 初始化后的检查、文件盘点、指定目录修复、框架配置 |
| `bemarkdown_info` | 查询转换环境及工作流信息 |
| `bemarkdown_convert` | 提交 DOCX/PDF 转换，立即返回任务 ID；`material_type="textbook"` 先备份原件并返回教材导入 skill |
| `bemarkdown_status` | 查询或短时等待任务状态 |
| `bemarkdown_read` | 读取 Markdown、报告、资源及复核任务（`handoff_summary` 汇总三模型文字候选、表格任务与图片区域复核） |
| `bemarkdown_source` | 按来源位置获取原文件证据；截图 Word 可查保留分页，小字可用 `region` + `dpi=288` 放大 |
| `bemarkdown_convert_image` | 将含完整正文段落的输出图片送入现有转换管线局部转换，保留父子任务与来源哈希 |
| `bemarkdown_review_context` | 返回被标记正文段及其前 3 后 1 上下文与当前哈希，供语义修复使用 |
| `bemarkdown_review` | 基于原文证据的唯一匹配替换；支持 `context_semantic` 语义修复（整段替换，字数变化正负 1 字内，记录推断依据） |
| `textbook_organize` | 教材整理：图片清单/缩略图检查、分章预览与发布到 `KNOWLEDGE_BASE/TEXTBOOKS/` |

转换产物写入 `<工作区>/tmp/bemarkdown/`。每个产物通常由 `document.md`、`conversion_report.json` 和引用图片组成；启用证据导出时还有调试／来源信息。修订后增加 `document.reviewed.md` 与 `agent_review.json`。

教材导入流程：`bemarkdown_convert(material_type="textbook")` → 按返回的 `textbook-import` skill 复核中间 Markdown 与全部图片 → `textbook_organize` 确认章节边界、清理无内容装饰、转写纯标题图片 → `preview` 校验后 `publish` 分章入库。示意图与含正文/公式/表格的图片一律保留引用。

运行记录位于 `<工作区>/tmp/bemarkdown/.mcp/jobs/<任务ID>/`，包括状态、标准输出、错误日志、校验结果和 agent 交接信息。安装日志在本仓库 `.local/logs/`。

## 技术与验证说明

详见 [BeMarkdown 完整技术路线](docs/BEMARKDOWN_TECHNICAL_SCHEME.md)。本版本保留三模型文字 OCR、FormulaNet 公式识别、表格专用模型、示意图区域保护与来源复核链路；适配层为 0.4.0，新增视觉模型门禁、教材导入与分章整理、上下文语义修复和段落图片局部转换。适配层更新内容与验证范围见 [发布验证](docs/VALIDATION.md) 的"适配层 0.4.0 更新"一节。

模型一致或转换任务成功，均不等于内容完全正确。复杂公式、表格、阅读顺序和图文边界仍可能需要复核；历史优化结果不能当作全语料 90%／95% 正确率保证。截图型 Word 的端到端速度也受图片数量、版面复杂度、模型启动及 agent 复核影响。验收范围见 [发布验证](docs/VALIDATION.md)。

## 发布到 GitHub

本文件夹可作为新的仓库根目录。保留 `.gitignore`、`.gitattributes`、许可证、源码和第三方声明；不要提交 `.local`、工作区、运行日志、下载缓存或模型权重。使用全新的 Git 历史，不要复制内部开发仓库的 `.git`。

执行安装、测试或二次构建后，可运行随包校验脚本检查发布文件。模型是独立下载资源，Git checkout 本身不能恢复权重。
