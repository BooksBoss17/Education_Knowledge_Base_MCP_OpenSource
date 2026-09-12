# Education Knowledge Base MCP · 基底版本

面向教学知识库的本地 MCP。当前主要工具 **BeMarkdown** 将 DOCX、PDF 转成保留来源信息的中间 Markdown，供 agent 继续复核和整理。

自有代码采用 **AGPL-3.0-only**；第三方组件保留原许可证。完整条款见 [LICENSE](LICENSE)、[许可核查](docs/LICENSE_AUDIT.md) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。

## 当前可以做什么

- 自动初始化教学知识库工作区，检查目录结构，分页列举文件；按明确请求修复目录或调整框架。
- 转换原生 Word、扫描截图型 Word、原生 PDF 和扫描 PDF。
- 解析正文、公式与表格，将示意图保存成图片并在 Markdown 中引用。
- 提供异步转换任务、状态查询、日志、原文件证据和待复核候选项。
- 让 agent 根据原文证据执行精确替换，保留原始 Markdown 和修改记录。
- 共享登记 13 个模型，包括 Qwen3-Embedding-4B 和 WeMM-Embedding-4B。两套 Embedding 模型目前是可下载资源，检索、索引及对应 MCP 工具作为后续扩展。

输出定位是**中间产物**。按教学资料类型删减、拼接、整理成知识库条目的流程，可继续通过 Skills 或新工具实现。

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
| `TOOLS/education_mcp/` | 标准 stdio MCP 适配层、工作区管理 |
| `SOURCE/bemarkdown/` | 与发布 wheel 对应的可修改源码及构建配置 |
| `SKILLS/` | 后续工作流编排的扩展位置 |
| `scripts/` | 自动安装、模型下载、校验等部署脚本 |
| `docs/` | 部署、许可、技术路线和扩展说明 |
| `licenses/` | 上游许可证据和组件清单 |

此仓库是一份可扩展基底。可以让 agent 设计新的 MCP 工具、共享模型接口或插件；具体方法见 [扩展指南](docs/EXTENDING.md)。

## MCP 接口

| 工具 | 用途 |
| --- | --- |
| `knowledge_workspace` | 初始化后的检查、文件盘点、指定目录修复、框架配置 |
| `bemarkdown_info` | 查询转换环境及工作流信息 |
| `bemarkdown_convert` | 提交 DOCX/PDF 转换，立即返回任务 ID |
| `bemarkdown_status` | 查询或短时等待任务状态 |
| `bemarkdown_read` | 读取 Markdown、报告、资源及复核任务 |
| `bemarkdown_source` | 按来源位置获取原文件证据 |
| `bemarkdown_review` | 基于原文证据进行唯一匹配替换，记录修订 |

转换产物写入 `<工作区>/tmp/bemarkdown/`。每个产物通常由 `document.md`、`conversion_report.json` 和引用图片组成；启用证据导出时还有调试／来源信息。修订后增加 `document.reviewed.md` 与 `agent_review.json`。

运行记录位于 `<工作区>/tmp/bemarkdown/.mcp/jobs/<任务ID>/`，包括状态、标准输出、错误日志、校验结果和 agent 交接信息。安装日志在本仓库 `.local/logs/`。

## 技术与验证说明

详见 [BeMarkdown 完整技术路线](docs/BEMARKDOWN_TECHNICAL_SCHEME.md)。本版本保留三模型文字 OCR、FormulaNet 公式识别、表格专用模型、示意图区域保护与来源复核链路。

模型一致或转换任务成功，均不等于内容完全正确。复杂公式、表格、阅读顺序和图文边界仍可能需要复核；历史优化结果不能当作全语料 90%／95% 正确率保证。截图型 Word 的端到端速度也受图片数量、版面复杂度、模型启动及 agent 复核影响。验收范围见 [发布验证](docs/VALIDATION.md)。

## 发布到 GitHub

本文件夹可作为新的仓库根目录。保留 `.gitignore`、`.gitattributes`、许可证、源码和第三方声明；不要提交 `.local`、工作区、运行日志、下载缓存或模型权重。使用全新的 Git 历史，不要复制内部开发仓库的 `.git`。

执行安装、测试或二次构建后，可运行随包校验脚本检查发布文件。模型是独立下载资源，Git checkout 本身不能恢复权重。
