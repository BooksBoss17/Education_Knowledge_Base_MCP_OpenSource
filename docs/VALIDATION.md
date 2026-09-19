# 本次发布验证范围

验证日期：2026-09-13。基底：BeMarkdown 0.2.0、MCP 适配层 0.2.0；同日将适配层更新至 0.4.0（见下节）。

## 适配层 0.4.0 更新

`TOOLS/education_mcp/` 由 0.2.0（7 工具）更新为 0.4.0（11 工具），并新增
`SKILLS/textbook-import/SKILL.md`：

- 新增 `bemarkdown_vision` 视觉模型门禁、`textbook_organize` 教材整理、
  `bemarkdown_convert_image` 段落图片局部转换、`bemarkdown_review_context`
  上下文语义修复；`bemarkdown_read`/`bemarkdown_source`/`bemarkdown_review`
  相应扩展。
- 六个适配层 Python 文件与内部正式发布（release id `mcp-si-v1`，0.4.0，
  其发布清单记录 101 项测试与一次真实转换验证）逐字节一致；
  `install.py`、`launch.py`、`workspace_manager.py`、`workspace_template.json`
  与基底原有文件本就相同，未改动。转换器 wheel、模型清单与运行时不变。
- 本次更新在本基底上执行的验证：`scripts/verify_release.py` 全部通过
  （含用户路径扫描、文件哈希清单、wheel 与 SOURCE 一致性、模型计划与权威
  清单一致）、`tests/test_downloads.py` 通过、全部 Python 文件可编译。
  2026-09-19 起另由公开仓库的 MCP SDK 状态检查覆盖 0.4.0 真实 stdio 会话
  （见下节）；教材导入的实机验收记录保留在内部发布（2026-09-13 教材导入
  验证），其验收限制（不宣称全语料准确率）继续适用。
- `SKILLS/textbook-import/SKILL.md` 与内部正式发布逐字节一致，未改动内容。

## 0.4.0 公共 MCP SDK 会话验收（2026-09-19）

公开仓库新增 `tests/test_mcp_sdk_session.py` 与
`.github/workflows/public-mcp-sdk.yml`，将此前只保留在 0.2.0 发布记录中的 MCP
SDK 会话检查变成 pull request、`main` push 与手动触发都会执行的状态检查。

自动化验收固定在当前官方稳定版 `mcp==2.2.0`、GitHub
`windows-latest`、CPython 3.11 上执行。测试另建隔离 venv，只安装
`requirements-core.lock` 与公开发布 wheel；服务必须从
`TOOLS/education_mcp/launch.py` 真实启动，因此仍会执行适配层自己的已安装
runtime/wheel 身份核对，而不是直接导入 `server.py` 绕过启动链。

状态检查同时建立 `mode="auto"` 与 `mode="legacy"` 两次独立 stdio 会话，并要求：

- 当前 SDK 的自动探测能回退到握手协议，两个会话均协商为 `2025-11-25`；
- `serverInfo` 为 `education-knowledge-base` / `0.4.0`，`tools/list` 精确返回
  11 个公开工具，不允许缺失或额外工具；
- `knowledge_workspace inspect` 在全新临时工作区返回完整框架；
  `bemarkdown_info` 返回 `installed_runtime.verified=true`，且公开 wheel 身份一致；
- 在视觉验证前，对 8 个受门禁保护的工具各发起一次真实 SDK 调用，全部必须
  以 `VISION_MODEL_REQUIRED` 拒绝，证明新增教材整理、局部图片转换、上下文修复
  等入口已实际经过 0.4.0 路由，而不只是出现在工具清单中；
- `bemarkdown_vision` 的 `challenge` 必须经 SDK 返回挑战元数据和一张可解码的
  PNG 图片。CI 不读取服务内部答案，也不自动绕过视觉门禁。

该自动检查的 GitHub Actions workflow/job 名称为 `public-mcp-sdk / sdk-session`。
仓库的分支规则应将它设为 `main` 合并所需状态检查；若未配置分支保护，workflow
仍会运行，但 GitHub 不会阻止管理员合并失败的提交。

这项检查补的是**公开发布协议与启动链验收**，不替代 GPU/模型推理验收。需要
发布 0.4.x 时，完整 Windows+GPU 环境仍应由真实视觉模型查看挑战图片并完成
`verify`，随后至少用 `tests/make_fixtures.py` 生成的公开样本完成一次实际转换并
核对 `status`、`read` 与原始来源；若发布声明覆盖教材导入，则还应在同一次
已验证会话中走到 `textbook_organize preview`，并实际查看全部待入库图片。

## 已验证（基底 0.2.0 发布时）

| 检查 | 结果与范围 |
| --- | --- |
| 独立 Python 运行环境 | 在新的 venv 中安装固定依赖，未借用原环境的 site-packages；完整 doctor 返回 `READY_FULL` |
| 根目录 `setup.ps1` | PowerShell 语法检查及真实入口执行通过；复用已校验模型，在独立运行环境重新安装并检查成功 |
| 系统组件来源 | Python 3.11、LibreOffice、VC++ 的三个官方 winget 包 ID 均查询成功 |
| 模型落地 | 13 套模型从本机只读缓存复制到独立部署目录，按文件 SHA-256 核验 |
| 官方网络下载 | 对全部 13 个固定模型版本各下载一份真实小配置；另下载约 84 MB 的 ch_SVTRv2 权重，覆盖分块下载与最终哈希检查，共 14 项通过 |
| MCP 协议 | 使用官方 Python MCP SDK 启动真实 stdio 服务，验证全部 7 个工具、状态查询和工作区检查 |
| 原生 DOCX | 自建正文、OMML 分式与表格输入转换成功；原生分式保留为 LaTeX |
| 原生 PDF | 自建文字、表格、运动示意图输入转换成功 |
| 扫描 PDF | 将自建页面光栅化，真实模型转换成功并通过产物结构校验 |
| 截图 DOCX | 自建整页图片插入 Word，默认 `DOCX_IMAGE_CONTENT` 路线转换成功；SDK 获取原始嵌入图成功 |
| agent 修订接口 | 过期哈希被拒绝；提供源证据的唯一替换成功；原始 Markdown 保持不变，修订版及日志可读取 |
| 源码对应 | wheel 内包文件与公开源码逐字节比对；原版 Python 代码只改变 FormulaNet 清理下载缓存后的指纹常量，未修改推理逻辑 |
| 分发校验 | 随包脚本检查文件 SHA-256、wheel 与源码、模型计划与权威清单、许可文件、个人路径以及未混入环境／模型权重／教材 |

## 已知条件和未验证范围

测试主机为 Windows x64、NVIDIA RTX 4070 12 GB，使用 10gb 资源档。主机原先已有 Python、LibreOffice、VC++ 与 NVIDIA 驱动；本次没有在完全空白 Windows 或不同 GPU 上重新安装全部系统组件。官方安装器已接入，首次安装时可能需要 UAC。

初次安装遇到 Torch 第三方许可证路径过长，已用 Windows 扩展路径执行安装解决，未裁去许可证。后续完整安装及根入口复测通过。

冻结环境保留已实测的 cuDNN 9.9.0.52，而 Paddle 元数据指定 9.5.1.17；安装器只接受这一个已知冲突，其他 pip 冲突会失败。doctor 验证当前环境，但不保证任意未来驱动或不同机器兼容。

首次 SDK 检查中，截图 Word 的测试脚本误请求分页图，随后按实际原图路线纠正并复用已成功任务完成来源核验。默认路线不调用 LibreOffice 分页；本轮四类转换验证不等于单独验证了可选分页路线。

这是一组原创最小功能样本，不是教材／试卷全集的独立准确率评测。任务成功、产物结构合法、三路 OCR 一致均不能证明 90%／95% 内容正确率，也不构成统一秒／页性能承诺。公式或表格仅保留原图时仍属于待识别。

本次未测量云端 agent 修复耗时；Qwen3-Embedding-4B、WeMM-Embedding-4B 已登记并验证模型文件，尚无向量化或检索 MCP 工具，也未据此认证其推理能力。

## 部署者可以复核

在安装之前运行：

```powershell
python .\scripts\verify_release.py
python -m unittest discover -s tests -p test_downloads.py -v
```

安装后允许本地状态及下载模型存在时，使用 `python .\scripts\verify_release.py --installed`。完整安装会生成 `.local/doctor.json`、`.local/pip-check.json`、`.local/setup-state.json` 和分次安装日志。转换任务的证据位于工作区 `tmp/bemarkdown/.mcp/jobs/`。

可用 `tests/make_fixtures.py` 生成原创测试输入，再通过 MCP 客户端提交转换。根 `RELEASE_MANIFEST.json` 记录交付文件哈希；它用于完整性检测，并非独立的发行者数字签名。含本机绝对路径的内部原始日志不随公开包分发。
