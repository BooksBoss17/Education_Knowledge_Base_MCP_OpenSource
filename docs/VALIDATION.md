# 本次发布验证范围

验证日期：2026-09-13。基底：BeMarkdown 0.2.0、MCP 适配层 0.2.0。

## 已验证

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
