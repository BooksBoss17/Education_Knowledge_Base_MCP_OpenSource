# 独立语义筛图 MCP（冻结版）

版本 semantic-retention-20261008-v1。输入已转换的 BeMarkdown PDF 包、MD 与对应 PDF；输出保留、排除候选和待核保留清单，不删除源文件，不接入已有转换线路。B200 权重沿用原 Release，Qwen3.5-9B Q3_K_M/F16 从固定上游 revision 下载，两者顺序运行。

## 安装

使用 Python 3.11+ 执行 `python scripts/install_semantic_retention.py`，流式下载并校验两组模型，支持已存在文件和 `--model-cache <模型缓存目录>`；`--verify-only` 只校验。B200 使用现有 GitHub 分片及整文件 SHA256，9B 和 processor 使用固定 Hugging Face revision 与 SHA256。没有重打包第三方运行库。

运行环境与原转换环境隔离。准备 Python 3.12.10 的独立 venv，安装下列精确版本：torch==2.10.0+cu130（PyTorch cu130 源）、transformers==5.5.0、tokenizers==0.22.2、numpy==2.5.3、Pillow==12.3.0、safetensors==0.8.0、jsonschema==4.26.0。另准备 Python 3.10.11 环境：PyMuPDF==1.27.2.3、numpy==2.2.6、Pillow==10.4.0、scipy==1.15.3、jsonschema==4.26.0。完整身份与原生文件哈希见 TOOLS/semantic_retention/PUBLIC_RUNTIME_LOCK.json。安装脚本检查现有环境，不修改全局包。

注册命令（Windows）：
```powershell
python scripts/install_semantic_retention.py --runtime-only --register-runtime --inference-python <3.12环境的python.exe> --source-python <3.10环境的python.exe> --accept-external-licenses
```
原生运行库按锁定哈希直接从 llama.cpp b11146 官方 Release 下载；先阅读 llama.cpp/MIT、LLVM OpenMP、NVIDIA CUDA 的各自许可。也可用 `--llama-dir <已有匹配目录>` 避免下载。仓库不包含 NVIDIA、Microsoft 运行库或 Python 环境。

注册后 `.local/semantic-retention-client.json` 提供独立客户端配置。把其 `SEMANTIC_RETENTION_ALLOWED_ROOTS` 改为允许读取的绝对目录 JSON 数组，再手动加入客户端。默认 `[]` 不允许分类输入。不会改写现有 MCP 配置。也可通过独立 launch.ps1 启动；资源实测使用直接 Python stdio 配置。

接口：semantic_retention_doctor、semantic_retention_start、semantic_retention_status、semantic_retention_results。start 返回 job_id，查询不会重启任务。传 package_dir（含 document.md/conversion_report.json/assets）和可选 source_pdf；全部源用途参与判断。asset_names 可限制输出范围。结果不会直接删除文件。

## 实测范围

12 本 8767 项、semantic-v2：正确率98.5172%，留存精确率97.1188%，误删0、误留130、待核0；99%未达，4本逐书留存精确率不足97%。28项评测标签按授权删除语义影响口径修订，原金标未改；相同预测按原gold误删28，v1为5。这不是28项模型能力提升，亦非新教材零误删保证。

累计执行5290.921秒，含人工恢复墙钟7095.657秒，属于断点恢复的逻辑冷运行。工作集保守4.881GiB，私有提交保守5.98354GiB，整卡显存5589MiB；这些是冻结分类链数据，并非完整OCR/BeMarkdown资源证明。发布时4图真实MCP推理与冻结决策一致；公开安装适配另有无模型测试。更换依赖/硬件后不自动继承实测质量结论。

公开内容只有程序、聚合指标、许可和模型下载身份。教材图片、MD、gold、逐图人工台账、私有模板、训练及响应缓存均不发布。组合工具AGPL-3.0-or-later；模型Apache-2.0，第三方各自许可。
