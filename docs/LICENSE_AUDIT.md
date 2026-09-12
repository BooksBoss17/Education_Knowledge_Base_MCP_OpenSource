# 许可与分发核查

本核查依据固定上游模型版本、已安装分发包的许可证元数据和随包许可文本。自有代码采用用户确认的 **AGPL-3.0-only**。第三方内容不因本项目的许可证而被重新许可。

## 分发结论

| 对象 | 核查结果 | 本仓库处理方式 |
| --- | --- | --- |
| BeMarkdown 自有代码、MCP 适配层、部署脚本 | AGPL-3.0-only | 提供源码、对应 wheel、完整许可证及构建说明 |
| PyMuPDF | GNU AGPL 3.0／Artifex 商业双许可 | 采用 AGPL 路线，由安装脚本获取；不把它误标为闭源或必须购买 |
| PaddlePaddle、PaddleX、Transformers | 上游声明 Apache-2.0 等开源许可 | 固定版本在线安装 |
| PyTorch／TorchVision | 开源代码许可证及多项第三方声明；CUDA 版本还涉及供应商运行库 | 获取官方 wheel，保留其上游条款，不随本仓库复制运行库 |
| NVIDIA CUDA／cuDNN 组件 | 专有许可，不能以本项目 AGPL 许可证分发 | 安装时从官方包渠道下载，并要求部署者接受外部条款 |
| Microsoft Visual C++ Redistributable | Microsoft 专有运行库；Torch 明确加载其中的 DLL | 检测后通过 winget 获取官方安装程序，不随仓库复制 DLL |
| Windows / GDI、NVIDIA 显卡驱动 | 系统／供应商组件 | 作为宿主环境前提，不复制系统 DLL 或驱动进仓库 |
| LibreOffice | 开源许可，主要为 MPL-2.0，另含上游第三方内容 | 优先复用现有安装，否则通过 winget 安装官方程序 |
| 13 个模型 | 固定上游版本均声明 Apache-2.0；具体证据见下方 | 记录模型版本、文件 SHA-256 和下载地址，运行时下载完整文件 |

**没有发现当前这 13 个固定模型版本明确禁止按其 Apache-2.0 条款分发。** 模型选择下载方式主要是因为 GitHub 单文件限制及整体体积，不能据此判断模型闭源。模型卡声明也不等于对训练数据、商标、专利或所有使用场景的额外授权。

## 模型

模型 ID、精确 revision、上游 API 证据和声明存于 [model-sources.json](../licenses/model-sources.json)。部署计划存于 [model-downloads.json](../scripts/model-downloads.json)，包含每个文件的大小和 SHA-256。

- PaddlePaddle：PP-DocLayout_plus-L、PP-OCRv6_medium_det、PP-OCRv6_medium_rec、PP-FormulaNet_plus-L、PP-LCNet_x1_0_table_cls、SLANeXt_wired、SLANeXt_wireless、RT-DETR-L_wired_table_cell_det、RT-DETR-L_wireless_table_cell_det、ch_SVTRv2_rec。
- stepfun-ai：GOT-OCR-2.0-hf。
- Qwen：Qwen3-Embedding-4B。
- tencent：WeMM-Embedding-4B。其 Hugging Face `license` 字段为 `other`，但同一固定版本的 `license_name`、README 和 LICENSE 声明 Apache-2.0；同时保留第三方条款。

GOT 和 ch_SVTRv2 的旧转换清单存在 `UNRESOLVED` 历史字段。为了保留冻结模型身份，这些字段没有被随意改写；本次独立核查提供了官方 Apache-2.0 声明与固定下载 revision。它们不是本次核查仍未知的模型。

FormulaNet 原清单中含有 9 个 Hugging Face 下载缓存文件。公开包移除了缓存并重建模型／套件指纹；实际推理权重和配置文件逐字节保持一致。

## NVIDIA 专有组件

冻结 Python 环境中的以下 8 个分发包标为 NVIDIA 专有许可：

`nvidia-cublas-cu12`、`nvidia-cuda-runtime-cu12`、`nvidia-cudnn-cu12`、`nvidia-cufft-cu12`、`nvidia-curand-cu12`、`nvidia-cusolver-cu12`、`nvidia-cusparse-cu12`、`nvidia-nvjitlink-cu12`。

上游条款入口：

- [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html)
- [cuDNN 许可](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html)
- [NVIDIA 驱动下载](https://www.nvidia.com/Download/index.aspx)
- [Microsoft Visual C++ Redistributable 官方下载与说明](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

安装脚本不宣称这些运行库变成了开源软件。`-AcceptExternalLicenses` 表示部署者已经阅读并接受其适用条款；不接受时，完整 GPU 路线不能完成部署。脚本不会自动签署商业合同或安装需要重启的显卡驱动。

## 源码与随包第三方内容

- PyMuPDF 官方说明：[许可说明](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)。AGPL 路线允许开源使用，分发及通过网络提供经修改服务时应遵守相应源码提供义务。若需要不兼容的闭源分发方式，应另行评估商业许可。
- `mathtypejx` 固定到 Git commit `7d90e7274c85cf56ac28d4d15e593044693d7e70`，MIT；其 fontmaps 附带 BSD 风格声明。安装使用固定源码归档及 SHA-256，不依赖用户预装 Git。
- `mathml2latex`：MIT，由安装脚本获取。
- 随 BeMarkdown 分发的 `omml2latex` 派生解析器：Apache-2.0，保留原许可及修改说明。
- 随包 MathLive：MIT；KaTeX 字体许可另行保留，见 [第三方声明](../THIRD_PARTY_NOTICES.md)。

全部 110 个已安装 Python 分发包的版本和许可证元数据保存在 [runtime-components.json](../licenses/runtime-components.json)。环境构建工具也列入清单，不等于全部由转换代码直接调用。部分元数据采用复合 SPDX 表达式或完整许可正文，应查看该组件自己的条款，不能仅凭名称推定。
