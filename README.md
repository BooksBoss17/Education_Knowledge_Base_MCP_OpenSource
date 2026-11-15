# Education Knowledge Base MCP · BeMarkdown API 版

将 PDF、截图、Word 转为可追溯的 Markdown。**模型由使用者选择：部署自己的本地服务，或通过本地网关调用联网 API。** 无需下载本项目过去的固定模型组合。默认不安装模型权重，不启用图片分类或 Semantic Retention。

核心版本 **0.4.3+localapi.4**；MCP 适配器 **0.6.0**。完整旧线路独立保留在 [BeMarkdown Classic](https://github.com/BooksBoss17/BeMarkdown-Classic)。本仓库历史标签仍可访问旧版本。

## 转换流程

```text
PDF / 图片 / Word中的截图
  → layout：版面框和阅读顺序
  → formula：公式与框内文字
  → table：完整表格（保留单元格内公式）
  → text：其余可见文字及漏框公式
  → 位置组装、同源去重、残片过滤
  → document.md + 图片 + 来源与校验清单

原生Word正文、表格、可解析公式 → 直接解析；无需先转PDF
```

成功识别的公式、表格区域在后续文字输入中遮盖，失败内容保留来源待核。四个角色可指向不同服务，也可以共用一个模型。普通聊天API不能直接充当版面检测器：layout 必须返回图像像素坐标、类别及阅读顺序。

## 快速开始（Windows MCP）

安装 Python 3.11、Git。复制并编辑 `config/model-api.example.json`，指向你自己的四角色服务。核心连接回环地址；远端服务通过下一节的网关接入。

```powershell
$env:BEMARKDOWN_GATEWAY_TOKEN = 'replace-with-a-long-random-local-token'
python scripts/setup.py --api-config config/model-api.example.json --workspace C:/KnowledgeWorkspace
```

安装器只安装转换依赖和本仓库的核心wheel，不下载Torch、模型或LibreOffice。将生成的 `.local/mcp-client.json` 合并到MCP客户端配置中；如需转换工作区外文件，在启动参数中添加 `--input-root` 和允许目录。不要提交 `.local` 或密钥。

受管 MCP 目前支持 Windows；API模式的客户端不要求本地GPU。推理服务自身负责资源预算，客户端只对自己的RAM和作业生命周期进行管理。内置本地模型模式另有GPU环境要求，见 [模型与部署](docs/MODELS.md)。Linux/macOS可尝试核心CLI，未宣称完整MCP管理器跨平台支持。

```powershell
# 独立核心CLI；不经过MCP队列，调用者自行管理并发
.local/runtimes/<wheel-hash>/Scripts/python.exe -m bemarkdown convert input.pdf --api-config config/model-api.example.json --output-root outputs --json
```

## 调用联网API

编辑 `config/gateway.example.json`：layout使用兼容的版面服务；formula/table/text可使用支持图片的 Chat Completions 兼容服务，填写完整URL和实际模型名。示例域名是占位符，不能直接调用。

```powershell
$env:MODEL_API_KEY = '填写你的密钥'
# 使用安装生成的Python路径启动网关（保持运行）
.local/runtimes/<wheel-hash>/Scripts/python.exe scripts/api_gateway.py --config config/gateway.example.json
```

远端调用会发送相应页面/裁图，并按服务商计费。密钥从环境变量读取；网关不自动重试，不将服务商错误正文写入日志。必须设置至少16字符的 `BEMARKDOWN_GATEWAY_TOKEN` 保护网关，并在客户端各角色配置 `token_env`；使用MCP时将此环境变量配置在MCP客户端启动环境中。

**这是协议适配，不保证每一家服务商原样兼容。** 不支持的图片格式、模型参数、结构化表格输出等须由使用者适配。公开测试使用本机模拟上游，未使用你的密钥或实际付费服务。远端model_fingerprint表示声明配置的指纹，不表示已核验远端权重。

## 本地模型

推荐通过自建服务实现 [四角色协议](docs/API.md)，这样可随时替换模型，无需修改转换核心。已有V3/Xiaomi/Ovis后端作为可选参考；不安装它们也能使用自建或远端API。见 [部署说明](docs/MODELS.md)。

## 输出与边界

- 输出包含Markdown、图片、来源坐标、模型响应身份和文件哈希，支持包校验。
- 低信息残片过滤有意接受少量孤立标点损失，以减少错误扩写；不是通用纠错器。
- OCR错字、复杂布局、公式与配分归属仍可能需要核对。工程测试通过不等于全文或检索问答准确率认证。
- 源码与已发布wheel逐文件对应；不包含模型权重、教材样本、个人路径、宿主环境或密钥。

许可证：AGPL-3.0-only；第三方与模型各自许可见 [说明](THIRD_PARTY_NOTICES.md)。
