# 自带模型与服务

使用者可为四个角色选择不同模型，关键是实现API协议，而不是把任意权重文件改名放进固定目录。推荐让服务自行管理CUDA、批量大小、上下文窗口、并发和卸载；API客户端的资源统计不等于远端服务显存统计。

## 自建本地或局域网服务

按API.md实现四角色协议，服务放在本机回环地址即可直接连接。局域网/互联网服务通过api_gateway.py的bemarkdown转发模式连接；远端需HTTPS。可以混搭本地layout和远端OCR。表格模型必须能够返回完整HTML表格；文本模型需支持图片输入。

## 本仓库已有后端

正式验证曾使用PP-DocLayoutV3、Xiaomi-OCR-0和OvisOCR3。四角色由三个物理模型承担，table/text共享Ovis架构但参数独立。这只是参考组合，不是必装依赖。权重来源和许可证由使用者自行向上游核对；本仓库不重新分发权重。

内置后端入口在SOURCE/bemarkdown/local_model_server.py及pdf/visual_serial_runtime.py。复制config/local-model-server.example.json至TOOLS/bemarkdown/config/local-model-server.json，填写绝对模型路径；分别需要匹配这些架构的Transformers/Torch/torchvision/accelerate/einops/timm/sentencepiece及OpenCV环境。其他架构建议实现独立API适配器。

内置托管模式只在Windows/NVIDIA环境验证：另外准备兼容的GPU依赖，确认已安装到BeMarkdown所选环境，随后移走用户生成的.local/model-api.json，才会启用TOOLS/bemarkdown/convert.py的内置监听器与GPU准入。恢复外部API模式时放回该配置。每次切换应先完成/取消当前任务并重启客户端及空闲管理器，避免混用注册表。

API客户端模式不要求本机GPU，转换本身仍需内存。不要让外部API模式偷偷在客户端进程中加载重模型来绕过GPU预算。模型服务的GPU必须由其部署者管理。

## 安装与迁移

保留公开安装适配层：setup.ps1 / scripts/setup.py / TOOLS/bemarkdown/install.py。公开版通过pip安装所需核心依赖，区别于本机正式版的离线大依赖包；原有业务工作区不会被覆盖。--dependency-runtime允许借用现有依赖环境，当前核心wheel仍独立安装，不修改被借用环境。

迁移时复制仓库，重新运行安装器并配置本地服务/远端网关；不复制另一台机器的.local/runtime-selection.json。MCP客户端生成文件采用当前机器绝对路径，它只存在被git忽略的.local中。外部API密钥不写入示例、模型身份或发行清单。
