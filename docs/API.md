# 四角色接口

客户端配置 schema 为 `bemarkdown-local-model-api-v1`，roles包含 layout/formula/table/text。每个角色设置回环HTTP `url`、可选 `token_env` 和正数 `timeout_seconds`。旧三角色配置可读取，但表格仅保图。

POST请求示例：

```json
{"schema":"bemarkdown-local-model-api-v1","role":"text","request_id":"request-1","items":[{"id":"0","image_base64":"BASE64_PNG","sha256":"64位图像SHA256","serialization_profile":"default"}]}
```

响应必须回传同一request_id、role和输入item ID。每个text/formula/table响应item包含字符串text；layout包含regions数组：

```json
{"schema":"bemarkdown-local-model-api-v1","role":"layout","request_id":"request-1","released":true,"identity":{"model_id":"your-layout-model","model_fingerprint":"64位版本或配置指纹"},"items":[{"id":"0","regions":[{"label":"text","bbox":[10,20,400,100],"reading_order":0,"score":0.99}]}]}
```

bbox使用收到图片的像素坐标，满足 `0≤x0<x1≤width` 和 `0≤y0<y1≤height`。常用label：text、doc_title、paragraph_title、formula、inline_formula、table、image、chart、figure_title、header、footer、number。不要返回归一化坐标或另一个缩放图的坐标。

公式返回LaTeX，混排普通文字时保留正文并用成对美元符号包围数学；table返回单个完整HTML table（显式闭合、支持rowspan/colspan）；text忠实保留可见内容和数学。不要补写被遮盖区域。空预测会触发保图/待核路径。

上限：128个item、64MiB请求/响应；服务必须逐item完整对应。released=true表示这一请求已完成、服务已释放请求级资源，不代表客户端验证了远端GPU卸载。模型身份在同一文档中不得漂移。

网关的 `bemarkdown` 协议原样转发本协议；`chat-completions` 协议逐裁图发送含text/image_url的messages，读取choices[0].message.content，拒绝length等不完整完成状态。后者仅支持formula/table/text，layout需专门的版面服务。配置URL必须HTTPS或回环HTTP，不跟随重定向。无自动重试。
