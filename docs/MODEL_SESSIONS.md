# 可选阶段会话 v2（0.4.4 开发候选）

客户端配置每个角色的 `session_mode`：`auto`默认发现能力，`off`使用原v1，`required`要求会话支持。只对标准`/v1/<role>`URL发现`GET /v1/capabilities`。404/405/501表示未提供扩展；认证失败、损坏能力响应和推理失败均不会回退重发。已有v1服务与Chat Completions网关不需要实现会话。

能力响应示例：

```json
{"schema":"bemarkdown-model-api-capabilities-v1","roles":["layout","formula","table","text"],"session_schema":"bemarkdown-model-session-v2","session_roles":["layout","formula","table","text"],"max_items":128,"max_bytes":67108864,"layout_batch_items":8,"idle_seconds":120}
```

`session_roles`可省略（等同roles）或列出支持的子集。公开网关按上游能力逐角色转接；普通Chat Completions仅复用连接协议，不能据此证明远端模型驻留或GPU释放。模型身份仍由服务声明，外部服务权重不可本地认证。

四个端点统一使用POST、原认证头及`schema=bemarkdown-model-session-v2`：

| 端点 | 输入附加字段 | 成功结果 |
| --- | --- | --- |
| `/v2/session/open` | 客户端生成的32位小写十六进制session_id、role | 同一session_id/role，state=OPEN；此时未必已加载权重 |
| `/v2/session/infer` | session_id、role、request_id、原items | 原身份与items，request_complete=true、model_unloaded=false、released=false |
| `/v2/session/status` | session_id，可选request_id | OPEN/CLOSED、model_unloaded；查询请求时返回COMPLETE/UNKNOWN与receipt |
| `/v2/session/close` | session_id | state=CLOSED、model_unloaded=true、worker_exit_code |

同一服务一次只允许一个活跃会话；重复open相同ID和角色幂等，阶段切换须先close。已关闭ID不能重新打开。receipt为`{payload_sha256,result}`；摘要对转成v1 schema、去掉session_id的完整请求做排序键、紧凑分隔符、UTF-8 JSON的SHA256。同一request_id相同摘要可返回完成收据，摘要不同拒绝。客户端遇推理响应超时/断连只查一次status；没有完整匹配收据则失败，不自动再推理。服务按任务目录保留收据，关闭后的内存索引最多64项，内置服务可回查落盘关闭收据；公开代理只保留最近64项关闭路由。

内置worker懒加载一次，直到显式关闭、120秒空闲或任务取消。每次推理最多540秒，客户端默认600秒；HTTP处理串行，正在推理时status/close可能等待当前请求结束。需要立即停止时使用受管任务取消，由Windows Job清理整棵所属进程树。客户端进程意外退出时，同Job取消或空闲期限负责回收；这不是断线后恢复转换任务的自动调度器。

传输分批可设置`batch_items`、`batch_bytes`、`batch_pixels`（可选图片总像素）；取服务和客户端较小限值，不改变模型内部推理批大小或生成预算。超大单项在发送前报错。每任务收据预算512 MiB，接近上限时在新推理前拒绝，已完成收据保留；归档由任务维护流程负责。更换模型后仍需实测耗时与资源，默认条目数不保证所有模型能在540秒内完成，可降低传输批数且保持生成预算。

请求metrics的`load_count`、`load_seconds`来自适配器实际加载计数/累计秒数的差值；适配器不提供时为null，不能填0或虚构加载一次。内置V3/Xiaomi/Ovis已提供该计数。formula、table、text属于不同阶段，即使table/text使用同一权重仍分别装卸，暂不跨角色或跨文档常驻。

受管任务写`model_api_progress.json`，MCP状态附带角色、stage/session/request ID和已完成请求/条目数；`live`以管理器RUNNING为准，终态保留最后观测且标记非活跃。进度不是全文准确率认证。运行注册表支持可选`runtime_files`哈希映射，在加载注册表和实际启动前校验导入源码及配置；不支持在运行期间热改文件。

