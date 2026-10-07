# 教材图片筛选

此接口用于减少教材整理时重复查看图片的工作。它按文件 SHA 合并相同图片的展示，列出装饰候选，并保护待识别内容。相似度高、颜色少或图片狭长，都不能单独证明可以排除图片。

## 调用

完成初次转换后，先生成筛选队列，再集中读取候选资产与对应源页：

```json
{"job_id":"已有教材作业ID","action":"screen_images","screening_model":"rules","screening_view":"triage","offset":0,"limit":100}
```

工具名是 `textbook_organize`。`rules` 使用本地像素特征，不加载嵌入模型。`screening_view="candidates"` 只返回经过内容保护检查后仍成立的装饰候选；默认 `all` 返回全部不同图片组。`offset` 在所选视图内分页，`view_groups` 是该视图的组数；整包总数保持不变。组内保留每一处引用和来源。分页当前仍会重新扫描整个包，不能把分页数当成额外处理图片数。

需要相似图候选时，将 `screening_model` 改为 `wemm`。首次返回 `QUEUED` 或 `RUNNING`，用相同参数轮询；不要重新提交转换。它复用本地已安装的 WeMM-Embedding-4B，与转换任务共用 GPU 队列，离线计算图片向量。只有模型指纹与内置模板一致时才进行正、负模板检索。不同源文件、Markdown、图片集合或模板不会混用缓存。

`SUCCESS` 只表示筛选执行成功。`candidate_groups` 是候选数，`actual_exclusions` 为本次实际排除数；筛选本身不会删除原资产或改写 Markdown。模型缓存的加载耗时与热图片耗时分开返回，不能当作 PDF 整页转换速度。

## 内容保护

`triage` 是优先核查视图：保留规则或 WeMM 标记的图片，即使来源保护将其分类为 `CONTENT_REVIEW`，也不会把它从列表隐藏。`triage_groups` 不是可以删除的数量。`triage_reasons` 与 `protection_reasons` 必须同时读；队列外的图片也未被自动认定为裁剪正确，仍需处理转换交接里的内容与边界问题。

把队列中每组的代表引用 ID 传入 `contact_sheet(image_ids=[...], base_sha256=...)`，可直接生成最多 24 张的指定图片联系表，避免按全书图片次序翻页。ID 绑定当前 Markdown SHA，过期会拒绝；联系表不替代源页核查。透明图在显示和 WeMM 输入中合成白底，原资产不变。

WeMM 使用工作区的 `6gb` / `10gb` 档位，分别以整个设备采样用量 6144 / 10240 MiB 为上限；共享转换锁并监控实际子进程。模型保持 bf16，以显式 CPU 权重卸载控制显存，不能称为纯 GPU 推理。模型输入预处理、资源策略、档位和模板身份参与缓存键；模板预处理不一致会拒绝检索。开发实测不能替代真正 6GB 显卡或完整教材验收。

- 公式、表格、文字图片、仍待审的转换结果保持保护。
- PDF 原页文字重叠、图题邻近、原生照片重叠或向外延续的矢量线条可以否决装饰候选。与原生图片完全相同的框也不能证明当前资产就是那张图片：遮罩残图可能沿用同一个框，因此同样保护。这些空间证据不是区域身份，也不是文字覆盖证明。
- 扫描页、旋转页或缺失来源字段时，原页检查可能无法判断。`UNKNOWN` 和 `NO_KNOWN_NATIVE_CONFLICT` 都不是自动排除许可。
- 薄线、箭头、刻度、照片边角可能低信息但仍有教学含义。仅存图片的公式仍属待识别，不能用排除图片掩盖漏转。

## 一次审核，按原来源复用

对源页和实际资产完成核查后，可调用 `review_images` 记录决定：

```json
{
  "job_id":"已有教材作业ID",
  "action":"review_images",
  "base_sha256":"本次检查的Markdown实际SHA256",
  "image_reviews":[{
    "image_id":"inspect返回的图片ID",
    "decision":"EXCLUDE_DECORATION",
    "source_context_checked":true,
    "no_meaningful_content":true,
    "no_unresolved_content":true,
    "source_evidence":"具体源页、区域及无教学内容的核查依据"
  }]
}
```

上述布尔值必须来自实际核查，不能按示例直接照填。转换仍将资产标为保护或待审时，不接受排除决定。其他决定为 `KEEP`、`NEEDS_RECOGNITION`、`REVIEW_UNKNOWN`。

决定绑定完整源文件 SHA、来源页/部分、资产 SHA 与策略版本；不会扩散到相似图片、其他页或其他教材。同一身份重复整理时，`preview` / `publish` 自动复用已审决定。显式整理计划的图片动作仍优先，实际生效动作写入计划及发布记录。记录 `KEEP` 可撤回此前的自动复用排除。原资产始终保留；本功能不清理旧知识库。

## 留痕与恢复

工作区 `tmp/bemarkdown/.mcp/` 中保存：

- `as/<缓存ID前24位>/`：筛选状态、完整缓存身份、提取命令、stdout/stderr、向量封口与计时。
- `screening-reviews/<源SHA前24位>.json`：审核决定与历史；记录内仍使用完整身份。

`FAILED` 保留错误现场；服务重启后失去运行句柄的旧任务返回 `INTERRUPTED`，不会自动无限重跑。当前没有公开的筛选重试/清理命令，应保留该目录交维护者确认，不要重跑教材转换或手改成功状态。

当前能力是候选筛选、内容保护和已审决定复用。它尚不代表新教材装饰图片可以全部免审删除；实际误筛数、候选召回、复用排除数和剩余审核数必须分别评价。

### 修改 Markdown 后复用 WeMM 特征

WeMM 向量任务按同一中间包的图片路径/SHA集合、模型指纹、预处理和工作进程代码、资源档位确定身份。修改正文或图题、移动或去掉重复引用、已转写公式不再引用原图时，可以复用覆盖当前图片集合的已封口结果，或等待仍在本服务运行的覆盖任务，不会仅因 Markdown SHA 改变重跑未变图片。

复用的只有像素向量。每次请求仍根据当前 Markdown、当前图片 SHA、当前模板阈值及源页保护条件重新生成筛选结果；审核与发布仍绑定当前 `base_sha256`。返回 `vector_cache_asset_count` 和 `requested_distinct_images` 区分原任务图片数与本次所需图片数。损坏/不完整结果不能当成功复用；增加新图片或改变像素且无兼容的完整缓存时仍会启动新提取，不宣称已实现逐图增量推理。新书的新转换包不复用旧书中间包任务。
# 逐次引用的来源身份

同一图片文件可在多页复用；文件 SHA 相同不表示源页相同。PDF 筛选优先使用 `conversion_report.json` 中的原始 Markdown UTF-8 节点跨度，再逐条验证并重放 `agent_review.json` 的修订链。`source_mapping=RENDERER_SPAN_AND_REVIEW_REPLAY` 表示该引用已绑定原始渲染节点；`source_document_node_id` 是 DocumentIR 节点身份，不是区域身份。图题 alt 的修订不会改变未变动图片目标的来源。

跨度、原文哈希、修订链或图片目标无法核实时，该引用保持 `SOURCE_OCCURRENCE_UNRESOLVED` 保护，不把去重资产清单里的最后一个页号复制给所有引用。原节点的待审状态和公式／表格／文本图片保护也随引用保留。旧产物单引用的 `SINGLE_REFERENCE_MANIFEST` 仅表示资产清单来源，不冒充已验证的渲染节点。

## Source heading backdrop rules

The `rules` organizer can additionally omit a heading backdrop when a reviewed, source-specific vector template is present under `tmp/bemarkdown/.mcp/screening-background-templates/<source-sha256-first-24>.json`. Missing templates or missing final DocumentIR keep existing behavior; image similarity alone never grants omission.

This policy requires all of the following: a uniquely mapped VECTOR_VISUAL_FALLBACK occurrence, byte-identical reproduction of the current asset, complete omitted-character delivery into the current Markdown through renderer spans and exact review replay, the same heading text and translated vector commands as the approved template, and retention of every excluded semantic image owner. Raster pixels must either belong to retained owners or provably make no contribution in an actual render comparison. Extra strokes, unmatched raster content, unresolved text or changed source/Markdown/asset identities prevent a match. Templates are scoped to the exact source document; they are not global deletion whitelists.

Decisions apply to individual image references, not every occurrence of the same SHA. Explicit KEEP/unknown/recognition decisions take precedence. A preview must retain required owner references; their boundary-review states are not cleared by background removal. Output exposes `automatic_background_actions` during screening and `automatic_background_omissions` during preview. Detailed evidence is written to a distinct per-run file under the job's `screening-background/` directory.

This is a narrow reusable heading policy, not general decoration recall or content-quality certification. Source formula/image/table review obligations remain independent. Preview omission counts must not be reported as whole-book accuracy or false-deletion rates.
