# BeMarkdown 当前完整技术路线

本说明对应基底版本 BeMarkdown 0.2.0 与 MCP 适配层 0.2.0。公开分发调整了许可证、安装方式、说明元数据及模型缓存清单，保留现有识别路线和推理参数。

## 1. 工作流边界

一次工作流由三部分组成：

1. **确定性转换与本地模型识别**：读取 DOCX/PDF，识别或提取内容，输出中间 Markdown、图片及来源信息。
2. **基于证据的复核**：MCP 将已标记的文字候选、表格内容任务、图像区域检查交给宿主 agent；agent 查询原文件证据后执行精确修订。
3. **知识库加工**：按教材、试卷、答案等类型进行删减、拼接、规范化和入库。这部分作为后续 Skills／工具扩展。

BeMarkdown 本地模型不需要云端 API key。宿主 agent 使用什么模型、是否连接云服务，由部署者自行配置。

## 2. 入口、任务和产物

`bemarkdown_convert` 创建持久任务 ID，将源文件路径与 SHA-256 写入任务状态，排队执行真实转换子进程。转换前后核对源文件身份，避免处理期间源文件变化。

共享工作区内用文件锁串行占用转换 GPU。MCP 通过 `bemarkdown_status` 返回排队、运行、成功、失败等状态，调用者应继续查询同一个任务，而不是重复提交仍在执行的转换。

输出根目录为 `<工作区>/tmp/bemarkdown/`：

| 文件／目录 | 作用 |
| --- | --- |
| `document.md` | 原始中间 Markdown |
| `conversion_report.json` | 来源、转换路径、识别／复核信息、耗时等 |
| 图片资源 | 示意图、保留区域或需要进一步识别的源图；Markdown 使用相对引用 |
| `debug/` 等证据文件 | 保存需要的区域 IR、原始分页或来源定位信息；具体内容由路线决定 |
| `document.reviewed.md` | agent 修订后的 Markdown，保留原始版本 |
| `agent_review.json` | 每次替换的前后文本、证据、时间及版本哈希 |

任务目录 `.mcp/jobs/<job_id>/` 包含 `state.json`、`stdout.json`、`stderr.log`、`validation.json`、`agent_handoff.json`。复核候选数不是已证实的错误数；全部 review 标记数也不等于 agent 的文字修复数量。

## 3. DOCX：先保留可直接解析的结构

DOCX 是 OOXML ZIP 包。正常路线先检查压缩包资源限制、关系路径和 XML 结构，再按文档顺序构造内部节点。

| 内容 | 优先路线 |
| --- | --- |
| 普通段落、标题、列表、超链接 | 从 OOXML 提取结构和文字，保留文档顺序 |
| 原生表格 | 读取行列、单元格和其中的内容 |
| OMML 原生公式 | 使用随包 OMML 解析器转换为 LaTeX，再做结构校验 |
| Word EQ 域 | 解析域指令并转换 |
| MathType / OLE / MTEF | 提取嵌入对象，使用固定版本 mathtypejx 解析，结合缓存和校验 |
| WMF 等兼容图形 | 检查内部结构，必要时使用 Windows GDI 渲染 |
| 无法可靠直接解析的公式图 | 交给 FormulaNet 与公式安全门，保留待复核状态和原始证据 |
| 普通图片／示意图 | 提取资源并保留 Markdown 图片引用 |

结构公式不能为了方便整页 OCR 而先全部光栅化。原生文字、OMML、EQ、OLE 等结构存在时，优先保留它们的直接解析通道。

关键源码：`SOURCE/bemarkdown/src/bemarkdown/package.py`、`pipeline.py`、`eq.py`、`mtef_cache.py`、`drawingml.py`、`wmf_renderer.py`、`formula_ocr.py`。

## 4. DOCX 中的截图和混合图片

这里有两条不同路线，不能把“Word 页数”“图片数量”和“内部 OCR 页数”混为一谈。

### 4.1 原生 Word 中夹带文字截图

先提取原始嵌入图片，做版面分析，判断它属于纯示意图、装饰／空白图，还是需要识别的图文区域。

- 纯示意图作为图像保留，图内标签随图保存。
- 文字截图或图文混排图片按区域进入 PDF 识别链路，保留原图像像素、坐标与 Word 中的插入位置。
- 表格、公式等区域进入各自专用路线；转换结果按原位置回填 Word 的中间节点。

10gb 调度将同一个 Word 内多张需要识别的图片组成一次多页 PDF 流水线调用，复用模型加载。**PP-OCR 的批处理仍按原始图片隔离**，避免不同图像混批改变 padding、文字结果和坐标。

关键源码：`docx_image_content.py`、`pdf/production_runtime.py`。

### 4.2 截图型 Word 的默认路线与可选分页路线

当前 `production.convert` 默认 `docx_image_route="original"`。即使截图资格检查通过，默认仍提取原始嵌入图片，并走上一节的区域识别；报告的实际路线为 `input_transform.route=DOCX_IMAGE_CONTENT`。资格检查中的 `profile.route=DOCX_SCREENSHOT_PDF` 只是候选路线信息，不能据此认定执行了分页。

只有显式选择 `docx_image_route="pagination"`、截图资格通过且 `formula_ocr="auto"`，才使用 **LibreOffice 无界面分页导出 PDF**，然后进入 PDF 模型链路。当前 MCP 默认调用原图路线；分页选项供 Python／命令行路线使用。它不是默认转换必经步骤，也不依赖 Microsoft Word。

截图资格条件为：原生正文非空白字符不超过 200；图片显示面积合计至少 30 平方英寸；无 OMML、OLE、EQ 等结构公式；无非超链接的外部内容关系。没有至少几张图片的硬性门槛，普通超链接也不会单独使其不合格。

分页路线会保存来源 PDF 和变换信息。`bemarkdown_source` 仅在实际有分页来源时使用 `source_view="pagination"`。默认原图路线应先用 `kind="images"` 列出 Word 原图，再用返回的 `image_name` 获取图像。

分页受字体和 LibreOffice 版本影响，应单独记录渲染耗时；默认原图路线不应把该耗时计入必需步骤。关键源码：`production.py`、`docx_visual.py`、`docx_image_content.py`。

## 5. PDF：按区域选择原生提取与识别

### 5.1 读取与版面坐标

PyMuPDF 读取 PDF 的原生文本、字形、图片、矢量及页面信息，提供渲染和裁剪。布局阶段当前默认以 **200 DPI** 渲染。

PP-DocLayout_plus-L 检测文字、公式、表格、图片等区域；布局融合与自适应细分处理遗漏、重叠和较复杂区域。统一保留 PDF 点坐标和渲染像素坐标之间的转换关系。

关键源码：`pdf_layout_runtime.py`、`pdf_layout_fusion.py`、`pdf/adaptive_pipeline.py`、`pdf/modular_pipeline.py`。

### 5.2 内容路由

区域路由包含以下实际通道：

| 通道 | 处理内容 |
| --- | --- |
| `NATIVE_TEXT_BRIDGE` | 可以可靠使用的 PDF 原生文字 |
| `OCR_TEXT_REGION` | 需要识别的普通文字区域 |
| `FORMULA_RECOGNITION` | 公式区域 |
| `TABLE_ENGINE` | 表格结构与内容 |
| `IMAGE_NATIVE_EXTRACT` / `IMAGE_RENDER_CROP` | 原生图片提取或页面区域裁剪 |
| `PAGE_VISUAL_TEXT_RECOVERY` | 需要补充恢复的页面视觉文字 |
| 保留／复核通道 | 无法可靠结构化或存在冲突的区域 |

原生 PDF 也可能包含扫描区域、不可用字形编码或不完整文本层；因此不是只按“原生 PDF／扫描 PDF”标签决定整页是否 OCR。

关键源码：`pdf_content_router.py`、`pdf_source.py` 及原生文字、字形、可见性相关模块。

### 5.3 普通文字的三模型路线

需要 OCR 的普通文字区域使用固定的三类识别证据：

| 提供者 | 模型 | 作用 |
| --- | --- | --- |
| A | PP-OCRv6_medium_det / PP-OCRv6_medium_rec | 文字检测、识别及位置证据 |
| B | ch_SVTRv2_rec | 第二路文字识别 |
| C | GOT-OCR2.0 | 生成式 OCR 证据 |

系统按区域对齐结果、记录分歧和运行失败，再进行内容选择及待复核标记。三路一致也可能一起出错；模型投票不构成独立准确率证明。

GOT 为自回归生成，耗时受裁剪数量、图像内容和生成长度影响。转换速度不能只用一个固定的“GOT 每页秒数”解释。

关键源码：`pdf/production_runtime.py`、`pdf/production_three_model_live.py`、`pdf/three_model_ocr.py`、`pdf/workers/`。

### 5.4 公式

公式区域进入 PP-FormulaNet_plus-L 路线，结合 LaTeX 结构、几何信息以及相应验证／修复逻辑处理。原生 PDF 中可恢复的数学信息和行内公式使用单独的几何与顺序处理。

不可靠结果保留为待复核／待识别，并保留来源。**只有清晰原图、没有可靠 LaTeX 的公式，不计作结构化识别正确。** 独立公式标记不应仅依赖精简文字 handoff；agent 还可读取完整 `issues` 和报告。

### 5.5 表格

PP-LCNet_x1_0_table_cls 判断表格类型；SLANeXt_wired / wireless 与 RT-DETR-L_wired / wireless_table_cell_det 提供相应结构、单元格证据。表格引擎结合原生线条、文字和公式来源信息组织内容。

能够可靠恢复时输出结构化内容；不能可靠恢复时保留原区域与表格任务。表格图片保留不等于结构化表格识别完成。

关键源码：`pdf_table_engine.py`、`pdf/table_runtime.py`、`pdf/source_table_grid.py`、`pdf/source_table_math.py`。

## 6. 示意图如何截取与保护

示意图的区域来自版面检测，并结合页面原生图片、矢量和边界信息。适合直接提取时使用原图；否则按区域从页面渲染裁剪。输出图片通过相对路径在 Markdown 中引用。

图像、表格、公式等非正文区域建立归属关系。文字恢复时排除被这些区域占有的内容，必要时生成只含文字区域的渲染视图，避免把图内坐标、字母和标签当成独立正文重复 OCR。

这依赖区域边界和归属判断，不是每幅图都能被完美分割。图文重叠、贴边标签、复杂几何图和检测漏框仍可能误分；相应图像区域可以进入复核。保留图像是教学示意图的正常产物形式。

关键源码：`pdf/non_text_ownership.py`、`pdf/image_containment.py`、`pdf/figure_graphic_markers.py`、`pdf/figure_waveforms.py`、`pdf_output_audit.py`。

## 7. 合并、顺序与 agent 修订

区域内容合并为文档 IR，再组织阅读顺序、列顺序、行内公式、图像引用及表格内容。几何关系参与排序，避免把每次生成的 OCR ID 当成相同位置元素的唯一顺序依据。

MCP 生成的 handoff 包含候选内容和原文件定位。agent 默认处理已标记项，通过 `bemarkdown_source` 查看原 PDF／Word 证据；修订必须提供当前版本哈希、唯一匹配的旧文本、新文本及来源证据。

这套修订接口记录更改，但不会自动证明 agent 的更改正确，也不会自动发现全部未标记错误。宿主 agent 时间、模型延迟与人工处理时间应和本地转换时间分开统计。

关键源码：`pdf_document_ir.py`、`TOOLS/education_mcp/review_handoff.py`、`server.py`。

## 8. 性能与显存调度

| 档位 | 自动选择条件 | GOT 分配器预算 | 语言批次 | 视觉微批 |
| --- | --- | --- | --- | --- |
| 6gb | 独立显存 6144–10240 MiB | 2816 MiB | 32 | 1 |
| 10gb | 独立显存大于 10240 MiB | 6144 MiB | 32 | 1 |

10gb 通过提前准备依赖、复用模型、多张 Word 嵌入图合并处理等方式减少重复开销。模型来源、权重、精度和识别范围保持不变。

MCP 采样整卡显存。6gb 档目标为 6144 MiB、硬阈值 8192 MiB；10gb 档硬阈值 10240 MiB。超限终止只针对本次拥有的转换进程树，采样不能保证捕捉瞬时峰值。

主要耗时通常来自模型加载、OCR／公式推理、复杂区域拆分以及截图分页。原生 Word 的 XML 解析和清晰原生 PDF 的文字提取通常较轻。不同文件的页数、嵌入图数量、OCR 裁剪数量和复核数量不同，不能直接比较单一秒／页数字。

## 9. 扩展位置

BeMarkdown 负责中间转换。可以在共享模型层加入其他模型，在工具层增加嵌入、检索、知识库维护等能力，在 Skills 层编排教材和试卷加工规则。

Qwen3-Embedding-4B 面向文字嵌入；WeMM-Embedding-4B 面向文字、图片、视频和视觉文档等输入。当前基底已提供两者的登记与下载，后续接入时需要独立验证其运行依赖、显存预算和检索质量。
