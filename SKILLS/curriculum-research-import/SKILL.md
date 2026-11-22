---
name: curriculum-research-import
description: 将物理课程标准、考试评价体系、命题分析或教学研究文献的 PDF 经 BeMarkdown 转录、原页校对和分模块整理，按原件独立导入教育知识库。适用于明确导入或在专用导入工作区接收这类资料。
---

# 课标与教学研究资料导入

接收原件与目标业务工作区。教材走教材导入。根据封面、目录、正文确认类型和版本。课标选 curriculum_standard，考试评价、试题分析、教育研究论文和研究专著选 teaching_research。仅转录、仅预览按请求停在相应阶段。

1. 调度接入时沿用主会话的 task_run_id 和同一 BeMarkdown 作业。先查已完成包与来源 SHA；整批新转换使用主调度 Skill 的 import_batch，将所有来源持久登记。单份直接转换时通过 bemarkdown_vision 验证后调用 bemarkdown_convert、bemarkdown_status，读取报告、handoff_summary 与原页。检查现有包时提供原 PDF。原件只读，草稿和审查计划放工作区 tmp。记录原件备份到 Original_Backup/CURRICULUM_STANDARDS_AND_RESEARCH_PAPERS 的受管原件位置，如已有同哈希原件则复用。
2. 按物理页逐页对照原件，最多四张一批。保留规范性用语、编号、版本、案例、表头与限定条件。论文保留作者、摘要、方法、证据、结论和参考文献的原次序，考题分析保留题干、选项、解析和图注。节选写明实际范围和跨页截断；目录引用的缺页不能扩成正文。有待核时保存草稿和页码，继续其余可执行页。
3. 按原有一级章节或论文自然段落划分模块。序号只表示顺序。短资料有一个正文模块即可。建立 JSON 计划，见下列字段。每页写简短具体的图文核对记录，逐项解决报告中的已知问题；无法判定的内容放 unresolved。定位到 OCR 错漏后记录原字串、准确的新字串和依据。原件限制如节选确实终止于半句，记录页界，无依据不补写。
4. 先预览再发布。用 Python 3.11+ 运行随 Skill 的 scripts/organize.py。预览在工作区 tmp 中生成整套可阅读的稿件，返回 preview 路径及内容指纹。核对模块、页界、图片、图注和目录，确认已获入库授权后原样带入 --publish 指纹。每份资料在 KNOWLEDGE_BASE/CURRICULUM_STANDARDS_AND_RESEARCH_PAPERS 下有独立目录，含 00-索引.md、顺序编号模块、manifest.json，必要插图使用该资料自己的 DIAGRAMS。备份由发布脚本按原件哈希完成。重试核对已有成品；遇到同名不同来源或版本冲突，先保留旧版并处理冲突。

```
python organize.py --source 原件.pdf --package 转录包目录 --workspace 业务根 --capture-id 采集ID --plan 审查计划.json
python organize.py --source 原件.pdf --package 转录包目录 --workspace 业务根 --capture-id 采集ID --plan 审查计划.json --publish 预览返回的指纹
```

计划含 title、kind、source_sha256、document_sha256；pages 是以本 PDF 物理页码字符串为键、原页核查说明为值的对象；有待核编号时用 report_sha256 绑定实际 conversion_report.json 的 SHA-256。decisions 以本次 conversion_report.review_items 的 **1 起始位置**为键、实际处理依据为值。unresolved 是尚未处理的问题列表。parts 按顺序给出 title 和 start_line（document.md 的 1 起始行号；首项为 1）；每段延伸至下一段开始前，最后一段至文末。corrections 是含 page、old、new、reason 的列表，old 要在该页唯一且位于单个模块内。已有 Markdown 插图从包内读取，保留真实字节、扩展名、替代文字、可选 title 和图注。遇到保图形式的未识别文字或公式，应对照原页决定是否需要补录。测试用库与正式业务库由 --workspace 指定。
## 跨页续句与标题

逐模块查看相邻物理页的末段和首段，特别是论文首页的作者简介、收稿信息和下一页的页眉。确认是同一段落后，在计划的可选 page_joins 列表加入一条决议。page 是左侧物理页，tail 是该页末段正文全文，head 是右页续文的唯一开头。footer、header 分别抄录 tail 后、head 前的全部页边文字（没有则写空字符串）。这些字符串取自执行 corrections 后的稿件。reason 写明核对了哪两页及如何判定续接。中文直接拼接，英文词间需要空格时用 separator 填一个空格。独立段落或跨模块的边界保持原样，归属不明留待核。

程序将连贯段落放在右页并标注双方页码，出版与页边信息另出有页码的模块。核对索引、正文和清单中的跨页决议。标题层级根据原页父子关系判定，原始 Markdown 的层级如有错误，先通过 corrections 纠正。渲染器在每个模块 H1 下从 H2 开始排列这些已确认的标题。

## 监控取证

独立执行也由当前主会话先调用 runtime_jobs(begin)，传入事先保存的 task_run_id。保存返回的 capture_id，确认 runtime_monitor(status) 已有首个样本，然后再提交转换。接收调度派工时复用主会话正在运行的同一采集。单份转换在 bemarkdown_convert 调用传 task_run_id；批次通过 import_batch(status) 取已分配的作业，两种路径均用 runtime_jobs(inspect) 核实关联。复用已有转录包时记录历史作业和来源哈希，从本轮审查开始计算覆盖。

实际导入与全流程验收的预览、发布命令都带 --capture-id。采集必须属于 --workspace 指向的业务库。记录转换、原页审查、脚本发布与核验各阶段的起止时间。等待所有子任务完成和成品核验通过后，主会话执行 runtime_jobs(end)，核对终态、各阶段的时间与样本覆盖。短于采样间隔的阶段报告实际耗时和可用样本数。已有其他任务共享采集时，按管理器回执判断释放归属。单独离线回放历史成品时可以省略 --capture-id，回放结果仅作内容回归。
