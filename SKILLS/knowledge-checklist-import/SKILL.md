---
name: knowledge-checklist-import
description: 将知识手册、考点总结、公式表或讲义中的知识段落整理为带原件出处的知识清单。用于独立清单入库，以及混合讲义的知识摘录支线。
---

# 知识清单导入

主会话按明确的入库意图选源。用 `checklist_import(inventory)` 盘点选定路径，保存文件 SHA-256、来源别名及不可读项。按原件内容区分独立知识资料、混合讲义、试卷和广告。若唯一可用来源在 ZIP 中，先以 inventory 返回的 origin 和 sha256 调用 stage，使用返回的 source 送 BeMarkdown。原 ZIP 路径和成员信息随来源登记。旧版中文 ZIP 的成员名乱码时，单独以对应的 zip_encoding 重新盘点，例如 gbk。优先复用已经核实的成功作业或讲义归档。

1. 用 `checklist_import(prepare)` 创建或找回草稿。独立资料用 `mode="standalone"`，有题有知识的讲义用 `mode="handout"`。sources 传 `[{ "job_id": "...", "origins": [...] }]`，origins 可省略。教、学两个版本可按阅读顺序放入同一组，以更完整的版本为首份。若已有题库发布全文，批次可直接传 `[{ "handout_sha256": "..." }]` 补提。处理同一份混合讲义时，题目和知识两条支线共用成功的 BeMarkdown 作业。讲义原件和全文可从题目发布回执核对。
2. 用 lines 分段阅读全文及行号。标题候选仅供定位，检查表格、图注和未编号段落。收录有独立知识用途的原文定义、规律、公式与条件、易错辨析、通法。保持公式与适用条件、量纲和必要图表相邻。遇到跨段定义或表格时使用完整区间。例题、练习及其作答交给题目支线，理解原文知识必需的简短示例可随解释保留。精准引用原件，纠正转录歧义时写清核对依据。
3. 每份文档用 review 保存当前完整的决议列表和 revision。端点为从 1 起的闭区间。按原序为全文每一行给出 include 或 exclude，两者可覆盖邻接的空行。每次提交会替换该文档的决议，分批保存时先取 status 的已有决议再补充。未核完时用 complete=false，疑点填 unresolved。最终审完后要求 complete=true、unresolved=[]，覆盖从首行到末行。若全是试题或附件杂项，可以全段排除并写依据。

```json
{
  "action": "review",
  "import_id": "<prepare 返回值>",
  "doc": 1,
  "revision": 0,
  "review": {
    "complete": true,
    "unresolved": [],
    "decisions": [
      {"span": [1, 2], "decision": "exclude", "reason": "目录"},
      {"span": [3, 15], "decision": "include", "title": "匀变速直线运动的基本公式"},
      {"span": [16, 25], "decision": "exclude", "reason": "例题与解析，交题目支线"}
    ]
  }
}
```

include 默认取原文。需要更正时在同一决议提交 text 与 basis，沿用原图引用。表格行、公式分隔符及多段推导保留 Markdown 结构。教师版与学生版完全相同的区间对齐为相同正文，工具会合并出处。差异段落分开保留。检查 preview 的 section_count，使用 offset 和 limit 读完各段，核对来源、图表与章节。长资料可按原章在 include 决议指定 module，此时 `知识清单.md` 链接到各模块。图片在该清单的 `DIAGRAMS/`，SHA 和来源区间在 manifest.json。无知识时回执为 no_knowledge。

发布前，讲义支线须已有原件、完整 Markdown 和题库回执。主会话凭 preview_sha256 执行 publish。重连先查 list/status 和原任务的回执，复用确认的工作。可按讲义原件 sha256 调用 list，反查其知识清单。对于同一组原件的新转录版本，确认新旧差异后使用已授权的新预览和 supersedes 指向旧活跃版本。检索时以 list 的活跃回执中的目录为准。双支线的完成条件及资源监控由主调度统一处理。
