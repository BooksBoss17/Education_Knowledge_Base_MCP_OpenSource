---
name: exercise-import
description: 将物理试卷、作业、讲义及配套解析导入题库，同时保存每份原件的完整 Markdown。用于用户要求录题、拆题或将题目存入 SQLite 的任务。
---

# 题目导入

主会话负责资料组、转录作业与资源监控。每组交接原件、来源角色、成功的 BeMarkdown job_id 及同一个 task_run_id。逐份保存清单；断开后用 runtime_jobs 查已有作业，用 exercise_import(list/status) 找回审查进度。

## 审查与保存

1. 使用 exercise_import(prepare)，sources 按固定顺序传入 job_id 和 role（题卷、答案、作业、讲义、解析版）。同一套题卷与答案卷放在同一组。lines 分页读取每份全文及稳定行号，status 给出按编号初筛的候选和同章节、同编号的候选答案行，逐题核对归属。断点接续时可用 draft(question_key) 读取具体决议，status 的 receipt 表示该套已发布。按原件顺序通读全文，补上漏检的例题或无编号题。跨页、小问与共用条件保持在同一题；独立变式另建题。原卷与教师版的章节、编号和题意共同决定配对。答案表的一行可为多题提供证据。
2. 逐题核对完整题干、选项、图片与解析。遇到转录歧义时查看原件中的对应图文或公式。只在有来源证据时纠正文本。先检查配套答案卷、卷末答案表、例题后的解答以及每个小问。确认原件缺答案或详解时，由审题 agent 依据完整题干、选项与题图解答并补写缺项。已有原文答案和详解照录。详解写出关键物理依据、计算或选项判断；按题型检查方向、单位、代回结果或核对正确选项。若题图缺条件、答案配对未明或独立推导和原答案矛盾，记入 unresolved，回源处理后再发布。全文保留资料原顺序和所有教学内容。
3. 审题 agent 用 labels(query) 查词表，按实际设问及解题依据选择四级叶标签，并标主考、关联。小问可单独指定 scope。实验背景、错误选项和整卷标题不能自行决定考点。类型只取单选、多选、填空、实验、计算、判断、作图、证明之一。实验大题内的小问共享题号。
4. 用 review 分批保存决议；每次携带 status 的 revision，过期时先刷新。questions 的键为本套内稳定临时编号。一个题的 parts 以有序片段分别保存 stem、options、answer、explanation。每个 span 是 [文档序号,首行,末行]（从 1 起、两端包含）。选项与答案均按原序，片段默认直接取原文。若一行混有多段、正文编号需剥离或转录有误，提供 text 和 basis 指明依据。保留图片的 Markdown 引用。相关行可复用；不同题的共用材料分别放入 stem。末次对账确认每个原题号、题图、选项、小问均已归属。讲义中的编号知识点、解题步骤等误检以 excluded 的“文档序号:行号”记录理由。

补写内容使用独立片段，例如 `{"origin":"agent","text":"$v=6\\,m/s$","verification":"从静止以 2 m/s² 加速 3 s，代入 v=at 并核对单位。"}`，放入 parts.answer 或 parts.explanation。每个补写片段均写具体复核依据，不填 span。部分小问缺项时，保留其余小问的原文片段。只要答案含补写，answer_status 填 `Agent补写`，只补详解时答案仍为 `已收录`。补写保存到题库，与原件全文分开。

请求示例（标签和行号仅展示格式）：

```json
{"action":"review","import_id":"<prepare 返回的编号>","revision":0,"review":{"questions":{"第一章-1":{"section":"第一章","number":"1","type":"单选","parts":{"stem":[{"span":[1,10,11]}],"options":[{"span":[1,12,15]}],"answer":[{"span":[2,8,8],"text":"B","basis":"答案表第1题"}],"explanation":[{"span":[2,18,21]}]},"answer_status":"已收录","tags":[{"leaf":"洛伦兹力","importance":"主考","scope":"整题"}]}}}}
```

5. 读完所有原件并解决所有疑点后提交 coverage_confirmed=true 和 unresolved=[]。preview 校验候选覆盖并返回 SHA。核对类型、题号、标签和图片数量，使用该 preview_sha256 调用 publish。重复提交同一预览应返回同一回执。若还有未解疑点，仅保存草稿，向主会话报告。
6. 逐题对账回执。每份完整转录见 ASSIGNMENTS_EXAMS_AND_HANDOUTS 分类内原件专属目录的全文.md、DIAGRAMS 和 manifest.json；原件见 Original_Backup。SQLite 位于 EXERCISE_BANK_SQLite/exercises.sqlite3，题号为 11 位文本，0 至 7 依次对应上述八类。sources 保存行锚点；题图保存为 images 的 BLOB，用 asset:哈希引用。主会话据实际结果统一收尾。

开发区隔离测试也可使用同一工具的 JSON 命令行入口：python exercise_import.py --workspace 知识库路径 --input-root 授权原件目录 --request 请求.json。命令行与 MCP 共用入库实现；不直接用 SQL 写题库。

