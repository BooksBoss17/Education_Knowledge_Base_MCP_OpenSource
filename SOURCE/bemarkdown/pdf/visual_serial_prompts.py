"""Versioned prompts for the visual serial transcription route.

PP-DocLayoutV3 is a detector and receives no generation prompt.  The two
recognizers run serially. Usable Xiaomi formula results are masked before
Ovis sees text crops; Ovis preserves all remaining visible content.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final


PROMPT_SCHEMA: Final[str] = "bemarkdown-visual-serial-prompts-v2"

XIAOMI_FORMULA_PROMPT: Final[str] = (
    "转录输入裁剪框内的可见内容，按原图顺序保留普通文字、数字、标点、数学公式及公式编号。"
    "只有公式时直接输出 LaTeX；文字和公式混排时，普通文字直接输出，每段公式用成对的 $ 或 $$ 包围。"
    "公式按统一格式输出：分式使用 \\frac，双重积分使用 \\iint，"
    "单位和直立字母使用 \\mathrm{}，函数名使用 \\operatorname{}，绝对值使用 \\mid；"
    "只有输入确实是多行结构时才使用 array/cases，不要给单行公式包 array 或 align。"
    "保留根号、上下标、希腊字母、运算符、括号、矩阵、对齐关系和公式编号。"
    "不要补全被裁掉的内容，"
    "不要改正原图，不要解释，不要使用 Markdown 代码围栏。若裁剪框内没有完整"
    "公式，输出空字符串，让后续文字识别处理原图。"
)

OVISOCR3_TEXT_PROMPT: Final[str] = (
    "你是教材页面文字转录器。只转录输入裁剪框内实际可见的文字、数字、标点、"
    "标题、图注和数学表达，按阅读顺序输出。已成功处理的公式可能被涂白；"
    "不要补写空白区域。不要跳过仍然可见的公式、变量、上下标或公式编号；"
    "数学表达使用 LaTeX 并用 $ 或 $$ 成对包围，普通文字不要改写成 LaTeX。"
    "保留原文大小写、数字、单位、标点、项目符号和换行层次；"
    "不要推测缺失内容，不要纠错，不要解释，不要输出 Markdown 代码围栏。"
)

# Some upstream annotated TEXT regions intentionally preserve a mixed TeX/text
# serialization.  They are a different contract from ordinary prose: dropping
# formula spans there creates systematic omissions even though the visual
# formula stage is still kept disjoint at the region level.
OVISOCR3_MIXED_TEXT_PROMPT: Final[str] = (
    "你是严格的教材混合文本转录器。只输出输入裁剪框内可见内容，不要解释、翻译、"
    "补全或改正。本输入的上游标注要求保留普通文字与数学 TeX 混合序列：不要跳过公式、变量、"
    "上下标、符号或公式编号；按原图阅读顺序逐字转录，保留原始大小写、语言、数字、"
    "标点、空格层次、项目符号以及 TeX 控制词、花括号和分隔符。不要把 TeX 转成 Unicode，"
    "也不要把普通文字改写成 LaTeX。只输出转录结果，不要 Markdown 代码围栏。"
)


OVISOCR3_TABLE_PROMPT: Final[str] = (
    "识别图片中的完整表格，只输出一个HTML table，不要解释或代码围栏。"
    "保留原行列、空单元格、表头和合并单元格，合并使用rowspan/colspan。"
    "逐格忠实转录可见文字、数字、单位和公式；单元格中的数学使用成对$包围的LaTeX。"
    "不得猜补、纠错、重复行或把单元格内容移到表外。"
    "只允许table/thead/tbody/tfoot/tr/th/td/br/sup/sub标签，所有标签显式闭合，换行使用<br/>；"
    "仅td/th可有rowspan/colspan属性。特殊字符使用XML转义。没有可辨认表格则输出空字符串。"
)

PROMPTS: Final[dict[str, str]] = {
    "ovisocr3-table": OVISOCR3_TABLE_PROMPT,
    "xiaomi-ocr-0": XIAOMI_FORMULA_PROMPT,
    "ovisocr3": OVISOCR3_TEXT_PROMPT,
    "ovisocr3-mixed-text": OVISOCR3_MIXED_TEXT_PROMPT,
    # Kept only so the untouched legacy replay path retains its old worker
    # contract while the new policy selects ``ovisocr3`` explicitly.
    "ovis-ocr2": (
        "Transcribe all visible text in the entire image faithfully in reading order. "
        "Preserve every visible number, sign, label, arrow and punctuation. "
        "Preserve two-dimensional mathematical structure using LaTeX. "
        "Do not infer missing content or correct the source. Output only the transcription."
    ),
}


def prompt_for_request(model_id: str, request: dict[str, object] | None = None) -> str:
    """Select the text contract from answer-free input metadata."""

    request = request or {}
    if model_id == "ovisocr3" and request.get("serialization_profile") == "TABLE_HTML_V1":
        return OVISOCR3_TABLE_PROMPT
    if model_id == "ovisocr3" and request.get("serialization_profile") == "UPSTREAM_TEX_MIXED_V1":
        return OVISOCR3_MIXED_TEXT_PROMPT
    return prompt_for(model_id)


def prompt_for(model_id: str) -> str:
    try:
        return PROMPTS[model_id]
    except KeyError as exc:
        raise ValueError(f"VISUAL_SERIAL_PROMPT_MODEL_UNSUPPORTED:{model_id}") from exc


def prompt_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prompt_fingerprint(model_id: str) -> str:
    payload = f"{PROMPT_SCHEMA}\0{model_id}\0{prompt_for(model_id)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_prompt_contract() -> None:
    """Reject accidental prompt drift that would reintroduce formula overlap."""

    formula = XIAOMI_FORMULA_PROMPT.lower()
    text = OVISOCR3_TEXT_PROMPT.lower()
    if "latex" not in formula or "公式" not in formula:
        raise ValueError("VISUAL_SERIAL_FORMULA_PROMPT_INCOMPLETE")
    if "代码围栏" not in formula or "成对的 $ 或 $$" not in formula:
        raise ValueError("VISUAL_SERIAL_FORMULA_DELIMITER_GUARD_MISSING")
    if "不要跳过仍然可见" not in text or "不要补写空白区域" not in text:
        raise ValueError("VISUAL_SERIAL_TEXT_PROMPT_CONTENT_GUARD_MISSING")
    mixed = OVISOCR3_MIXED_TEXT_PROMPT
    if "不要跳过公式" not in mixed or "TeX" not in mixed:
        raise ValueError("VISUAL_SERIAL_MIXED_TEXT_PROMPT_INCOMPLETE")
    if "裁剪框内的可见内容" not in formula or "只转录" not in text:
        raise ValueError("VISUAL_SERIAL_PROMPT_SCOPE_GUARD_MISSING")


validate_prompt_contract()

