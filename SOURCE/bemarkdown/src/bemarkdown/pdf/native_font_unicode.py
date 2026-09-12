"""Identify CID fonts whose extracted character codes have no Unicode mapping."""

from __future__ import annotations

import re


def normalized_font_name(name: str) -> str:
    return re.sub(r"^[A-Z]{6}\+", "", name.lstrip("/"))


def native_font_unicode_issues(page, cache: dict[int, bool]) -> dict[str, dict]:
    """Flag missing ToUnicode only for Identity-encoded Type0 fonts.

    Identity-H/V maps character codes to CIDs, not Unicode. MuPDF can return
    printable unrelated characters for such fonts, even on a high-trust page.
    Named standard CMaps and ordinary Latin fonts retain their existing path.
    The caller owns a fresh cache for each open source document.
    """
    issues = {}
    for font in page.get_fonts(full=True):
        xref, _, kind, base_name, _, encoding, *_ = font
        if kind != "Type0" or encoding not in {"Identity-H", "Identity-V"}:
            continue
        if xref not in cache:
            mapping_type, _ = page.parent.xref_get_key(xref, "ToUnicode")
            cache[xref] = mapping_type == "null"
        if cache[xref]:
            name = normalized_font_name(base_name)
            issues[name] = {"reason": "IDENTITY_CID_FONT_WITHOUT_TOUNICODE",
                            "font_name": name, "font_xref": xref, "encoding": encoding}
    return issues
