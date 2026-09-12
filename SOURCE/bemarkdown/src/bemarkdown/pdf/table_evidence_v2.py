"""Position-aware table evidence comparison."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Mapping
from typing import Any


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def _html_table(value: str) -> dict[str, Any] | None:
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", value, re.IGNORECASE | re.DOTALL)
    if not rows:
        return None
    cells: list[dict[str, Any]] = []
    max_column = 0
    for row_index, row in enumerate(rows):
        column = 0
        for match in re.finditer(
            r"<t[dh]\b([^>]*)>(.*?)</t[dh]>", row, re.IGNORECASE | re.DOTALL
        ):
            attrs, content = match.groups()
            colspan_match = re.search(
                r"colspan\s*=\s*['\"]?(\d+)", attrs, re.IGNORECASE
            )
            rowspan_match = re.search(
                r"rowspan\s*=\s*['\"]?(\d+)", attrs, re.IGNORECASE
            )
            colspan = int(colspan_match.group(1)) if colspan_match else 1
            rowspan = int(rowspan_match.group(1)) if rowspan_match else 1
            cells.append(
                {
                    "row": row_index,
                    "column": column,
                    "rowspan": rowspan,
                    "colspan": colspan,
                    "text": _text(html.unescape(re.sub(r"<[^>]+>", "", content))),
                }
            )
            column += colspan
        max_column = max(max_column, column)
    return {"rows": len(rows), "columns": max_column, "cells": cells}


def normalize_table_v2(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        return _html_table(value)
    if not isinstance(value, Mapping):
        return None
    raw_cells = value.get("cells", [])
    declared_columns = int(value.get("columns", 0) or 0)
    cells: list[dict[str, Any]] = []
    if raw_cells and all(isinstance(row, (list, tuple)) for row in raw_cells):
        for row_index, row in enumerate(raw_cells):
            for column, cell in enumerate(row):
                cells.append(
                    {
                        "row": row_index,
                        "column": column,
                        "rowspan": 1,
                        "colspan": 1,
                        "text": _text(cell),
                    }
                )
    elif isinstance(raw_cells, list):
        for index, cell in enumerate(raw_cells):
            if isinstance(cell, Mapping):
                cells.append(
                    {
                        "row": int(cell.get("row", cell.get("row_index", 0))),
                        "column": int(
                            cell.get("column", cell.get("column_index", index))
                        ),
                        "rowspan": int(cell.get("rowspan", 1)),
                        "colspan": int(cell.get("colspan", 1)),
                        "text": _text(cell.get("text", cell.get("content", ""))),
                    }
                )
            else:
                row_index = index // declared_columns if declared_columns else 0
                column_index = index % declared_columns if declared_columns else index
                cells.append(
                    {
                        "row": row_index,
                        "column": column_index,
                        "rowspan": 1,
                        "colspan": 1,
                        "text": _text(cell),
                    }
                )
    rows = int(value.get("rows", max((cell["row"] for cell in cells), default=-1) + 1))
    columns = declared_columns or max(
        (cell["column"] for cell in cells), default=-1
    ) + 1
    return {"rows": rows, "columns": columns, "cells": cells}


def compare_table_evidence_v2(first: Any, second: Any) -> dict[str, Any]:
    left = normalize_table_v2(first)
    right = normalize_table_v2(second)
    if (
        left is None
        or right is None
        or (left["rows"], left["columns"]) != (right["rows"], right["columns"])
    ):
        state = "TABLE_STRUCTURE_CONFLICT"
    else:
        left_topology = [
            (c["row"], c["column"], c["rowspan"], c["colspan"]) for c in left["cells"]
        ]
        right_topology = [
            (c["row"], c["column"], c["rowspan"], c["colspan"]) for c in right["cells"]
        ]
        left_positioned = [(c["row"], c["column"], c["text"]) for c in left["cells"]]
        right_positioned = [(c["row"], c["column"], c["text"]) for c in right["cells"]]
        if left_topology != right_topology:
            state = "TABLE_MERGED_CELL_TOPOLOGY_CONFLICT"
        elif left_positioned == right_positioned:
            state = "TABLE_MATCH"
        elif sorted(c["text"] for c in left["cells"]) == sorted(
            c["text"] for c in right["cells"]
        ):
            state = "TABLE_CELL_POSITION_CONFLICT"
        else:
            state = "TABLE_CELL_CONTENT_CONFLICT"
    return {
        "schema": "bemarkdown-table-evidence-comparison-v2",
        "state": state,
        "conflict_types": [] if state == "TABLE_MATCH" else [state],
        "authoritative_rewrite": None,
    }
