"""Deterministic multi-source reading-order precedence graph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import combinations, pairwise
from typing import Any


@dataclass(frozen=True, slots=True)
class ReadingOrderResult:
    schema: str
    resolved_order: tuple[str, ...]
    rollback_order: tuple[str, ...]
    precedence_edges: tuple[tuple[str, str, str], ...]
    risk_types: tuple[str, ...]
    automatic_reorder_applied: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["resolved_order"] = list(self.resolved_order)
        result["rollback_order"] = list(self.rollback_order)
        result["precedence_edges"] = [list(edge) for edge in self.precedence_edges]
        result["risk_types"] = list(self.risk_types)
        return result


def _source_order(blocks: list[Mapping[str, Any]], key: str) -> tuple[str, ...]:
    present = [row for row in blocks if row.get(key) is not None]
    return tuple(
        str(row["node_id"])
        for row in sorted(present, key=lambda row: (row[key], str(row["node_id"])))
    )


def _has_cycle(edges: list[tuple[str, str, str]]) -> bool:
    graph: dict[str, list[str]] = {}
    for source, target, _reason in edges:
        graph.setdefault(source, []).append(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(target) for target in graph.get(node, [])):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _orders_invert(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
    common = set(first) & set(second)
    first_positions = {node: index for index, node in enumerate(first)}
    second_positions = {node: index for index, node in enumerate(second)}
    return any(
        (first_positions[left] - first_positions[right])
        * (second_positions[left] - second_positions[right])
        < 0
        for left, right in combinations(sorted(common), 2)
    )


def _relation_is_reversed(
    orders: Iterable[tuple[str, ...]], expected_left: str, expected_right: str
) -> bool:
    for order in orders:
        if expected_left not in order or expected_right not in order:
            continue
        if order.index(expected_left) > order.index(expected_right):
            return True
    return False


class ReadingOrderResolverV2:
    def resolve(self, blocks: Iterable[Mapping[str, Any]]) -> ReadingOrderResult:
        rows = list(blocks)
        risks: list[str] = []
        edges: list[tuple[str, str, str]] = []
        native = _source_order(rows, "native_order")
        vl = _source_order(rows, "vl_order")
        plus = _source_order(rows, "plus_l_order")
        non_empty = [order for order in (native, vl, plus) if order]
        if len(set(non_empty)) > 1:
            risks.append("ORDER_SOURCE_DISAGREEMENT")
        if any(
            _orders_invert(first, second)
            for first, second in combinations(non_empty, 2)
        ):
            risks.append("ORDER_INVERSION")
        rollback = vl or native or tuple(str(row["node_id"]) for row in rows)

        for order, source_name in ((native, "NATIVE"), (vl, "VL"), (plus, "PLUS_L")):
            edges.extend(
                (left, right, source_name) for left, right in pairwise(order)
            )
        known_ids = {str(row["node_id"]) for row in rows}
        for row in rows:
            relation = row.get("relation")
            if not isinstance(relation, Mapping):
                continue
            target = str(relation.get("target", ""))
            relation_type = str(relation.get("type", "")).lower()
            node_id = str(row["node_id"])
            if target not in known_ids:
                risks.append("ORPHAN_BLOCK")
            elif relation_type == "caption":
                if _relation_is_reversed(non_empty, target, node_id):
                    risks.append("CAPTION_RELATION_CONFLICT")
                edges.append((target, node_id, "FIGURE_CAPTION_RELATION"))
            elif relation_type == "title":
                if _relation_is_reversed(non_empty, node_id, target):
                    risks.append("TITLE_RELATION_CONFLICT")
                edges.append((node_id, target, "TABLE_TITLE_RELATION"))

        if _has_cycle(edges):
            risks.append("ORDER_CYCLE")
        ordered_by_native = [
            next(row for row in rows if str(row["node_id"]) == node)
            for node in (native or rollback)
        ]
        centers = [
            (float(row["bbox_pdf_pt"][0]) + float(row["bbox_pdf_pt"][2])) / 2
            for row in ordered_by_native
        ]
        if len(centers) >= 3:
            split = (min(centers) + max(centers)) / 2
            columns = [0 if center < split else 1 for center in centers]
            if sum(left != right for left, right in pairwise(columns)) > 1:
                risks.append("CROSS_COLUMN_JUMP")

        geometry = tuple(
            str(row["node_id"])
            for row in sorted(
                rows,
                key=lambda row: (
                    round(float(row["bbox_pdf_pt"][0]) / 150),
                    float(row["bbox_pdf_pt"][1]),
                    float(row["bbox_pdf_pt"][0]),
                    str(row["node_id"]),
                ),
            )
        )
        return ReadingOrderResult(
            schema="bemarkdown-reading-order-resolution-v2",
            resolved_order=geometry,
            rollback_order=rollback,
            precedence_edges=tuple(dict.fromkeys(edges)),
            risk_types=tuple(dict.fromkeys(risks)),
            automatic_reorder_applied=not risks and geometry != rollback,
        )
