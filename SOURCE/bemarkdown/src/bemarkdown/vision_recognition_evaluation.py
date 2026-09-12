from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

VISION_RECOGNITION_PAGE_SPLIT_SCHEMA = (
    "bemarkdown-vision-recognition-page-split-manifest-v0"
)
VISION_RECOGNITION_PAGE_SPLIT_VERSION = "vision-recognition-page-split-builder-v0"
PINNED_VALIDATION_REFERENCE_ID = "formula-reference-0015"

# Frozen before the first corpus run. Material deviation dominates the complete
# current corpus; the remaining fixed weights make the declared balance tradeoffs
# explicit and reproducible.
VISION_RECOGNITION_OBJECTIVE_WEIGHTS = {
    "material_target_deviation": 1_000_000_000,
    "certain_40_percent_deviation": 1_000_000,
    "acceptable_target_range_deviation": 100_000,
    "document_overlap": 10_000,
    "source_profile_imbalance": 100,
    "density_band_imbalance": 10,
    "page_count_40_percent_deviation": 1,
}

_TRUTH_LEAKAGE_TOKENS = (
    "human_label",
    "human_quality",
    "formulanet_label",
    "ground_truth",
    "gt_latex",
    "reference_latex",
    "expected_correction",
    "known_error_count",
    "material",
    "acceptable",
    PINNED_VALIDATION_REFERENCE_ID,
)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def recognition_input_universe_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "reference_id": str(row["reference_id"]),
            "document_id": str(row["document_id"]),
            "page_index": int(row["page_index"]),
            "label_group": str(row["label_group"]),
            "source_profile": str(row.get("source_profile") or "UNKNOWN"),
            "crop_width": int(row.get("crop_width") or 0),
            "crop_height": int(row.get("crop_height") or 0),
        }
        for row in rows
    ]
    canonical.sort(key=lambda row: row["reference_id"])
    return canonical_json_sha256(canonical)


class VisionRecognitionPageSplitBuilder:
    """Freeze a deterministic, truth-firewalled page-level evaluation split."""

    def __init__(self, objective_weights: dict[str, int] | None = None):
        self.objective_weights = dict(
            objective_weights or VISION_RECOGNITION_OBJECTIVE_WEIGHTS
        )
        if set(self.objective_weights) != set(VISION_RECOGNITION_OBJECTIVE_WEIGHTS):
            raise ValueError("Vision split objective weights must use the frozen keys")
        if any(not isinstance(value, int) or value <= 0 for value in self.objective_weights.values()):
            raise ValueError("Vision split objective weights must be positive integers")

    def build(
        self,
        rows: list[dict[str, Any]],
        *,
        historical_split_sha256: str,
        expected_input_universe_sha256: str | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_rows(rows)
        input_sha = recognition_input_universe_sha256(normalized)
        if expected_input_universe_sha256 and input_sha != expected_input_universe_sha256:
            raise ValueError("Vision Recognition input universe SHA-256 mismatch")

        pages = self._group_pages(normalized)
        pinned = [
            index
            for index, page in enumerate(pages)
            if PINNED_VALIDATION_REFERENCE_ID in page["formula_ids"]
        ]
        if len(pinned) != 1:
            raise ValueError("formula-reference-0015 must occur on exactly one page")
        validation_indexes, objective = self._select_validation_pages(pages, pinned[0])
        validation_set = set(validation_indexes)
        reference_pages = [page for index, page in enumerate(pages) if index not in validation_set]
        validation_pages = [page for index, page in enumerate(pages) if index in validation_set]

        reference_page_ids = [page["page_id"] for page in reference_pages]
        validation_page_ids = [page["page_id"] for page in validation_pages]
        reference_formula_ids = sorted(
            formula_id for page in reference_pages for formula_id in page["formula_ids"]
        )
        validation_formula_ids = sorted(
            formula_id for page in validation_pages for formula_id in page["formula_ids"]
        )
        page_overlap = sorted(set(reference_page_ids) & set(validation_page_ids))
        formula_overlap = sorted(set(reference_formula_ids) & set(validation_formula_ids))
        reference_counts = self._counts(reference_pages)
        validation_counts = self._counts(validation_pages)
        reference_documents = {page["document_id"] for page in reference_pages}
        validation_documents = {page["document_id"] for page in validation_pages}
        document_overlap = sorted(reference_documents & validation_documents)
        material_target_exact = (
            reference_counts["material"] == 4 and validation_counts["material"] == 2
        )

        manifest: dict[str, Any] = {
            "schema": VISION_RECOGNITION_PAGE_SPLIT_SCHEMA,
            "algorithm_version": VISION_RECOGNITION_PAGE_SPLIT_VERSION,
            "input_universe_sha256": input_sha,
            "historical_split_sha256": historical_split_sha256,
            "historical_split_preserved": True,
            "historical_split_mutated": False,
            "hard_constraints": [
                "REFERENCE_VALIDATION_PAGE_DISJOINT",
                "REFERENCE_VALIDATION_FORMULA_DISJOINT",
                "FORMULA_REFERENCE_0015_PAGE_IN_VALIDATION",
                "BOTH_COHORTS_CONTAIN_MATERIAL",
                "PAGE_IS_ATOMIC",
            ],
            "hard_constraints_satisfied": not page_overlap
            and not formula_overlap
            and PINNED_VALIDATION_REFERENCE_ID in validation_formula_ids
            and reference_counts["material"] >= 1
            and validation_counts["material"] >= 1,
            "objective_weights": dict(self.objective_weights),
            "objective": objective,
            "tie_break": "SHA256_OF_SORTED_VALIDATION_PAGE_IDS_THEN_LEXICOGRAPHIC",
            "targets": {
                "reference_material": 4,
                "validation_material": 2,
                "reference_certain_ratio": 0.6,
                "validation_certain_ratio": 0.4,
                "reference_acceptable_range": [65, 70],
                "validation_acceptable_range": [39, 44],
            },
            "counts": {
                "universe": self._counts(pages),
                "reference": reference_counts,
                "validation": validation_counts,
            },
            "reference_page_ids": reference_page_ids,
            "validation_page_ids": validation_page_ids,
            "reference_formula_ids": reference_formula_ids,
            "validation_formula_ids": validation_formula_ids,
            "reference_certain_formula_ids": sorted(
                formula_id
                for page in reference_pages
                for formula_id in page["certain_formula_ids"]
            ),
            "validation_certain_formula_ids": sorted(
                formula_id
                for page in validation_pages
                for formula_id in page["certain_formula_ids"]
            ),
            "uncertain_diagnostics": {
                "reference_formula_ids": sorted(
                    formula_id
                    for page in reference_pages
                    for formula_id in page["uncertain_formula_ids"]
                ),
                "validation_formula_ids": sorted(
                    formula_id
                    for page in validation_pages
                    for formula_id in page["uncertain_formula_ids"]
                ),
                "excluded_from_accuracy_gate": True,
            },
            "reference_document_ids": sorted(reference_documents),
            "validation_document_ids": sorted(validation_documents),
            "document_overlap": document_overlap,
            "document_overlap_count": len(document_overlap),
            "page_overlap": page_overlap,
            "page_overlap_count": len(page_overlap),
            "formula_overlap": formula_overlap,
            "formula_overlap_count": len(formula_overlap),
            "pinned_validation_reference_id": PINNED_VALIDATION_REFERENCE_ID,
            "pinned_validation_page_id": next(
                page["page_id"]
                for page in validation_pages
                if PINNED_VALIDATION_REFERENCE_ID in page["formula_ids"]
            ),
            "page_grouping_constraint": {
                "status": "TARGET_EXACT" if material_target_exact else "PAGE_GROUPING_CONSTRAINT",
                "material_target_exact": material_target_exact,
                "page_split_forbidden": True,
            },
            "page_groups": [
                {
                    **page,
                    "cohort": (
                        "VISION_RECOGNITION_PAGE_VALIDATION"
                        if index in validation_set
                        else "VISION_RECOGNITION_PAGE_REFERENCE"
                    ),
                }
                for index, page in enumerate(pages)
            ],
        }
        if not manifest["hard_constraints_satisfied"]:
            raise RuntimeError("Vision Recognition page split hard constraints failed")
        manifest["split_sha256"] = canonical_json_sha256(manifest)
        return manifest

    @staticmethod
    def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            raise ValueError("Vision Recognition split requires input rows")
        normalized = []
        seen: set[str] = set()
        allowed = {"acceptable", "material", "reference_uncertain"}
        for source in rows:
            reference_id = str(source["reference_id"])
            if reference_id in seen:
                raise ValueError("Vision Recognition reference IDs must be unique")
            seen.add(reference_id)
            label_group = str(source["label_group"])
            if label_group not in allowed:
                raise ValueError(f"Unsupported Vision Recognition label group: {label_group}")
            normalized.append(
                {
                    "reference_id": reference_id,
                    "document_id": str(source["document_id"]),
                    "page_index": int(source["page_index"]),
                    "label_group": label_group,
                    "source_profile": str(source.get("source_profile") or "UNKNOWN"),
                    "crop_width": int(source.get("crop_width") or 0),
                    "crop_height": int(source.get("crop_height") or 0),
                }
            )
        return sorted(normalized, key=lambda row: row["reference_id"])

    @staticmethod
    def _group_pages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["document_id"], row["page_index"])].append(row)
        pages = []
        for (document_id, page_index), page_rows in sorted(grouped.items()):
            label_counts = Counter(row["label_group"] for row in page_rows)
            certain = label_counts["acceptable"] + label_counts["material"]
            density_band = (
                "SINGLE"
                if certain <= 1
                else "LOW"
                if certain <= 4
                else "MEDIUM"
                if certain <= 8
                else "HIGH"
            )
            profiles = Counter(row["source_profile"] for row in page_rows)
            source_profile = min(
                profiles,
                key=lambda value: (-profiles[value], value),
            )
            pages.append(
                {
                    "page_id": f"{document_id}:{page_index}",
                    "document_id": document_id,
                    "page_index": page_index,
                    "source_profile": source_profile,
                    "density_band": density_band,
                    "formula_ids": sorted(row["reference_id"] for row in page_rows),
                    "certain_formula_ids": sorted(
                        row["reference_id"]
                        for row in page_rows
                        if row["label_group"] != "reference_uncertain"
                    ),
                    "uncertain_formula_ids": sorted(
                        row["reference_id"]
                        for row in page_rows
                        if row["label_group"] == "reference_uncertain"
                    ),
                    "acceptable": label_counts["acceptable"],
                    "material": label_counts["material"],
                    "uncertain": label_counts["reference_uncertain"],
                    "certain": certain,
                }
            )
        return pages

    @staticmethod
    def _counts(pages: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "pages": len(pages),
            "formulas": sum(len(page["formula_ids"]) for page in pages),
            "certain": sum(page["certain"] for page in pages),
            "acceptable": sum(page["acceptable"] for page in pages),
            "material": sum(page["material"] for page in pages),
            "uncertain": sum(page["uncertain"] for page in pages),
        }

    def _select_validation_pages(
        self, pages: list[dict[str, Any]], pinned_index: int
    ) -> tuple[list[int], dict[str, Any]]:
        total = self._counts(pages)
        target_certain = round(total["certain"] * 0.4)
        target_pages = round(total["pages"] * 0.4)
        target_material = 2
        free_indexes = [index for index in range(len(pages)) if index != pinned_index]
        documents = sorted({page["document_id"] for page in pages})
        document_index = {value: index for index, value in enumerate(documents)}
        document_totals = Counter(page["document_id"] for page in pages)
        profiles = sorted({page["source_profile"] for page in pages})
        densities = sorted({page["density_band"] for page in pages})
        profile_totals = Counter(page["source_profile"] for page in pages)
        density_totals = Counter(page["density_band"] for page in pages)

        selected_document_counts = [0] * len(documents)
        selected_profile_counts = Counter()
        selected_density_counts = Counter()
        selected_indexes = {pinned_index}
        pinned_page = pages[pinned_index]
        selected_document_counts[document_index[pinned_page["document_id"]]] = 1
        selected_profile_counts[pinned_page["source_profile"]] = 1
        selected_density_counts[pinned_page["density_band"]] = 1
        material = pinned_page["material"]
        certain = pinned_page["certain"]
        acceptable = pinned_page["acceptable"]
        selected_page_count = 1

        best_key: tuple[Any, ...] | None = None
        best_indexes: list[int] | None = None
        best_objective: dict[str, Any] | None = None
        previous_gray = 0
        for value in range(1 << len(free_indexes)):
            gray = value ^ (value >> 1)
            if value:
                changed = gray ^ previous_gray
                bit = changed.bit_length() - 1
                page_index = free_indexes[bit]
                page = pages[page_index]
                direction = 1 if gray & changed else -1
                if direction > 0:
                    selected_indexes.add(page_index)
                else:
                    selected_indexes.remove(page_index)
                material += direction * page["material"]
                certain += direction * page["certain"]
                acceptable += direction * page["acceptable"]
                selected_page_count += direction
                selected_document_counts[document_index[page["document_id"]]] += direction
                selected_profile_counts[page["source_profile"]] += direction
                selected_density_counts[page["density_band"]] += direction
            previous_gray = gray

            if material < 1 or material >= total["material"]:
                continue
            acceptable_range_deviation = (
                39 - acceptable
                if acceptable < 39
                else acceptable - 44
                if acceptable > 44
                else 0
            )
            document_overlap = sum(
                0 < selected_document_counts[index] < document_totals[document]
                for index, document in enumerate(documents)
            )
            profile_imbalance = sum(
                abs(2 * selected_profile_counts[profile] - profile_totals[profile])
                for profile in profiles
            )
            density_imbalance = sum(
                abs(2 * selected_density_counts[density] - density_totals[density])
                for density in densities
            )
            components = {
                "material_target_deviation": abs(material - target_material),
                "certain_40_percent_deviation": abs(certain - target_certain),
                "acceptable_target_range_deviation": acceptable_range_deviation,
                "document_overlap": document_overlap,
                "source_profile_imbalance": profile_imbalance,
                "density_band_imbalance": density_imbalance,
                "page_count_40_percent_deviation": abs(selected_page_count - target_pages),
            }
            weighted_score = sum(
                components[name] * self.objective_weights[name] for name in components
            )
            validation_page_ids = sorted(pages[index]["page_id"] for index in selected_indexes)
            tie_sha = hashlib.sha256("\n".join(validation_page_ids).encode("utf-8")).hexdigest()
            key = (weighted_score, tie_sha, tuple(validation_page_ids))
            if best_key is None or key < best_key:
                best_key = key
                best_indexes = sorted(selected_indexes)
                best_objective = {
                    "weighted_score": weighted_score,
                    "components": components,
                    "validation_page_ids_sha256": tie_sha,
                    "searched_assignments": 1 << len(free_indexes),
                    "search": "EXHAUSTIVE_GRAY_CODE_PAGE_ASSIGNMENT",
                }
        if best_indexes is None or best_objective is None:
            raise ValueError("No Vision Recognition split satisfies the hard constraints")
        return best_indexes, best_objective


def inspect_vision_evaluation_bundle(
    bundle_root: str | Path,
    *,
    expected_page_ids: set[str],
    expected_formula_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Inspect request membership, local images, relative paths, and truth leakage."""
    root = Path(bundle_root)
    request_paths = sorted((root / "requests").glob("*.json"))
    page_ids: list[str] = []
    formula_ids: list[str] = []
    truth_leakage: list[dict[str, str]] = []
    absolute_paths: list[dict[str, str]] = []
    missing_images: list[dict[str, str]] = []
    image_sha_mismatches: list[dict[str, str]] = []
    duplicate_audit_ids: list[str] = []
    audit_ids: set[str] = set()

    for request_path in request_paths:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        audit_id = str(request.get("audit_id", ""))
        if audit_id in audit_ids:
            duplicate_audit_ids.append(audit_id)
        audit_ids.add(audit_id)
        page_ids.append(f"{request['document_id']}:{int(request['page_index'])}")
        formula_ids.extend(str(value) for value in request.get("known_formula_ids", []))
        lowered = request_path.read_text(encoding="utf-8").lower()
        for token in _TRUTH_LEAKAGE_TOKENS:
            if token.lower() in lowered:
                truth_leakage.append({"request": request_path.name, "token": token})
        assets = list(request.get("images", {}).values()) + list(
            request.get("formula_sheet_images", [])
        )
        for asset in assets:
            value = str(asset.get("path", ""))
            if (
                PureWindowsPath(value).is_absolute()
                or PurePosixPath(value).is_absolute()
                or ".." in PurePosixPath(value).parts
            ):
                absolute_paths.append({"request": request_path.name, "path": value})
                continue
            target = root / PurePosixPath(value)
            if not target.is_file():
                missing_images.append({"request": request_path.name, "path": value})
            elif asset.get("sha256"):
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                if actual != asset["sha256"]:
                    image_sha_mismatches.append(
                        {"request": request_path.name, "path": value}
                    )

    page_overlap = sorted(set(page_ids) - expected_page_ids)
    missing_pages = sorted(expected_page_ids - set(page_ids))
    unexpected_formulas = (
        sorted(set(formula_ids) - expected_formula_ids)
        if expected_formula_ids is not None
        else []
    )
    missing_formulas = (
        sorted(expected_formula_ids - set(formula_ids))
        if expected_formula_ids is not None
        else []
    )
    passed = not any(
        (
            truth_leakage,
            absolute_paths,
            missing_images,
            image_sha_mismatches,
            duplicate_audit_ids,
            page_overlap,
            missing_pages,
            unexpected_formulas,
            missing_formulas,
        )
    )
    return {
        "schema": "bemarkdown-vision-evaluation-bundle-integrity-v0",
        "bundle": root.name,
        "request_count": len(request_paths),
        "page_count": len(set(page_ids)),
        "formula_count": len(formula_ids),
        "truth_leakage_count": len(truth_leakage),
        "truth_leakage": truth_leakage,
        "absolute_path_count": len(absolute_paths),
        "absolute_paths": absolute_paths,
        "missing_image_count": len(missing_images),
        "missing_images": missing_images,
        "image_sha_mismatch_count": len(image_sha_mismatches),
        "image_sha_mismatches": image_sha_mismatches,
        "duplicate_audit_ids": sorted(duplicate_audit_ids),
        "unexpected_page_ids": page_overlap,
        "missing_page_ids": missing_pages,
        "unexpected_formula_ids": unexpected_formulas,
        "missing_formula_ids": missing_formulas,
        "passed": passed,
    }
