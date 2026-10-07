"""Source locking, exact artifact provenance, and conservative figure scoring."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """Evidence is incomplete, ambiguous or no longer matches its identity."""


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    # Never overwrite an earlier run or repair evidence in place.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_caption(text: str) -> str:
    """Ignore whitespace only; do not remove punctuation or rewrite content."""
    return re.sub(r"\s+", "", text)


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise EvidenceError(f"Path escapes evidence root: {relative}")
    return path


def seal(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != root / "checksums.json":
            relative = path.relative_to(root).as_posix()
            contained(root, relative)  # Includes symlink traversal protection.
            files[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    result = {"schema": "figure-evidence-seal-v1", "files": files}
    write_json(root / "checksums.json", result)
    return result


def verify_seal(root: Path) -> dict:
    manifest = read_json(root / "checksums.json")
    if manifest.get("schema") != "figure-evidence-seal-v1" or not manifest.get("files"):
        raise EvidenceError("Unsupported or empty seal")
    failures = []
    for name, expected in manifest["files"].items():
        path = contained(root, name)
        if (not path.is_file() or sha256(path) != expected["sha256"]
                or path.stat().st_size != expected["bytes"]):
            failures.append(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p != root / "checksums.json"}
    failures.extend(sorted(actual - set(manifest["files"])))
    return {"files_checked": len(manifest["files"]), "valid": not failures,
            "failures": sorted(set(failures))}


def validate_corpus(corpus: dict, *, check_files: bool = True) -> None:
    if corpus.get("schema") != "figure-corpus-v1" or not corpus.get("pages"):
        raise EvidenceError("Expected nonempty figure-corpus-v1")
    seen, document_splits = set(), {}
    for page in corpus["pages"]:
        digest, index = page["source_sha256"], page["page_index"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise EvidenceError("Full SHA256 required; prefixes are not identities")
        if type(index) is not int or index < 0:
            raise EvidenceError("page_index must be a nonnegative integer")
        identity = f"{digest}:p{index}"
        if page["page_id"] != identity or identity in seen:
            raise EvidenceError("Page identity mismatch or duplicate")
        seen.add(identity)
        split = page["split"]
        if split not in {"development", "holdout"}:
            raise EvidenceError("Unknown split")
        if digest in document_splits and document_splits[digest] != split:
            raise EvidenceError("Document appears in both development and holdout")
        document_splits[digest] = split
        if check_files:
            source = Path(page["source_path"])
            if not source.is_absolute() or sha256(source) != digest:
                raise EvidenceError(f"Source identity changed: {source}")


def freeze_corpus(spec_path: Path, output: Path) -> dict:
    import fitz

    spec = read_json(spec_path)
    pages, cache = [], {}
    for request in spec["pages"]:
        source = (spec_path.parent / request["source_path"]).resolve()
        if source.suffix.lower() != ".pdf":
            raise EvidenceError("v1 accepts PDF pages; DOCX requires a separate source mapping")
        if source not in cache:
            with fitz.open(source) as doc:
                cache[source] = sha256(source), len(doc)
        digest, count = cache[source]
        index = request["page_index"]
        if type(index) is not int or not 0 <= index < count:
            raise EvidenceError("Page out of source range")
        if request.get("expected_source_sha256", digest) != digest:
            raise EvidenceError("Requested source hash differs from actual source")
        pages.append({"page_id": f"{digest}:p{index}", "source_path": str(source),
                      "source_sha256": digest, "page_index": index,
                      "split": request["split"], "label": request.get("label", ""),
                      "tags": request.get("tags", [])})
    corpus = {"schema": "figure-corpus-v1", "pages": pages}
    validate_corpus(corpus, check_files=False)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "corpus.json", corpus)
    write_json(output / "truth.draft.json", {
        "schema": "figure-truth-v1", "status": "DRAFT", "reviewer": None,
        "corpus_sha256": sha256(output / "corpus.json"),
        "pages": [{"page_id": p["page_id"], "inventory_complete": False,
                   "figures": []} for p in pages],
    })
    seal(output)
    return corpus


def audit_package(package: Path) -> dict:
    """Join identities in final DocumentIR. Geometry is never an identity key."""
    ir_paths = list(package.rglob("document_ir.json"))
    if len(ir_paths) != 1:
        raise EvidenceError("Expected exactly one final document_ir.json")
    document = read_json(ir_paths[0])
    md_path = package / "document.md"
    markdown = md_path.read_text(encoding="utf-8")
    manifest_path = package / "assets_manifest.jsonl"
    assets = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    ir_assets = {}
    for asset in document["assets"]:
        uid = asset["asset_uid"]
        if uid in ir_assets:
            raise EvidenceError(f"Duplicate IR asset identity: {uid}")
        ir_assets[uid] = asset
    blocks = document["blocks"]
    by_node = {b["node_id"]: b for b in blocks}
    if len(by_node) != len(blocks):
        raise EvidenceError("Duplicate document block identity")
    traces = []
    for asset in assets:
        uid = asset["asset_uid"]
        source_content = ir_assets.get(uid, {}).get("provenance", {}).get("source_content_id")
        matches = []
        for block in blocks:
            fields = []
            if block.get("content", {}).get("asset_uid") == uid:
                fields.append("content.asset_uid")
            if source_content and source_content in block.get("source_content_ids", []):
                fields.append("source_content_ids")
            if fields:
                route = block.get("provenance", {}).get("route_provenance", {})
                matches.append({"node_id": block["node_id"], "matching_fields": fields,
                                "source_region_ids": block.get("source_region_ids", []),
                                "adapter": route.get("route_decision", {}).get("adapter")})
        relative = asset.get("relative_path")
        path = contained(package, relative) if relative else None
        actual = sha256(path) if path and path.is_file() else None
        # Exact Markdown target, not substring of another filename or alt text.
        referenced = bool(relative and re.search(
            r"!?\[[^\]\n]*\]\(<?" + re.escape(relative) + r">?(?:\s+\"[^\"]*\")?\)", markdown))
        traces.append({"asset_id": asset["asset_id"], "asset_uid": uid,
                       "relative_path": relative, "actual_sha256": actual,
                       "content_sha256": asset.get("content_sha256"),
                       "sha_matches": actual is not None and actual == asset.get("content_sha256"),
                       "referenced_in_markdown": referenced,
                       "identity_status": "UNIQUE" if len(matches) == 1 else "AMBIGUOUS" if matches else "UNLINKED",
                       "matches": matches})
    captions = []
    for block in blocks:
        if block.get("kind") != "CAPTION":
            continue
        text = block.get("content", {}).get("text") or ""
        norm = normalized_caption(text)
        target = block.get("relations", {}).get("caption_for")
        captions.append({"node_id": block["node_id"], "text": text, "normalized_text": norm,
                         "complete_field_occurrences_in_markdown": normalized_caption(markdown).count(norm) if norm else 0,
                         "caption_for": target, "target_kind": by_node.get(target, {}).get("kind"),
                         "source_completeness": "NOT_ASSESSED_BY_FIELD_PRESERVATION_CHECK"})
    return {"schema": "figure-package-audit-v1", "document_id": document["document_id"],
            "input_sha256": {"document_ir": sha256(ir_paths[0]), "markdown": sha256(md_path),
                             "assets_manifest": sha256(manifest_path)},
            "assets": traces, "captions": captions,
            "summary": {"assets": len(traces), "unique_block_links": sum(a["identity_status"] == "UNIQUE" for a in traces),
                        "assets_with_region_ids": sum(any(m["source_region_ids"] for m in a["matches"]) for a in traces),
                        "sha_matches": sum(a["sha_matches"] for a in traces),
                        "markdown_references": sum(a["referenced_in_markdown"] for a in traces),
                        "quality_verdict": "NOT_SCORED"}}


def score(corpus_path: Path, truth_path: Path, judgments_path: Path, run_root: Path) -> dict:
    """Only reviewed, source/run-bound judgments can support quality metrics."""
    corpus, truth, judgments = map(read_json, (corpus_path, truth_path, judgments_path))
    validate_corpus(corpus, check_files=False)
    if not verify_seal(run_root)["valid"]:
        raise EvidenceError("Run evidence changed")
    run = read_json(run_root / "run.json")
    if run.get("status") != "SUCCESS":
        raise EvidenceError("Cannot score failed conversion")
    corpus_sha = sha256(corpus_path)
    if any(v.get("corpus_sha256") != corpus_sha for v in (truth, judgments, run)):
        raise EvidenceError("Corpus binding mismatch")
    if truth.get("schema") != "figure-truth-v1" or truth.get("status") != "ROOT_REVIEWED":
        raise EvidenceError("Draft/DS truth cannot authorize acceptance")
    if not truth.get("reviewer"):
        raise EvidenceError("Truth reviewer required")
    if (judgments.get("schema") != "figure-judgments-v1"
            or judgments.get("status") != "ROOT_REVIEWED" or not judgments.get("reviewer")
            or judgments.get("truth_sha256") != sha256(truth_path)
            or judgments.get("run_sha256") != sha256(run_root / "run.json")):
        raise EvidenceError("Judgments must bind to reviewed truth and this exact run")
    corpus_pages = {p["page_id"]: p for p in corpus["pages"]}
    selected = run["page_id"]
    pages = {p["page_id"]: p for p in truth["pages"]}
    if len(pages) != len(truth["pages"]) or set(pages) != set(corpus_pages) or selected not in pages:
        raise EvidenceError("Truth page membership differs from corpus")
    page = pages[selected]
    if page.get("inventory_complete") is not True:
        raise EvidenceError("Incomplete inventory cannot measure missing figures")
    figures = {f["figure_id"]: f for f in page["figures"]}
    if len(figures) != len(page["figures"]):
        raise EvidenceError("Duplicate truth figure")
    results = {}
    for row in judgments["figures"]:
        fid = row["figure_id"]
        if row["page_id"] != selected or fid not in figures or fid in results:
            raise EvidenceError("Unknown, duplicated or wrong-page judgment")
        if row["verdict"] not in {"PASS", "FAIL", "MISSING", "UNKNOWN"}:
            raise EvidenceError("Unsupported judgment")
        if not row.get("evidence"):
            raise EvidenceError("Visual judgments require inspectable evidence")
        for evidence in row["evidence"]:
            path = contained(run_root, evidence["path"])
            if sha256(path) != evidence["sha256"]:
                raise EvidenceError("Judgment evidence changed")
        if row.get("automatic_release") is not None and type(row["automatic_release"]) is not bool:
            raise EvidenceError("Invalid automatic release flag")
        if row["verdict"] == "PASS":
            if any(row.get(k) is not True for k in ("body_complete", "clean_crop", "grouping_correct")):
                raise EvidenceError("PASS requires all completeness checks")
            if figures[fid].get("caption_text") and row.get("caption_complete") is not True:
                raise EvidenceError("Caption is part of this figure's acceptance contract")
        results[fid] = row
    if set(results) != set(figures):
        raise EvidenceError("Missing judgments; explicitly record UNKNOWN or MISSING")
    eligible = [results[k] for k, v in figures.items() if v["kind"] in {"science_diagram", "photo"}]
    if any(v["kind"] not in {"science_diagram", "photo", "decoration"} for v in figures.values()):
        raise EvidenceError("Unclassified truth item")
    counts = Counter(r["verdict"] for r in eligible)
    auto = [r for r in eligible if r.get("automatic_release") is True]
    seconds = run.get("timing_seconds", {}).get("convert")
    if seconds is not None and (not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0):
        raise EvidenceError("Invalid measured time")
    count = len(eligible)
    return {"schema": "figure-score-v1", "page_id": selected,
            "corpus_sha256": corpus_sha, "truth_sha256": sha256(truth_path),
            "run_sha256": sha256(run_root / "run.json"),
            "scope": "SINGLE_PAGE_REVIEWED_SAMPLE_NOT_PROJECT_ACCURACY",
            "split": corpus_pages[selected]["split"], "figures": count,
            "counts": dict(counts), "complete_usable_rate": counts["PASS"] / count if count else None,
            "missing_rate": counts["MISSING"] / count if count else None,
            "automatic_release_precision": sum(r["verdict"] == "PASS" for r in auto) / len(auto) if auto else None,
            "review_required_rate": (sum(r.get("automatic_release") is False for r in eligible) / count
                                     if count and all(r.get("automatic_release") is not None for r in eligible) else None),
            "release_status_unknown": sum(r.get("automatic_release") is None for r in eligible),
            "conversion_seconds_per_page": seconds,
            "annotation_identity_note": "Reviewer role is a recorded assertion, not a cryptographic signature."}


def compare_scores(left: list[dict], right: list[dict]) -> dict:
    """Paired quality comparison only; timings remain descriptive, not causal."""
    indexed = []
    for rows in (left, right):
        by_page = {r["page_id"]: r for r in rows}
        if not rows or len(by_page) != len(rows):
            raise EvidenceError("Comparison requires nonempty unique page scores")
        if any(r.get("schema") != "figure-score-v1" for r in rows):
            raise EvidenceError("Unrecognized score schema")
        indexed.append(by_page)
    if set(indexed[0]) != set(indexed[1]):
        raise EvidenceError("Unequal source pages cannot form a paired comparison")
    for key, a in indexed[0].items():
        b = indexed[1][key]
        if any(a.get(field) != b.get(field) for field in ("corpus_sha256", "truth_sha256", "split", "figures")):
            raise EvidenceError("Different corpus, truth, split or denominator")
    splits = {r["split"] for r in left}
    if len(splits) != 1:
        raise EvidenceError("Report development and holdout separately")
    summaries = []
    for rows in (left, right):
        count = sum(r["figures"] for r in rows)
        total = Counter()
        for row in rows:
            total.update(row["counts"])
        seconds = [r["conversion_seconds_per_page"] for r in rows]
        summaries.append({"figures": count, "counts": dict(total),
                          "complete_usable_rate": total["PASS"] / count if count else None,
                          "conversion_seconds_total": sum(seconds) if all(s is not None for s in seconds) else None,
                          "conversion_seconds_per_page": sum(seconds) / len(rows) if all(s is not None for s in seconds) else None})
    return {"schema": "figure-paired-comparison-v1", "pages": len(left), "split": splits.pop(),
            "left": summaries[0], "right": summaries[1],
            "scope": "FIXED_SAMPLE_ONLY; verify hardware, code, parameters and timing mode before causal speed claims"}
