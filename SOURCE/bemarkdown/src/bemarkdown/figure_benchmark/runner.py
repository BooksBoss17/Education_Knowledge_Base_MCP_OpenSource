"""Single-process observer of the actual production call; no stage reconstruction."""
from __future__ import annotations

import json
import tempfile
import time
import traceback
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from .core import EvidenceError, audit_package, read_json, seal, sha256, validate_corpus, write_json


def event(root: Path, kind: str, **fields) -> None:
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                                 "kind": kind, **fields}, ensure_ascii=False) + "\n")


@contextmanager
def observe_layout(module, output: Path):
    """Wrap the real function, snapshot its return, return the same object.

    For the isolated benchmark CLI process only: global monkeypatching must not
    be used in a concurrent production server. No algorithms or inputs change.
    """
    original = module._run_layout
    observations = []

    def observed(*args, **kwargs):
        started = time.perf_counter()
        value = original(*args, **kwargs)
        elapsed = time.perf_counter() - started
        number = len(observations)
        folder = output / f"layout-{number:03d}"
        folder.mkdir()
        renders = []
        for i, render in enumerate(kwargs["renders"]):
            suffix = ".png" if render.image_bytes.startswith(b"\x89PNG") else ".jpg"
            path = folder / f"page-{i:04d}{suffix}"
            path.write_bytes(render.image_bytes)
            renders.append({"page_index": render.page_index,
                            "path": path.relative_to(output).as_posix(), "sha256": sha256(path),
                            "transform": render.transform.to_dict()})
        # Serialize now, before downstream code can mutate the returned objects.
        write_json(folder / "stage.json", {
            "schema": "figure-layout-observation-v1", "origin": "ACTUAL_PRODUCTION_RETURN",
            "stage": "bemarkdown.pdf.production_runtime._run_layout",
            "not_final_fusion_ir": True, "layout_identity": kwargs["layout_identity"],
            "raw_sdk_pages": value[0], "region_pages": value[1], "renders": renders,
            "seconds": elapsed,
        })
        observations.append({"stage_path": (folder / "stage.json").relative_to(output).as_posix(),
                             "layout_identity": kwargs["layout_identity"], "seconds": elapsed,
                             "sdk_inference_seconds": sum(float(p.get("inference_seconds") or 0) for p in value[0])})
        return value

    with patch.object(module, "_run_layout", observed):
        yield observations


def capture_run(corpus_path: Path, page_id: str, output: Path, *, mcp_root: Path, work_root: Path,
                layout_runtime_factory=None) -> dict:
    """Existing plus-L by default; injected models share the same downstream path.

    No model installation/download is performed. Alternative providers implement
    the existing production layout protocol (load/predict_image/unload), and must
    report their real fingerprint/configuration in load().
    """
    import fitz
    from bemarkdown.pdf import production_runtime
    from bemarkdown.pdf.modular_pipeline import ModularPdfPipeline

    corpus_path = corpus_path.resolve()
    corpus = read_json(corpus_path)
    validate_corpus(corpus, check_files=False)
    selected = [p for p in corpus["pages"] if p["page_id"] == page_id]
    if len(selected) != 1:
        raise EvidenceError("Select an exact frozen page_id")
    page = selected[0]
    source = Path(page["source_path"])
    if sha256(source) != page["source_sha256"]:
        raise EvidenceError("Source changed after corpus freeze")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "corpus.snapshot.json", corpus)
    event(output, "start", page_id=page_id)
    started = time.perf_counter()
    run = {"schema": "figure-run-v1", "status": "RUNNING", "page_id": page_id,
           "corpus_sha256": sha256(corpus_path), "source_sha256": page["source_sha256"],
           "parent_page_index": page["page_index"], "derived_page_index": 0,
           "mode": "ONE_PAGE_FRESH_RUNTIME", "timing_seconds": {},
           "quality_status": "UNREVIEWED"}
    temporary = None
    previous_tempdir = tempfile.tempdir
    runtime = None
    failure = None
    try:
        # Keep production's long OCR filenames independent of evidence depth.
        work_root = work_root.resolve()
        expected = (work_root / "r-12345678" / "bmdpdf-123456789abc-12345678" /
                    "content/transient_text_only" / ("ocr-line-sha256-" + "0" * 64 + ".png"))
        run["scratch_path_budget"] = {"expected_longest_path_chars": len(str(expected)),
                                      "limit_exclusive": 260}
        if len(str(expected)) >= 260:
            raise EvidenceError("Scratch root is too long for production OCR paths; use a shorter --work-root under developer tmp")
        if work_root == output or output in work_root.parents:
            raise EvidenceError("Scratch root must be outside the sealed evidence directory")
        work_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="r-", dir=work_root))
        run["scratch_path"] = str(temporary)
        event(output, "scratch_ready", path=str(temporary))
        derived = output / "input.pdf"
        with fitz.open(source) as parent, fitz.open() as single:
            original_page = parent[page["page_index"]]
            geometry = {"width": original_page.rect.width, "height": original_page.rect.height,
                        "rotation": original_page.rotation}
            single.insert_pdf(parent, from_page=page["page_index"], to_page=page["page_index"])
            single.save(derived)
        with fitz.open(derived) as single:
            if {"width": single[0].rect.width, "height": single[0].rect.height,
                "rotation": single[0].rotation} != geometry:
                raise EvidenceError("Derived page geometry changed")
        run["derived_sha256"] = sha256(derived)
        run["page_geometry"] = geometry
        run["timing_seconds"]["derive"] = time.perf_counter() - started
        import bemarkdown
        source_root = Path(bemarkdown.__file__).parent
        run["code_sha256"] = {p.relative_to(source_root).as_posix(): sha256(p)
                              for p in sorted(source_root.rglob("*.py"))}
        run["runtime_module_path"] = str(production_runtime.__file__)
        # Production creates temporary working folders; scope those to this run.
        tempfile.tempdir = str(temporary)
        runtime = production_runtime.create_production_pdf_runtime(
            mcp_root=mcp_root, models_root=mcp_root / "MODELS",
            config_path=mcp_root / "TOOLS/bemarkdown/bemarkdown.toml",
            independent_source_pages=True, layout_runtime_factory=layout_runtime_factory,
        )
        package = output / "package"
        (package / "assets").mkdir(parents=True)
        document_id = ModularPdfPipeline.inspect_source(derived)["document_id"]
        with observe_layout(production_runtime, output) as observations:
            before = time.perf_counter()
            try:
                # Production persists final DocumentIR only with debug export.
                runtime.convert(derived, package, document_id=document_id, debug=True)
            finally:
                run["timing_seconds"]["convert"] = time.perf_counter() - before
        if not observations:
            raise EvidenceError("Production did not call the observed layout entry")
        run["observations"] = observations
        run["timing_seconds"]["sdk_inference"] = sum(o["sdk_inference_seconds"] for o in observations)
        audit = audit_package(package)
        write_json(output / "package-audit.json", audit)
        if any(not a["sha_matches"] or not a["referenced_in_markdown"] for a in audit["assets"]):
            raise EvidenceError("Package asset integrity/reference check failed")
        run["status"] = "SUCCESS"
    except Exception as exc:
        failure = exc
        run["status"] = "FAILED"
        run["error"] = {"type": type(exc).__name__, "message": str(exc)}
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        tempfile.tempdir = previous_tempdir
        if runtime is not None:
            try:
                runtime.close_prepared_resources()
            except Exception as exc:
                run["cleanup_error"] = str(exc)
                (output / "cleanup-traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                run["status"] = "FAILED"
                failure = failure or exc
        if temporary is not None:
            try:
                # Archive long names without recreating long filesystem paths.
                files = sorted(p for p in temporary.rglob("*") if p.is_file())
                if files:
                    archive = output / "scratch.zip"
                    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
                        for path in files:
                            stream.write(path, path.relative_to(temporary).as_posix())
                    run["scratch_evidence"] = {"path": "scratch.zip", "sha256": sha256(archive),
                                               "file_count": len(files), "original_retained": True}
                elif not any(temporary.iterdir()):
                    temporary.rmdir()
            except Exception as exc:
                run["scratch_evidence_error"] = str(exc)
                run["status"] = "FAILED"
                failure = failure or exc
        run["timing_seconds"]["wall"] = time.perf_counter() - started
        event(output, "end", status=run["status"], timing_seconds=run["timing_seconds"])
        write_json(output / "run.json", run)
        seal(output)
    if failure:
        raise EvidenceError(f"Run failed; preserved evidence at {output}: {failure}") from failure
    return run
