"""Source-bound local evidence for unique asset inputs; never a visual gold oracle.

The caller owns the external resource supervisor. Paddle is closed before the
subprocess Ovis stage; that process exits before conditional Xiaomi is started.
No Torch dependency is imported into the Paddle host and no cloud work occurs
while a local model is alive.
"""
from __future__ import annotations

import hashlib
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .pdf.adaptive_ocr import RoutingPolicy, canonical
from .pdf.production_three_model_live import (
    CurrentRunGenerativeStageProvider,
    CurrentRunRecognitionStageProvider,
)
from .pdf.qwen_stage import QwenStageRunner
from .pdf.text_evidence import TextRecognitionRequest

VERSION = "asset-local-evidence-v1"


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def candidate_conflicts(local, proposed_text):
    """Flag disagreement with actual valid candidates, including single glyphs.

    A SUCCESS from a source-reading agent cannot silently erase a contradictory
    local candidate. This flag requests a separate source decision; voting does
    not repair a glyph or authorize replacing a graphic with plain text.
    """
    proposed = canonical(str(proposed_text))
    return [name for name, evidence in local.get("candidates", {}).items()
            if evidence is not None
            and evidence.get("status") in {"OK", "PASS"}
            and str(evidence.get("text", "")).strip()
            and not re.search(r'<\s*img\b|!\[[^\]\n]*\]\s*\(|\[Unresolved image:',
                              str(evidence.get("text", "")), re.IGNORECASE)
            and canonical(str(evidence["text"])) != proposed]


def independent_review_reasons(local):
    reasons = []
    if 'ISOLATED_SYMBOL_SOURCE_REVIEW' in local.get('decision', {}).get('review_reasons', []):
        reasons.append('ISOLATED_SYMBOL_SOURCE_REVIEW')
    coverage = local.get('source_coverage', {})
    if coverage.get('reasons'):
        if coverage.get('input_sha256') != local.get('input_sha256'):
            raise ValueError('ASSET_SOURCE_COVERAGE_BINDING_CHANGED')
        reasons.extend(coverage['reasons'])
    return list(dict.fromkeys(reasons))


def apply_bound_source_review(answer, review_path, receipt_path):
    """Reuse an independent source decision bound to the exact images/receipt.

    Content kind has not been certified by this transcription-only reviewer, so
    retain the image and deliver its visible transcript. This is neither a
    majority decision nor permission to discard a graphic.
    """
    review_path = Path(review_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    review = json.loads(review_path.read_text(encoding='utf-8'))
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    # Accept the independent review worker's flat receipt without rewriting it.
    if 'output' not in receipt and 'output_path' in receipt:
        receipt = {**receipt, 'agent_task_name': receipt['agent_name'],
            'output': {'path': receipt['output_path'], 'sha256': receipt['output_sha256']},
            'packet': {'path': receipt['packet_path'], 'sha256': receipt['packet_sha256']},
            'image_tool_calls': receipt['image_reads']}
    if Path(receipt['output']['path']).resolve() != review_path or _sha(review_path) != receipt['output']['sha256']:
        raise ValueError('ASSET_SOURCE_REVIEW_OUTPUT_CHANGED')
    packet = Path(receipt['packet']['path']).resolve()
    if _sha(packet) != receipt['packet']['sha256']:
        raise ValueError('ASSET_SOURCE_REVIEW_PACKET_CHANGED')
    packet_data = json.loads(packet.read_text(encoding='utf-8'))
    if (packet_data.get('source_sha256') and packet_data['source_sha256'] !=
            (answer.get('repair_task') or {}).get('source_sha256')):
        raise ValueError('ASSET_SOURCE_REVIEW_DOCUMENT_MISMATCH')
    reads = {r['input_sha256'] for r in receipt['image_tool_calls']
             if r.get('actual_image_returned_and_visually_inspected') is True}
    bound = {r['sha256'] for r in review['source_hashes']}
    required = {answer['original_image_sha256'], answer['actual_input_sha256']}
    if not required <= reads or not required <= bound:
        raise ValueError('ASSET_SOURCE_REVIEW_INPUT_MISMATCH')
    for row in review['source_hashes']:
        if _sha(row['path']) != row['sha256']:
            raise ValueError('ASSET_SOURCE_REVIEW_IMAGE_CHANGED')
    if (review['status'] != 'RESOLVED' or review.get('uncertainties')
            or not str(review.get('transcription', '')).strip()):
        raise ValueError('ASSET_SOURCE_REVIEW_UNRESOLVED')
    result = copy.deepcopy(answer)
    result['prior_source_decision'] = {key: result.pop(key) for key in
        ('source_review', 'unresolved_source_review') if key in result}
    result['prior_source_decision']['repair_result'] = copy.deepcopy(result['repair_result'])
    result.update(status='SUCCESS', uncertain=False, markdown=review['transcription'],
        kind='DIAGRAM', preserve_image=True, text_presence='PRESENT',
        visual_content_kind='IMAGE_WITH_TEXT', transcription_status='SUCCESS',
        source_review_required=False, reason=review['visible_strokes'],
        content_kind_verified=False,
        source_review={'review_path': str(review_path), 'review_sha256': _sha(review_path),
            'receipt_path': str(receipt_path), 'receipt_sha256': _sha(receipt_path),
            'agent_task_name': receipt['agent_task_name'],
            'evidence_level': 'INDEPENDENT_SOURCE_REVIEW_REUSED', 'human_verified': False})
    result['repair_result'] = {**result['repair_result'], 'status': 'SUCCESS',
        'text': review['transcription'], 'uncertain': False, 'text_presence': 'PRESENT',
        'visual_content_kind': 'IMAGE_WITH_TEXT', 'source_verified': True,
        'evidence_kind': 'SOURCE_IMAGE_VERIFIED', 'reason': review['visible_strokes']}
    # A Flash receipt attests only its original text, never a subsequent correction.
    for key in ('source_review_receipt', 'source_observation'):
        result['repair_result'].pop(key, None)
    result['repair_result']['independent_source_review'] = copy.deepcopy(result['source_review'])
    return result


class BoundedAssetLocalEvidence:
    def __init__(self, work, *, models_root, config_path, mcp_root,
                 pp_frozen=None, policy=None):
        self.work = Path(work).resolve()
        self.models_root = models_root
        self.config_path = config_path
        self.mcp_root = mcp_root
        self.pp_frozen = pp_frozen
        self.policy = policy or RoutingPolicy(
            pp_fast_path=False, secondary_scope="all_pp_disagreement")

    def _pp(self, inputs):
        if self.pp_frozen is not None:
            frozen = self.pp_frozen
            rows = {r["content_uid"]: r for r in frozen["results"]}
            if set(rows) != {r["content_uid"] for r in inputs}:
                raise ValueError("ASSET_PP_CACHE_POPULATION_MISMATCH")
            for row in inputs:
                if rows[row["content_uid"]]["input_sha256"] != row["sha256"]:
                    raise ValueError("ASSET_PP_CACHE_INPUT_MISMATCH")
            return rows, {"reuse": "DIAGNOSTIC_BOUND_INPUT_CACHE",
                          "fingerprint": frozen["runtime_fingerprint"]}
        folder = self.work / 'pp-stage'
        folder.mkdir(parents=True, exist_ok=True)
        manifest = folder / 'inputs.json'
        output = folder / 'outputs-frozen.json'
        payload = {'inputs': inputs,
            'models_root': str(self.models_root) if self.models_root is not None else None,
            'config_path': str(self.config_path) if self.config_path is not None else None,
            'mcp_root': str(self.mcp_root) if self.mcp_root is not None else None,
            'worker_source_sha256': _sha(__file__)}
        manifest.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
        env = os.environ.copy()
        env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
        with (folder / 'stdout.log').open('wb') as stdout, (folder / 'stderr.log').open('wb') as stderr:
            process = subprocess.run([sys.executable, '-X', 'utf8', '-m',
                'bemarkdown.asset_local_evidence', '--pp-worker', str(manifest), str(output)],
                env=env, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if process.returncode or not output.exists():
            raise RuntimeError(f'ASSET_PP_WORKER_FAILED:exit={process.returncode}:diagnostics={folder}')
        frozen = json.loads(output.read_text(encoding='utf-8'))
        if frozen['worker_pid'] == os.getpid():
            raise RuntimeError('ASSET_PP_PROCESS_ISOLATION_FAILED')
        if set(frozen['results']) != {r['content_uid'] for r in inputs}:
            raise RuntimeError('ASSET_PP_OUTPUT_POPULATION_MISMATCH')
        return frozen['results'], {'reuse': 'NONE', 'fingerprint': frozen['runtime_fingerprint'],
            'worker_pid': frozen['worker_pid'], 'worker_exit_code': process.returncode,
            'process_exit_unload_before_ovis': True, 'models_closed_before_ovis': True,
            'prediction_artifact_sha256': _sha(output)}

    def _pp_in_child(self, inputs):
        from .pdf_content_router import PaddleXPdfOcrRuntime
        runtime = PaddleXPdfOcrRuntime(models_root=self.models_root,
            config_path=self.config_path, mcp_root=self.mcp_root)
        rows = {}
        try:
            for item in inputs:
                crop = dict(path=item["path"], page_index=0,
                    bbox_pdf_pt=[0, 0, item["width"], item["height"]],
                    scale_x=1.0, scale_y=1.0, complete_region_lines=True)
                lines = runtime.recognize(crop)
                rows[item["content_uid"]] = {
                    "content_uid": item["content_uid"], "lines": lines,
                    "input_sha256": item["sha256"],
                    "text": "\n".join(str(r.get("text", "")) for r in lines)}
            fingerprint = runtime.fingerprint()
        finally:
            runtime.close()
        return rows, {"reuse": "NONE", "fingerprint": fingerprint,
                      "models_closed_before_ovis": True}

    def __call__(self, inputs):
        inputs = list(inputs)
        if len({r["content_uid"] for r in inputs}) != len(inputs):
            raise ValueError("ASSET_LOCAL_INPUTS_MUST_BE_UNIQUE")
        self.work.mkdir(parents=True, exist_ok=True)
        for row in inputs:
            Path(row["path"]).resolve().relative_to(self.work.parent)
            if _sha(row["path"]) != row["sha256"]:
                raise ValueError("ASSET_LOCAL_INPUT_CHANGED")
        started = time.monotonic()
        pp, pp_meta = self._pp(inputs)
        requests = [TextRecognitionRequest(document_id=r["source_sha256"][:16],
            page_id="asset-content", region_id=r["content_uid"],
            source_candidate_id=r["content_uid"],
            bbox=(0, 0, r["width"], r["height"]), crop_ref=r["path"],
            crop_sha256=r["sha256"], route_kind="OCR_TEXT_REGION",
            text_track="TEXT_REGION_CROP") for r in inputs]
        common = dict(crop_root=self.work.parent, output_root=self.work / "stages",
            models_root=self.models_root, config_path=self.config_path,
            mcp_root=self.mcp_root, vram_limit_mib=2816, max_new_tokens=2048)
        ovis_runner = QwenStageRunner.from_registry(model_id="ovis-ocr2", **common)
        ovis = CurrentRunRecognitionStageProvider("B_OVIS_OCR2", ovis_runner)
        primary = ovis.recognize_batch(requests) if requests else []
        pp_candidates = {key: {"text": row["text"], "status": "OK",
            "confidence": min((float(x.get("rec_confidence", x.get("confidence", 0))
                               or 0) for x in row["lines"]), default=None)}
            for key, row in pp.items()}
        b_candidates = {r.region_id: {"text": e.text,
            "status": e.output_contract_status, "confidence": e.confidence}
            for r, e in zip(requests, primary, strict=True)}
        triggers = {r.region_id: self.policy.trigger(pp_candidates[r.region_id],
            b_candidates[r.region_id], item)
            for r, item in zip(requests, inputs, strict=True)}
        pending = [r for r in requests if triggers[r.region_id]]
        xiaomi_runner = None
        secondary = {}
        if pending:
            xiaomi_runner = QwenStageRunner.from_registry(model_id="xiaomi-ocr-0", **common)
            xiaomi = CurrentRunGenerativeStageProvider("C_XIAOMI_OCR_0", xiaomi_runner)
            secondary = {r.region_id: e for r, e in zip(pending,
                xiaomi.recognize_batch(pending), strict=True)}
        result = {}
        for request, b in zip(requests, primary, strict=True):
            uid = request.region_id
            c = secondary.get(uid)
            cc = {"text": c.text, "status": c.output_contract_status,
                  "confidence": c.confidence} if c is not None else None
            candidates = {"PP": pp_candidates[uid], "Ovis": b_candidates[uid],
                          "Xiaomi": cc}
            decision = self.policy.select(candidates["PP"], candidates["Ovis"], cc)
            from .source_routing_risk import source_coverage_risks
            coverage = source_coverage_risks({'path': request.crop_ref, 'sha256': request.crop_sha256},
                {'input_sha256': request.crop_sha256, 'pp_lines': pp[uid]['lines']})
            if coverage['reasons']:
                decision['review_required'] = True
                decision['review_reasons'] = list(dict.fromkeys([*decision['review_reasons'], *coverage['reasons']]))
            conflicts = candidate_conflicts({"candidates": candidates},
                                            decision["selected_text"])
            if conflicts:
                decision["review_required"] = True
                decision["review_reasons"] = list(dict.fromkeys([
                    *decision["review_reasons"], "VISIBLE_ASSET_CANDIDATE_CONFLICT_REVIEW"]))
            result[uid] = {"content_uid": uid, "input_sha256": request.crop_sha256,
                "candidates": candidates, "decision": decision,
                "secondary_trigger_reasons": list(triggers[uid]),
                "pp_lines": pp[uid]["lines"], "ovis_evidence": b.to_dict(),
                "xiaomi_evidence": c.to_dict() if c else None,
                "content_kind_verified": False, "automatic_release": False,
                "source_coverage": coverage}
        artifact = {"version": VERSION, "results": result, "pp": pp_meta,
            "policy_fingerprint": self.policy.fingerprint(),
            "ovis_stages": ovis_runner.stage_metrics,
            "xiaomi_stages": xiaomi_runner.stage_metrics if xiaomi_runner else [],
            "model_processes_exited_before_return": True,
            "elapsed_seconds": time.monotonic() - started,
            "gold_read": False, "source_sha256": _sha(__file__)}
        (self.work / "predictions-frozen.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result


def _pp_worker(manifest, output):
    manifest = Path(manifest).resolve()
    request = json.loads(manifest.read_text(encoding='utf-8'))
    if _sha(__file__) != request['worker_source_sha256']:
        raise RuntimeError('ASSET_PP_EXECUTED_SOURCE_CHANGED')
    for row in request['inputs']:
        if _sha(row['path']) != row['sha256']:
            raise RuntimeError('ASSET_PP_INPUT_CHANGED')
    runner = BoundedAssetLocalEvidence(manifest.parent, models_root=request['models_root'],
        config_path=request['config_path'], mcp_root=request['mcp_root'])
    rows, metadata = runner._pp_in_child(request['inputs'])
    result = {'results': rows, 'runtime_fingerprint': metadata['fingerprint'],
        'worker_pid': os.getpid(), 'worker_source_sha256': _sha(__file__),
        'models_closed': True, 'gold_read': False}
    Path(output).write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--pp-worker':
        raise SystemExit('Expected --pp-worker INPUT_MANIFEST OUTPUT')
    _pp_worker(sys.argv[2], sys.argv[3])
