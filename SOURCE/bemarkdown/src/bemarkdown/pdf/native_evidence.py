"""Source-owned text and image evidence extracted from PDF objects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .object_profiler import PDFObjectProfile

NATIVE_EVIDENCE_SCHEMA = "bemarkdown-native-evidence-ir-v1"


def _stable_id(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"native-evidence-{hashlib.sha256(payload).hexdigest()[:24]}"


def _bbox(value: Iterable[float]) -> tuple[float, float, float, float]:
    result = tuple(round(float(item), 6) for item in value)
    if len(result) != 4 or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"INVALID_NATIVE_EVIDENCE_BBOX:{result!r}")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class NativeEvidenceIR:
    schema: str
    document_id: str
    page_id: str
    page_index: int
    native_evidence_id: str
    evidence_kind: str
    bbox_pdf_pt: tuple[float, float, float, float]
    text: str | None
    font_metadata: dict[str, Any]
    source_object_refs: tuple[str, ...]
    extraction_order: int
    reliability_state: str
    reliability_reasons: tuple[str, ...]
    source_pdf_sha256: str
    native_image_object_id: str | None = None
    asset_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bbox_pdf_pt"] = list(self.bbox_pdf_pt)
        result["source_object_refs"] = list(self.source_object_refs)
        result["reliability_reasons"] = list(self.reliability_reasons)
        return result


class NativeEvidenceExtractor:
    """Extract page-native evidence without rendering or learned inference."""

    def extract(
        self,
        source_pdf: Path | str,
        *,
        profiles: Iterable[PDFObjectProfile],
    ) -> list[NativeEvidenceIR]:
        import fitz

        by_index = {profile.page_index: profile for profile in profiles}
        document = fitz.open(Path(source_pdf))
        rows: list[NativeEvidenceIR] = []
        try:
            for page_index, page in enumerate(document):
                profile = by_index[page_index]
                order = 0
                raw = page.get_text("dict", sort=False)
                for block_index, block in enumerate(raw.get("blocks", [])):
                    if block.get("type") != 0:
                        continue
                    for line_index, line in enumerate(block.get("lines", [])):
                        for span_index, span in enumerate(line.get("spans", [])):
                            text = str(span.get("text", ""))
                            if not text.strip():
                                continue
                            bbox = _bbox(span["bbox"])
                            refs = (
                                f"pdf-text-block:{block.get('number', block_index)}",
                                f"line:{line_index}",
                                f"span:{span_index}",
                            )
                            identity = {
                                "document_id": profile.document_id,
                                "page_index": page_index,
                                "kind": "TEXT",
                                "bbox": bbox,
                                "text": text,
                                "refs": refs,
                            }
                            rows.append(
                                NativeEvidenceIR(
                                    schema=NATIVE_EVIDENCE_SCHEMA,
                                    document_id=profile.document_id,
                                    page_id=profile.page_id,
                                    page_index=page_index,
                                    native_evidence_id=_stable_id(identity),
                                    evidence_kind="TEXT",
                                    bbox_pdf_pt=bbox,
                                    text=text,
                                    font_metadata={
                                        "font": span.get("font"),
                                        "size": span.get("size"),
                                        "flags": span.get("flags"),
                                        "color": span.get("color"),
                                    },
                                    source_object_refs=refs,
                                    extraction_order=order,
                                    reliability_state=profile.profile_state.value,
                                    reliability_reasons=profile.reliability_reasons,
                                    source_pdf_sha256=profile.source_pdf_sha256,
                                )
                            )
                            order += 1

                for image_index, image in enumerate(page.get_images(full=True)):
                    xref = int(image[0])
                    extracted = document.extract_image(xref)
                    asset_sha = hashlib.sha256(extracted["image"]).hexdigest()
                    for occurrence, rect in enumerate(page.get_image_rects(xref)):
                        bbox = _bbox((rect.x0, rect.y0, rect.x1, rect.y1))
                        object_id = f"pdf-image-xref:{xref}:occurrence:{occurrence}"
                        identity = {
                            "document_id": profile.document_id,
                            "page_index": page_index,
                            "kind": "IMAGE",
                            "bbox": bbox,
                            "xref": xref,
                            "asset_sha256": asset_sha,
                        }
                        rows.append(
                            NativeEvidenceIR(
                                schema=NATIVE_EVIDENCE_SCHEMA,
                                document_id=profile.document_id,
                                page_id=profile.page_id,
                                page_index=page_index,
                                native_evidence_id=_stable_id(identity),
                                evidence_kind="IMAGE",
                                bbox_pdf_pt=bbox,
                                text=None,
                                font_metadata={},
                                source_object_refs=(object_id,),
                                extraction_order=order + image_index,
                                reliability_state="SOURCE_ASSET_AUTHORITY",
                                reliability_reasons=("NATIVE_IMAGE_OBJECT_AVAILABLE",),
                                source_pdf_sha256=profile.source_pdf_sha256,
                                native_image_object_id=object_id,
                                asset_sha256=asset_sha,
                            )
                        )
        finally:
            document.close()
        return rows
