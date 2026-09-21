import fitz
import pytest

from bemarkdown.pdf.prose_edge_crop import annotate_recovered_prose
from bemarkdown.pdf_content_router import PdfContentAdapterExecutor, plan_page_content


@pytest.mark.parametrize("extra", [None, "vector", "label"])
def test_trim_external_paragraph_but_preserve_source_graphics_and_unowned_labels(tmp_path, extra):
    path = tmp_path / "source.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=100, height=110)
        page.insert_text((5, 12), "paragraph beginning", fontsize=7)
        page.insert_text((5, 26), "previous paragraph tail", fontsize=7)
        page.draw_rect(fitz.Rect(35, 45, 80, 90), fill=(0.4, 0.5, 0.6))
        if extra == "vector":
            page.draw_line(fitz.Point(35, 22), fitz.Point(70, 22), color=(0, 0, 0), width=0.1)
        if extra == "label":
            page.insert_text((80, 19), "X", fontsize=5)
        lines = [line for block in page.get_text("dict")["blocks"] for line in block.get("lines", [])]
        native = [{"evidence_id": f"line-{i}", "text": "".join(s["text"] for s in line["spans"]),
                   "bbox_pdf_pt": list(line["bbox"])} for i, line in enumerate(lines)]
        doc.save(path)
    source = {"native_images": [], "native_text": native}
    record = {"document_id": "source", "page_index": 0, "source_path": str(path),
              "source_profile": "MIXED_NATIVE_VISUAL", "native_text_trust": "HIGH",
              "geometry": {"width_pt": 100, "height_pt": 110, "rotation": 0}}
    plan = plan_page_content({"document_id": "source", "page_index": 0,
        "page_escalation": {"status": "NONE"}, "fusion_candidates": [
        {"candidate_id": "figure", "candidate_kind": "MODEL_CANONICAL_REGION",
         "semantic_hint": "IMAGE", "bbox_pdf_pt": [25, 18, 90, 95]}]}, record, source)
    figure = next(r for r in plan["routes"] if r["output_kind"] == "IMAGE")
    paragraph = {"route_id": "paragraph", "adapter": "NATIVE_TEXT_BRIDGE", "semantic_evidence": ["TEXT"],
                 "provenance": {"candidate_kinds": ["MODEL_CANONICAL_REGION"],
                                "native_text_evidence_ids": ["line-0", "line-1"]}}
    annotate_recovered_prose([figure, paragraph], source, record)
    executor = PdfContentAdapterExecutor(tmp_path / "content")
    crop = executor.cropper.render(figure)
    row = executor._image_crop(figure, crop)
    if extra is None:
        assert row["bbox_pdf_pt"][1] > 28
        assert row["provenance"]["prose_edge_refinement"]["status"] == "TRIMMED_TOP_PROSE"
        assert row["provenance"]["render_crop"]["bbox_pdf_pt"][1] > native[1]["bbox_pdf_pt"][3]
    else:
        assert row["bbox_pdf_pt"] == [25, 18, 90, 95]
        assert row["provenance"]["prose_edge_refinement"]["status"] == "UNCHANGED"
        assert row["provenance"]["prose_edge_refinement"]["reason"] == (
            "SOURCE_GRAPHICS_IN_STRIP" if extra == "vector" else "UNCLAIMED_NATIVE_TEXT_IN_STRIP")
