from __future__ import annotations

import io
import hashlib

import fitz
import pytest
from PIL import Image

from bemarkdown.pdf.graphics_render import (
    native_prose_in_fallback,
    protect_semantic_images_in_background_crops,
    render_source_graphics,
)
from bemarkdown.pdf_content_router import (
    PdfContentAdapterExecutor,
    PdfCropper,
    plan_page_content,
)
from bemarkdown.pdf_document_ir import assemble_document_ir


def _graphics_page(document, text=False):
    page = document.new_page(width=160, height=180)
    page.draw_rect(page.rect, color=None, fill=(0.7, 0.8, 0.9))
    stream = io.BytesIO()
    Image.new("RGBA", (20, 20), (220, 40, 30, 120)).save(stream, format="PNG")
    page.insert_image(fitz.Rect(15, 20, 130, 140), stream=stream.getvalue())
    page.draw_circle((65, 65), 20, color=(0, 0, 1), fill_opacity=0.4, fill=(0, 1, 0))
    if text:
        page.insert_text((35, 70), "BODY", fontsize=16)
    return page


def test_text_filter_preserves_source_color_alpha_and_vector_graphics():
    with fitz.open() as expected_document, fitz.open() as document:
        expected_page = _graphics_page(expected_document)
        page = _graphics_page(document, text=True)
        clip = fitz.Rect(10, 12, 145, 165)
        expected = expected_page.get_pixmap(dpi=144, clip=clip)
        actual, audit = render_source_graphics(page, clip, 144)
        assert actual.samples == expected.samples
        assert actual.width == expected.width and actual.height == expected.height
        assert audit["omitted_text_operations"] > 0


def test_text_used_in_source_soft_mask_is_kept():
    with fitz.open() as document:
        page = document.new_page(width=120, height=120)
        font = page.insert_font(fontname="helv")
        form = document.get_new_xref()
        document.update_object(
            form,
            f"<< /Type /XObject /Subtype /Form /BBox [0 0 120 120] /Group << /S /Transparency /CS /DeviceGray >> /Resources << /Font << /F1 {font} 0 R >> >> >>",
        )
        document.update_stream(form, b"1 g BT /F1 30 Tf 10 40 Td (MASK) Tj ET")
        state = document.get_new_xref()
        document.update_object(
            state, f"<< /Type /ExtGState /SMask << /S /Luminosity /G {form} 0 R >> >>"
        )
        resources = int(document.xref_get_key(page.xref, "Resources")[1].split()[0])
        document.xref_set_key(resources, "ExtGState", f"<< /GS {state} 0 R >>")
        contents = document.get_new_xref()
        document.update_object(contents, "<<>>")
        document.update_stream(contents, b"q /GS gs 1 0 0 rg 0 0 120 120 re f Q")
        page.set_contents(contents)
        expected = page.get_pixmap(dpi=144)
        actual, _ = render_source_graphics(page, page.rect, 144)
        assert actual.samples == expected.samples
        assert min(expected.samples) == 0


def test_partial_background_crop_does_not_repeat_owned_figure_pixels():
    """A native fallback includes a photograph and unique adjacent decoration."""
    routes = [
        {
            "route_id": "figure",
            "output_kind": "IMAGE",
            "adapter": "IMAGE_RENDER_CROP",
            "provenance": {
                "candidate_kinds": ["MODEL_CANONICAL_REGION"],
                "bbox_pdf_pt": [20, 20, 80, 80],
            },
        },
        {
            "route_id": "fallback",
            "output_kind": "IMAGE",
            "adapter": "IMAGE_RENDER_CROP",
            "provenance": {
                "candidate_kinds": ["NATIVE_IMAGE_FALLBACK"],
                "bbox_pdf_pt": [30, 25, 140, 85],
            },
        },
    ]
    protect_semantic_images_in_background_crops(routes)
    exclusions = routes[1]["provenance"].get("source_graphics_exclude_regions", [])
    assert exclusions, "Overlapping native fallback currently duplicates the figure"
    with fitz.open() as document:
        page = document.new_page(width=160, height=100)
        page.draw_rect(fitz.Rect(20, 20, 80, 80), color=None, fill=(1, 0, 0))
        page.insert_text((95, 60), "LABEL", fontsize=8)
        page.draw_line((135, 25), (135, 80), color=(0, 0, 1))
        actual, audit = render_source_graphics(
            page,
            [30, 25, 140, 85],
            144,
            omit_native_prose=False,
            exclude_regions=exclusions,
        )
        assert actual.pixel(20, 20) == (255, 255, 255)
        original = page.get_pixmap(dpi=144, clip=fitz.Rect(30, 25, 140, 85))
        # Adjacent native labels and source decoration stay byte-for-byte intact.
        for y in range(actual.height):
            for x in range(110, actual.width):
                assert actual.pixel(x, y) == original.pixel(x, y)
        assert audit["semantic_image_excluded_regions_pdf_pt"]
    assert "source_graphics_exclude_regions" not in routes[0]["provenance"]


@pytest.mark.parametrize('kind,adapter', [('FORMULA', 'FORMULA_RECOGNITION'), ('TABLE', 'TABLE_ENGINE')])
def test_specialist_content_is_not_duplicated_in_native_image_fallback(kind, adapter):
    owner = {'route_id': 'owner', 'output_kind': kind, 'adapter': adapter,
             'provenance': {'candidate_kinds': ['MODEL_CANONICAL_REGION'], 'bbox_pdf_pt': [20, 20, 80, 80]}}
    fallback = {'route_id': 'fallback', 'output_kind': 'IMAGE', 'adapter': 'IMAGE_NATIVE_EXTRACT',
                'provenance': {'candidate_kinds': ['NATIVE_IMAGE_FALLBACK'], 'bbox_pdf_pt': [0, 0, 100, 100]}}
    protect_semantic_images_in_background_crops([owner, fallback])
    assert fallback['adapter'] == 'IMAGE_RENDER_CROP'
    exclusions = fallback['provenance']['source_graphics_exclude_regions']
    with fitz.open() as document:
        page = document.new_page(width=100, height=100)
        page.draw_rect(fitz.Rect(25, 25, 75, 75), color=None, fill=(0, 0, 0))
        page.draw_line((90, 10), (90, 90), color=(1, 0, 0))
        actual, _ = render_source_graphics(page, page.rect, 72,
            omit_native_prose=False, exclude_regions=exclusions)
        assert actual.pixel(50, 50) == (255, 255, 255)
        original = page.get_pixmap(dpi=72)
        assert actual.pixel(90, 50) == original.pixel(90, 50)
    assert 'source_graphics_exclude_regions' not in owner['provenance']


def test_exclusion_at_crop_bottom_is_bounded_with_nonzero_pixmap_origin():
    with fitz.open() as document:
        page = document.new_page(width=500, height=700)
        page.draw_rect(page.rect, color=None, fill=(1, 0, 0))
        clip = [470, 627, 484, 641]
        actual, _ = render_source_graphics(
            page,
            clip,
            144,
            omit_native_prose=False,
            exclude_regions=[[470, 627, 480, 641]],
        )
        assert actual.irect == fitz.IRect(940, 1254, 968, 1282)
        assert actual.pixel(0, actual.height - 1) == (255, 255, 255)
        assert actual.pixel(actual.width - 1, actual.height - 1) == (255, 0, 0)


@pytest.mark.parametrize('outside_mark', [False, True])
def test_empty_graphics_fallback_is_suppressed_but_unowned_pixels_are_kept(tmp_path, outside_mark):
    path = tmp_path/'source.pdf'
    with fitz.open() as document:
        page = document.new_page(width=100, height=100)
        page.draw_rect(fitz.Rect(30, 30, 70, 70), color=None, fill=(0, 0, 0))
        if outside_mark:
            page.draw_rect(fitz.Rect(90, 90, 95, 95), color=None, fill=(.99, .99, .99))
        document.save(path)
    record = {'document_id': 'source', 'page_index': 0, 'source_path': str(path),
              'source_profile': 'IMAGE_ONLY', 'native_text_trust': 'NONE',
              'geometry': {'width_pt': 100, 'height_pt': 100, 'rotation': 0}}
    source = {'native_images': [], 'native_text': []}
    plan = plan_page_content({'document_id': 'source', 'page_index': 0,
        'page_escalation': {'status': 'NONE'}, 'fusion_candidates': [
            {'candidate_id': 'formula', 'candidate_kind': 'MODEL_CANONICAL_REGION',
             'semantic_hint': 'FORMULA', 'bbox_pdf_pt': [20, 20, 80, 80]},
            {'candidate_id': 'fallback', 'candidate_kind': 'NATIVE_IMAGE_FALLBACK',
             'semantic_hint': 'IMAGE_LIKE', 'bbox_pdf_pt': [0, 0, 100, 100]},
        ]}, record, source)
    route = next(r for r in plan['routes'] if r['output_kind'] == 'IMAGE')
    executor = PdfContentAdapterExecutor(tmp_path/'out')
    row = executor._image_crop(route, executor.cropper.render(route))
    result = assemble_document_ir(document_id='source',
        source={'type': 'PDF', 'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size},
        page_records=[record], region_content=[row], source_evidence={('source',0): source})
    if outside_mark:
        assert len(result['blocks']) == 1
        assert row['status'] == 'REVIEW_REQUIRED'
        assert row['quality_status'] == 'GRAPHICS_RESIDUAL_REVIEW_REQUIRED'
        assert row['provenance']['source_graphics_ownership']['excluded_owners']
        assert result['blocks'][0]['review_state'] == 'REVIEW_REQUIRED'
    else:
        assert result['blocks'] == []
        assert result['suppressed_blocks'][0]['provenance']['quality_status'] == 'MASKED_EMPTY_GRAPHICS_ROUTE'


def test_direct_native_owner_excludes_only_its_verified_placement():
    figure = {
        "output_kind": "IMAGE",
        "adapter": "IMAGE_NATIVE_EXTRACT",
        "provenance": {
            "candidate_kinds": ["MODEL_CANONICAL_REGION"],
            "bbox_pdf_pt": [10, 10, 90, 90],
            "native_image_placements": [
                {
                    "bbox_pdf_pt": [20, 20, 80, 80],
                    "native_appearance": {"verified": True},
                }
            ],
        },
    }
    background = {
        "output_kind": "IMAGE",
        "adapter": "IMAGE_RENDER_CROP",
        "provenance": {
            "candidate_kinds": ["NATIVE_IMAGE_FALLBACK"],
            "bbox_pdf_pt": [0, 0, 100, 100],
        },
    }
    protect_semantic_images_in_background_crops([figure, background])
    assert background["provenance"]["source_graphics_exclude_regions"] == [
        [20, 20, 80, 80]
    ]


def test_semantic_figures_and_untrusted_native_text_keep_original_crop():
    source = {"native_text": [{"text": "LABEL", "bbox_pdf_pt": [20, 20, 40, 40]}]}
    candidate = {
        "candidate_kind": "NATIVE_IMAGE_FALLBACK",
        "bbox_pdf_pt": [10, 10, 90, 90],
    }
    assert native_prose_in_fallback(candidate, source, {"native_text_trust": "HIGH"})
    assert native_prose_in_fallback(
        {"candidate_kind": "NATIVE_IMAGE_FALLBACK", "bbox_pdf_pt": [35, 10, 90, 90]},
        source,
        {"native_text_trust": "HIGH"},
    )
    assert not native_prose_in_fallback(
        candidate, source, {"native_text_trust": "MEDIUM"}
    )
    assert not native_prose_in_fallback(
        {**candidate, "candidate_kind": "MODEL_CANONICAL_REGION"},
        source,
        {"native_text_trust": "HIGH"},
    )


def test_protected_figure_keeps_native_label_while_external_prose_is_omitted():
    with fitz.open() as document, fitz.open() as expected_document:
        page = _graphics_page(document, text=True)
        page.insert_text((10, 170), "EXTERNAL BODY")
        expected = _graphics_page(expected_document, text=True).get_pixmap(dpi=144)
        actual, audit = render_source_graphics(
            page, page.rect, 144, preserve_regions=[[10, 15, 135, 145]]
        )
        assert actual.samples == expected.samples
        assert audit["native_text_preserved_regions_pdf_pt"] == [
            [10.0, 15.0, 135.0, 145.0]
        ]


def test_fallback_route_keeps_prose_and_source_graphics_as_separate_outputs(tmp_path):
    path = tmp_path / "source.pdf"
    with fitz.open() as document:
        _graphics_page(document, text=True)
        document.save(path)
    evidence = {
        "native_text": [
            {"evidence_id": "line-1", "text": "BODY", "bbox_pdf_pt": [35, 52, 81, 75]}
        ],
        "native_images": [],
    }
    image = {
        "candidate_id": "fallback",
        "candidate_kind": "NATIVE_IMAGE_FALLBACK",
        "semantic_hint": "IMAGE_LIKE",
        "bbox_pdf_pt": [10, 10, 145, 165],
        "evidence_ids": [],
    }
    prose = {
        "candidate_id": "prose",
        "candidate_kind": "NATIVE_TEXT_FALLBACK",
        "semantic_hint": "TEXT",
        "bbox_pdf_pt": [35, 52, 81, 75],
        "evidence_ids": ["line-1"],
    }
    record = {
        "document_id": "source-test",
        "page_index": 0,
        "source_path": str(path),
        "native_text_trust": "HIGH",
        "source_profile": "MIXED_NATIVE_VISUAL",
        "routing_decision": "NATIVE_FIRST",
    }
    fusion = {
        "document_id": "source-test",
        "page_index": 0,
        "page_geometry": {"width_pt": 160, "height_pt": 180},
        "fusion_candidates": [image, prose],
        "page_escalation": {"status": "NONE"},
    }
    plan = plan_page_content(fusion, record, evidence)
    executor = PdfContentAdapterExecutor(
        tmp_path / "out", cropper=PdfCropper(tmp_path / "crops")
    )
    rows = executor.execute(plan, evidence)["region_content"]
    text = next(row for row in rows if row.get("text") == "BODY")
    graphics = next(row for row in rows if row.get("binary_artifact_ref"))
    assert text["status"] == graphics["status"] == "SUCCESS"
    assert (
        graphics["provenance"]["render_crop"]["source_graphics_render"]["method"]
        == "MUPDF_SOURCE_GRAPHICS_WITHOUT_NATIVE_PROSE"
    )


def test_vector_review_remainder_does_not_repeat_owned_prose_or_figure(tmp_path):
    path = tmp_path / "vector-review.pdf"
    with fitz.open() as document:
        page = document.new_page(width=200, height=150)
        page.insert_text((20, 40), "BODY", fontsize=12)
        page.draw_rect(fitz.Rect(120, 25, 160, 70), color=None, fill=(1, 0, 0))
        page.draw_circle((40, 110), 12, color=(0, 0, 1), fill=(0, 0, 1))
        document.save(path)
    evidence = {"native_text": [{"evidence_id": "line-1", "text": "BODY",
                                "bbox_pdf_pt": [20, 27, 60, 44]}], "native_images": []}
    candidates = [
        {"candidate_id": "vector", "candidate_kind": "VECTOR_VISUAL_FALLBACK",
         "semantic_hint": "VISUAL_UNKNOWN", "bbox_pdf_pt": [0, 0, 200, 150], "evidence_ids": []},
        {"candidate_id": "figure", "candidate_kind": "MODEL_CANONICAL_REGION",
         "semantic_hint": "IMAGE", "bbox_pdf_pt": [120, 25, 160, 70], "evidence_ids": []},
        {"candidate_id": "prose", "candidate_kind": "NATIVE_TEXT_FALLBACK",
         "semantic_hint": "TEXT", "bbox_pdf_pt": [20, 27, 60, 44], "evidence_ids": ["line-1"]},
    ]
    record = {"document_id": "vector-test", "page_index": 0, "source_path": str(path),
              "native_text_trust": "HIGH", "source_profile": "MIXED_NATIVE_VISUAL",
              "routing_decision": "NATIVE_FIRST"}
    fusion = {"document_id": "vector-test", "page_index": 0,
              "page_geometry": {"width_pt": 200, "height_pt": 150},
              "fusion_candidates": candidates, "page_escalation": {"status": "NONE"}}
    plan = plan_page_content(fusion, record, evidence)
    executor = PdfContentAdapterExecutor(tmp_path / "out", cropper=PdfCropper(tmp_path / "crops"))
    rows = executor.execute(plan, evidence)["region_content"]
    from bemarkdown.production_gpu_closeout import materialize_missing_asset_refs

    rows, _ = materialize_missing_asset_refs(
        rows, routes_by_id={route["route_id"]: route for route in plan["routes"]},
        render_crop=lambda route: executor.cropper.render(route, full_page=False),
    )
    review = next(row for row in rows if row["semantic_hint"] == "VISUAL_UNKNOWN")
    assert review["status"] == "REVIEW_REQUIRED"
    assert review.get("binary_artifact_ref"), "Vector remainder needs its own bounded graphics asset"
    with Image.open(review["binary_artifact_ref"]) as actual:
        # Both separately delivered text and figure must be absent from the remainder.
        sx, sy = actual.width / 200, actual.height / 150
        for box in ([15, 20, 65, 48], [125, 30, 155, 65]):
            region = actual.crop(tuple(round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(box)))
            assert region.getextrema() == ((255, 255),) * 3
        assert actual.getpixel((round(40*sx), round(110*sy))) == (0, 0, 255)
    assert next(row for row in rows if row.get("text") == "BODY")["status"] == "SUCCESS"
    figure = next(row for row in rows if row["semantic_hint"] == "IMAGE")
    with Image.open(figure["binary_artifact_ref"]) as original:
        assert original.getpixel((original.width // 2, original.height // 2)) == (255, 0, 0)
