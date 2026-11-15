"""Developer-only Transformers adapters for the visual serial route.

The legacy BeMarkdown runtime is intentionally unchanged by this module.  The
adapters are lazy: importing them does not import Torch, and each model is
released before the next stage starts.  They are suitable for a developer
candidate run while the V3/OvisOCR3 payloads remain outside formal MCP.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..page_render import PageRender as LayoutPageRender
from .visual_serial_pipeline import (
    VisualRegion,
    VisualSerialOrchestrator,
    VisualSerialPolicy,
)
from .visual_serial_prompts import (
    prompt_fingerprint,
    prompt_for,
    prompt_for_request,
    prompt_source_sha256,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower()) or "region"


def _developer_registry_value(registry_path: str | Path, model_id: str, key: str) -> str:
    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    for row in payload.get("models", []):
        if row.get("id") == model_id and row.get(key):
            return str(row[key])
    raise RuntimeError(f"VISUAL_SERIAL_DEVELOPER_MODEL_NOT_REGISTERED:{model_id}:{key}")


class TransformersV3LayoutRuntime:
    """PP-DocLayoutV3 detector with page-image-only input."""

    def __init__(self, model_root: str | Path, *, threshold: float = 0.35) -> None:
        self.model_root = Path(model_root).resolve()
        self.threshold = float(threshold)
        self.processor = None
        self.model = None
        self.torch = None
        self.identity: dict[str, Any] = {
            "model_id": "pp-doclayout-v3",
            "model_root": str(self.model_root),
            "threshold": self.threshold,
            "device": "cuda:0",
        }
        self.metrics: dict[str, Any] = {}

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        if not torch.cuda.is_available():
            raise RuntimeError("VISUAL_SERIAL_GPU_REQUIRED")
        started = time.perf_counter()
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_root, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_root, local_files_only=True, trust_remote_code=False
        ).to("cuda:0").eval()
        self.metrics["load_seconds"] = time.perf_counter() - started
        self.identity.update(
            {
                "model_fingerprint": _sha256(self.model_root / "model.safetensors"),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
            }
        )

    def detect(
        self,
        renders: Sequence[LayoutPageRender],
        *,
        output_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        if output_dir is None:
            raise RuntimeError("VISUAL_SERIAL_LAYOUT_CROP_ROOT_REQUIRED")
        from io import BytesIO
        from PIL import Image

        self._load()
        assert self.processor is not None and self.model is not None and self.torch is not None
        crop_root = Path(output_dir).resolve() / "_bemarkdown" / "visual_serial" / "v3_crops"
        crop_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        peak_before = self.torch.cuda.max_memory_allocated()
        for render in renders:
            with Image.open(BytesIO(render.image_bytes)) as source:
                image = source.convert("RGB")
                image.load()
            inputs = self.processor(images=image, return_tensors="pt").to("cuda:0")
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            self.torch.cuda.synchronize()
            decoded = self.processor.post_process_object_detection(
                outputs,
                threshold=self.threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            detections = []
            for index, (score, label, box) in enumerate(
                zip(decoded["scores"], decoded["labels"], decoded["boxes"], strict=True)
            ):
                class_id = int(label)
                label_name = str(
                    self.model.config.id2label.get(
                        class_id,
                        self.model.config.id2label.get(str(class_id), "unknown"),
                    )
                )
                x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append(
                    {
                        "score": float(score),
                        "class_id": class_id,
                        "label": label_name,
                        "box": (x1, y1, x2, y2),
                    }
                )
            # PPDocLayoutV3's processor already sorts boxes by the learned
            # order_seq. Sorting again by y/x interleaves multi-column prose.
            if "order_seq" not in decoded or len(decoded["order_seq"]) != len(decoded["scores"]):
                raise RuntimeError("V3_READING_ORDER_MISSING")
            for order, detection in enumerate(detections):
                x1, y1, x2, y2 = detection["box"]
                crop = image.crop((x1, y1, x2, y2))
                crop_path = crop_root / (
                    f"page-{render.page_number:04d}-{order:04d}-{_safe_label(detection['label'])}.png"
                )
                crop.save(crop_path, format="PNG")
                crop.close()
                rows.append(
                    {
                        "region_id": f"p{render.page_index:04d}-r{order:04d}",
                        "label": detection["label"],
                        "bbox": [x1, y1, x2, y2],
                        "crop": {
                            "path": str(crop_path),
                            "sha256": _sha256(crop_path),
                        },
                        "page_index": int(render.page_index),
                        "reading_order": order,
                        "metadata": {
                            "score": detection["score"],
                            "class_id": detection["class_id"],
                            "render_sha256": render.render_sha256,
                        },
                    }
                )
            image.close()
            del inputs, outputs, decoded
        self.metrics.update(
            {
                "inference_seconds": time.perf_counter() - started,
                "region_count": len(rows),
                "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
                "peak_delta_allocated_bytes": self.torch.cuda.max_memory_allocated() - peak_before,
            }
        )
        return rows

    def close(self) -> None:
        if self.torch is not None:
            self.torch.cuda.synchronize()
        self.model = None
        self.processor = None
        if self.torch is not None:
            self.torch.cuda.empty_cache()
        gc.collect()


class TransformersVisionOcrRuntime:
    """Lazy Xiaomi-OCR-0 or OvisOCR3 recognizer."""

    def __init__(self, model_id: str, model_root: str | Path, *, max_new_tokens: int = 2048) -> None:
        if model_id not in {"xiaomi-ocr-0", "ovisocr3"}:
            raise ValueError(f"VISUAL_SERIAL_OCR_MODEL_UNSUPPORTED:{model_id}")
        self.model_id = model_id
        self.model_root = Path(model_root).resolve()
        self.max_new_tokens = int(max_new_tokens)
        self.processor = None
        self.model = None
        self.torch = None
        self.metrics: dict[str, Any] = {}
        self.identity: dict[str, Any] = {
            "model_id": model_id,
            "model_root": str(self.model_root),
            "prompt": prompt_for(model_id),
            "prompt_fingerprint": prompt_fingerprint(model_id),
            "prompt_source_sha256": prompt_source_sha256(),
            "max_new_tokens": self.max_new_tokens,
            "device": "cuda:0",
            "dtype": "bfloat16",
        }

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("VISUAL_SERIAL_GPU_REQUIRED")
        started = time.perf_counter()
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_root, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_root,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()
        self.metrics["load_seconds"] = time.perf_counter() - started
        self.identity.update(
            {
                "model_fingerprint": _sha256(self.model_root / "model.safetensors"),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
            }
        )

    def recognize_batch(self, crops: Sequence[Mapping[str, Any]]) -> list[str]:
        self._load()
        assert self.processor is not None and self.model is not None and self.torch is not None
        from PIL import Image

        values: list[str] = []
        prompt_variants: list[str] = []
        started = time.perf_counter()
        for crop in crops:
            with Image.open(str(crop["path"])) as source:
                image = source.convert("RGB")
                image.load()
            request_prompt = prompt_for_request(self.model_id, dict(crop))
            if request_prompt not in prompt_variants:
                prompt_variants.append(request_prompt)
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": request_prompt},
                    ],
                }
            ]
            if self.model_id == "xiaomi-ocr-0":
                inputs = self.processor.apply_chat_template(
                    message,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                ).to("cuda:0")
            else:
                chat = self.processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                inputs = self.processor(
                    text=[chat], images=[image], padding=True, return_tensors="pt"
                ).to("cuda:0")
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            self.torch.cuda.synchronize()
            generated = output[:, inputs["input_ids"].shape[-1] :]
            values.append(self.processor.batch_decode(generated, skip_special_tokens=True)[0])
            image.close()
            del inputs, output, generated
        self.metrics.update(
            {
                "inference_seconds": time.perf_counter() - started,
                "crop_count": len(crops),
                "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
            }
        )
        self.identity["prompt_variants_used"] = [
            prompt_fingerprint("ovisocr3-table") if value == prompt_for("ovisocr3-table") else prompt_fingerprint("ovisocr3-mixed-text")
            if value != prompt_for(self.model_id)
            else prompt_fingerprint(self.model_id)
            for value in prompt_variants
        ]
        return values

    def close(self) -> None:
        if self.torch is not None:
            self.torch.cuda.synchronize()
        self.model = None
        self.processor = None
        if self.torch is not None:
            self.torch.cuda.empty_cache()
        gc.collect()


from .visual_package import _write_visual_package


def create_developer_visual_serial_runtime_factory(
    *,
    layout_model_root: str | Path,
    formula_model_root: str | Path,
    text_model_root: str | Path,
    policy: VisualSerialPolicy | None = None,
    layout_threshold: float = 0.35,
    max_new_tokens: int = 2048,
) -> Any:
    """Return a factory for the installed developer-only model payloads."""

    policy = policy or VisualSerialPolicy()

    def factory() -> VisualSerialOrchestrator:
        return VisualSerialOrchestrator(
            layout_runtime=TransformersV3LayoutRuntime(
                layout_model_root, threshold=layout_threshold
            ),
            formula_runtime=TransformersVisionOcrRuntime(
                "xiaomi-ocr-0", formula_model_root, max_new_tokens=max_new_tokens
            ),
            text_runtime=TransformersVisionOcrRuntime(
                "ovisocr3", text_model_root, max_new_tokens=max_new_tokens
            ),
            result_factory=_write_visual_package,
            policy=policy,
        )

    return factory


__all__ = [
    "TransformersV3LayoutRuntime",
    "TransformersVisionOcrRuntime",
    "create_developer_visual_serial_runtime_factory",
]

