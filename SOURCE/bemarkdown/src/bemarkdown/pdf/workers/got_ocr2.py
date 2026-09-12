"""Low-VRAM packaged worker for the production GOT-OCR2 stage."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from PIL import Image

from ...text_recognition_contract import normalize_text
from ..got_batching import GOT_BATCH_RETRY_POLICY, adaptive_got_batches
from ..vram_monitor import VramMonitor

HUNYUAN_MAX_PIXELS_6GB = 4 * 1024 * 1024


def generated_length(token_ids, *, eos_token_ids: set[int], pad_token_id: int | None) -> tuple[int, bool]:
    """Return this sequence's length and termination, ignoring batch padding."""
    for index, token in enumerate(token_ids):
        if int(token) in eos_token_ids:
            return index + 1, True
        if pad_token_id is not None and int(token) == pad_token_id:
            return index, True
    return len(token_ids), False


def install_got_vision_sdpa(model) -> int:
    """Fuse the existing GOT vision attention while preserving relative positions."""
    import types
    import torch

    def make_forward(original):
        def forward(self, hidden_states, output_attentions=None):
            if output_attentions:
                return original(hidden_states, output_attentions=output_attentions)
            batch, height, width, _ = hidden_states.shape
            qkv = self.qkv(hidden_states).reshape(batch, height * width, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)
            bias = None
            if self.use_rel_pos:
                bias = self.get_decomposed_rel_pos(
                    query.reshape(batch * self.num_attention_heads, height * width, -1),
                    self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
                ).reshape(batch, self.num_attention_heads, height * width, height * width)
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=bias,
                dropout_p=self.dropout if self.training else 0.0, scale=self.scale
            ).transpose(1, 2).reshape(batch, height, width, -1)
            return self.proj(output), None
        return forward

    count = 0
    for layer in model.modules():
        if type(layer).__name__ == 'GotOcr2VisionAttention':
            if not getattr(layer, '_bemarkdown_sdpa', False):
                layer.forward = types.MethodType(make_forward(layer.forward), layer)
                layer._bemarkdown_sdpa = True
            count += 1
    return count


def install_got_vision_microbatch(model, *, chunk_size: int = 1) -> None:
    """Bound vision attention memory while leaving text generation batched."""
    import torch

    original = model.model.get_image_features

    def encode(pixel_values, **kwargs):
        if len(pixel_values) <= chunk_size:
            return original(pixel_values=pixel_values, **kwargs)
        outputs = [original(pixel_values=pixel_values[i:i+chunk_size], **kwargs)
                   for i in range(0, len(pixel_values), chunk_size)]

        def concatenate(values):
            first = values[0]
            if isinstance(first, torch.Tensor):
                return torch.cat(values, dim=0)
            if isinstance(first, (tuple, list)):
                return type(first)(concatenate(list(items)) for items in zip(*values, strict=True))
            if all(value is None for value in values):
                return None
            raise TypeError('GOT_VISION_MICROBATCH_OUTPUT_CONTRACT')

        combined = type(outputs[0])(**{key: concatenate([value[key] for value in outputs]) for key in outputs[0]})
        # Transformers attaches projected features dynamically; they are not
        # a dataclass/dictionary field of GotOcr2VisionEncoderOutput.
        combined.pooler_output = torch.cat([value.pooler_output for value in outputs], dim=0)
        return combined

    model.model.get_image_features = encode


def prepare_provider_image(
    provider: str, image: Image.Image
) -> tuple[Image.Image, dict[str, Any]]:
    source_width, source_height = image.size
    target_width, target_height = source_width, source_height
    if provider == "HUNYUAN" and source_width * source_height > HUNYUAN_MAX_PIXELS_6GB:
        scale = math.sqrt(HUNYUAN_MAX_PIXELS_6GB / (source_width * source_height))
        target_width = max(1, math.floor(source_width * scale))
        target_height = max(1, math.floor(source_height * scale))
    resized = (target_width, target_height) != (source_width, source_height)
    prepared = (
        image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        if resized
        else image
    )
    return prepared, {
        "source_size": [source_width, source_height],
        "model_input_size": [target_width, target_height],
        "max_pixels": HUNYUAN_MAX_PIXELS_6GB if provider == "HUNYUAN" else None,
        "resized": resized,
        "content_cropped": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("GOT", "HUNYUAN"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--generation-budget", type=int, required=True)
    parser.add_argument("--vram-limit-mib", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument('--vision-batch-size', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--freeze-lock", type=Path)
    args = parser.parse_args()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    import psutil
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))["rows"]
    visual_freeze_lock_sha256 = _verify_freeze_lock(args.manifest, args.freeze_lock)
    for row in rows:
        crop_path = (args.crop_root / row["crop_local_relpath"]).resolve()
        try:
            crop_path.relative_to(args.crop_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"CROP_PATH_OUTSIDE_ROOT:{crop_path}") from exc
        if _sha256_file(crop_path) != row["crop_sha256"]:
            raise RuntimeError(f"CROP_SHA_MISMATCH:{row['sample_id']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device_properties = torch.cuda.get_device_properties(0)
    total_mib = device_properties.total_memory / 1024**2
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, args.vram_limit_mib / total_mib), 0
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    load_started = time.perf_counter()
    error = None
    outputs: list[dict[str, Any]] = []
    latencies: list[float] = []
    generated_lengths: list[int] = []
    input_token_lengths: list[int] = []
    input_transforms: list[dict[str, Any]] = []
    oom_events: list[dict[str, Any]] = []
    successful_batch_sizes: list[int] = []
    process = psutil.Process()
    peak_ram = process.memory_info().rss
    with VramMonitor(device_selector=str(getattr(device_properties, 'uuid', '0'))) as nvml:
        try:
            processor = AutoProcessor.from_pretrained(
                args.model_root, local_files_only=True
            )
            if (
                args.provider == "HUNYUAN"
                and getattr(processor, "tokenizer", None) is not None
            ):
                processor.tokenizer.padding_side = "left"
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_root,
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map="cuda",
            )
            model.eval()
            if args.provider == 'GOT' and not install_got_vision_sdpa(model):
                raise RuntimeError('GOT_VISION_ATTENTION_IMPLEMENTATION_UNAVAILABLE')
            if args.provider == "GOT" and args.batch_size > 1:
                install_got_vision_microbatch(model, chunk_size=args.vision_batch_size)
            load_seconds = time.perf_counter() - load_started
            parameter_devices = sorted(
                {parameter.device.type for parameter in model.parameters()}
            )
            warmup_count = min(max(0, args.warmup), len(rows))
            if warmup_count and args.provider != "GOT":
                raise RuntimeError("WARMUP_CURRENTLY_SUPPORTED_FOR_GOT_ONLY")
            warmup_started = time.perf_counter()
            for row in rows[:warmup_count]:
                with Image.open(args.crop_root / row["crop_local_relpath"]) as image:
                    prepared, _transform = prepare_provider_image(
                        args.provider, image.convert("RGB")
                    )
                warm_inputs = processor([prepared], return_tensors="pt", padding=True)
                warm_inputs = warm_inputs.to("cuda")
                with torch.inference_mode():
                    model.generate(
                        **warm_inputs,
                        do_sample=False,
                        tokenizer=processor.tokenizer,
                        stop_strings="<|im_end|>",
                        max_new_tokens=args.generation_budget,
                        use_cache=True,
                    )
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - warmup_started
            first_latency = None
            def infer_batch(batch_rows):
                batch_outputs = []
                images = []
                batch_transforms = []
                for row in batch_rows:
                    with Image.open(
                        args.crop_root / row["crop_local_relpath"]
                    ) as image:
                        prepared, transform = prepare_provider_image(
                            args.provider, image.convert("RGB")
                        )
                        images.append(prepared)
                        batch_transforms.append(transform)
                prep_started = time.perf_counter()
                if args.provider == "GOT":
                    modes = {row.get('ocr_format', False) for row in batch_rows}
                    if len(modes) != 1 or any(type(mode) is not bool for mode in modes):
                        raise ValueError('GOT_FORMAT_MODE_MIXED_OR_INVALID')
                    mode = next(iter(modes))
                    inputs = processor(images, return_tensors="pt", padding=True, **({'format': True} if mode else {}))
                    generate_kwargs = {
                        "do_sample": False,
                        "tokenizer": processor.tokenizer,
                        "stop_strings": "<|im_end|>",
                        "eos_token_id": processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                        "pad_token_id": processor.tokenizer.pad_token_id,
                        "max_new_tokens": args.generation_budget,
                        "use_cache": True,
                    }
                else:
                    messages = [
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {
                                        "type": "text",
                                        "text": (
                                            "逐字转写图中可见文字，不解释、不总结、"
                                            "不纠错、不补全。"
                                        ),
                                    },
                                ],
                            }
                        ]
                        for image in images
                    ]
                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs.pop("token_type_ids", None)
                    generate_kwargs = {
                        "do_sample": False,
                        "max_new_tokens": args.generation_budget,
                        "use_cache": True,
                        "repetition_penalty": 1.08,
                    }
                preprocess_seconds = time.perf_counter() - prep_started
                input_tokens = int(inputs["input_ids"].shape[-1])
                inputs = inputs.to("cuda")
                torch.cuda.synchronize()
                inference_started = time.perf_counter()
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, **generate_kwargs)
                torch.cuda.synchronize()
                batch_latency = time.perf_counter() - inference_started
                generated = generated_ids[:, input_tokens:]
                decoded = processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if len(decoded) != len(batch_rows):
                    raise RuntimeError("BATCH_DECODE_CARDINALITY_MISMATCH")
                per_item = batch_latency / len(batch_rows)
                for row, raw, token_ids, transform in zip(
                    batch_rows, decoded, generated, batch_transforms, strict=True
                ):
                    effective = raw.strip()
                    if args.provider == "HUNYUAN":
                        fence = re.fullmatch(
                            r"```(?:text|plaintext)?\s*\n?(.*?)\n?```",
                            effective,
                            re.DOTALL,
                        )
                        if fence:
                            effective = fence.group(1).strip()
                    eos_ids = set()
                    for value in (generate_kwargs.get("eos_token_id"), model.generation_config.eos_token_id,
                                  processor.tokenizer.eos_token_id):
                        if value is not None:
                            eos_ids.update(value if isinstance(value, (list, tuple)) else [value])
                    token_count, terminated = generated_length(
                        token_ids.tolist(), eos_token_ids=eos_ids,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                    batch_outputs.append(
                        {
                            "sample_id": row["sample_id"],
                            "raw_output": raw,
                            "normalized_output": normalize_text(effective),
                            "generated_token_count": token_count,
                            "generation_limit_reached": (
                                token_count >= args.generation_budget and not terminated
                            ),
                            "source_crop_sha256": row["crop_sha256"],
                            "model_input_transform": transform,
                            "input_token_count": input_tokens,
                            "preprocess_seconds": preprocess_seconds / len(batch_rows),
                            "latency_seconds": per_item,
                            "visual_freeze_lock_sha256": visual_freeze_lock_sha256,
                        }
                    )
                return batch_outputs, batch_latency

            def release_failed_batch():
                gc.collect()
                torch.cuda.empty_cache()

            def is_cuda_oom(exc):
                return args.provider == 'GOT' and (isinstance(exc, torch.cuda.OutOfMemoryError)
                    or isinstance(exc, RuntimeError) and 'CUDA out of memory' in str(exc))

            from bemarkdown.pdf.got_batching import got_mode_groups
            def mode_batches():
                for group in got_mode_groups(rows):
                    yield from adaptive_got_batches(
                        group, batch_size=args.batch_size, infer=infer_batch, is_oom=is_cuda_oom,
                        release=release_failed_batch, events=oom_events)
            for batch_rows, (batch_outputs, batch_latency) in mode_batches():
                successful_batch_sizes.append(len(batch_rows))
                if first_latency is None:
                    first_latency = batch_latency
                outputs.extend(batch_outputs)
                latencies.extend(row['latency_seconds'] for row in batch_outputs)
                generated_lengths.extend(row['generated_token_count'] for row in batch_outputs)
                input_token_lengths.extend(row['input_token_count'] for row in batch_outputs)
                input_transforms.extend(row['model_input_transform'] for row in batch_outputs)
                peak_ram = max(peak_ram, process.memory_info().rss)
        except Exception as exc:  # noqa: BLE001 - candidate failure is evidence
            error = f"{type(exc).__name__}: {exc}"
            load_seconds = time.perf_counter() - load_started
            parameter_devices = []
            first_latency = None
    peak_torch = max(
        torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    )
    metrics = {
        "schema": "bemarkdown-low-vram-provider-benchmark-v1",
        "provider": args.provider,
        "backend": "Transformers",
        "population": len(rows),
        "successful_outputs": len(outputs),
        "batch_size": args.batch_size,
        'vision_batch_size': args.vision_batch_size,
        "generation_budget": args.generation_budget,
        "vram_allocator_limit_mib": args.vram_limit_mib,
        "model_load_seconds": load_seconds,
        "warmup_count": locals().get("warmup_count", 0),
        "warmup_seconds": locals().get("warmup_seconds", 0.0),
        "first_inference_seconds": first_latency,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "regions_per_second": len(outputs) / sum(latencies) if latencies else 0.0,
        "wall_seconds": time.perf_counter() - started,
        "peak_ram_bytes": peak_ram,
        "peak_torch_vram_bytes": int(peak_torch),
        "peak_nvml_process_vram_mib": nvml.peak_mib,
        "peak_nvml_global_used_vram_mib": nvml.peak_global_mib,
        "nvml_sampling_error_count": nvml.sampling_error_count,
        "gpu_memory_observation": nvml.summary(),
        "oom_count": len(oom_events) or int(error is not None and "OutOfMemory" in error),
        "batch_execution": {"policy": GOT_BATCH_RETRY_POLICY,
                            "successful_batch_sizes": successful_batch_sizes, "oom_events": oom_events},
        "cpu_fallback_count": int(
            any(device != "cuda" for device in parameter_devices)
        ),
        "parameter_devices": parameter_devices,
        "generation_tokens": {
            "max": max(generated_lengths, default=0),
            "mean": (
                statistics.fmean(generated_lengths) if generated_lengths else 0.0
            ),
        },
        "input_tokens": {
            "max": max(input_token_lengths, default=0),
            "mean": (
                statistics.fmean(input_token_lengths) if input_token_lengths else 0.0
            ),
        },
        "resized_input_count": sum(bool(row["resized"]) for row in input_transforms),
        "hunyuan_max_pixels_6gb": (
            HUNYUAN_MAX_PIXELS_6GB if args.provider == "HUNYUAN" else None
        ),
        "truncated_outputs": sum(
            bool(row["generation_limit_reached"]) for row in outputs
        ),
        "error": error,
        "output_jsonl": str(args.output.with_suffix(".jsonl").resolve()),
        "visual_freeze_lock_sha256": visual_freeze_lock_sha256,
    }
    args.output.with_suffix(".jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in outputs
        ),
        encoding="utf-8",
        newline="\n",
    )
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        del model, processor
    except UnboundLocalError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_freeze_lock(manifest_path: Path, lock_path: Path | None) -> str | None:
    if lock_path is None:
        return None
    lock_path = lock_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    digest = _sha256_file(lock_path)
    if (
        lock.get("gate") != "VISUAL_AUDIT_FROZEN"
        or manifest.get("visual_freeze_lock_sha256") != digest
    ):
        raise RuntimeError("VISUAL_FREEZE_LOCK_INVALID")
    return digest


if __name__ == "__main__":
    raise SystemExit(main())
