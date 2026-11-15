from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from ctypes import wintypes
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

from .wmf import WmfInspector


@dataclass(frozen=True)
class WmfRenderMetadata:
    status: str
    source_type: str = "wmf"
    renderer: str = "win32_gdi"
    bounds_source: str | None = None
    logical_bounds: tuple[int, int, int, int] | None = None
    natural_aspect_ratio: float | None = None
    render_dpi: int = 0
    supersample: int = 0
    raw_size: tuple[int, int] = (0, 0)
    crop_box: tuple[int, int, int, int] | None = None
    final_size: tuple[int, int] = (0, 0)
    foreground_pixel_ratio: float = 0.0
    padding: int = 0
    scale_x: float = 0.0
    scale_y: float = 0.0
    wall_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class WmfRenderer:
    """Deterministic WMF -> PNG renderer backed by the Windows GDI layer."""

    def __init__(
        self,
        *,
        dpi: int = 600,
        supersample: int = 3,
        padding: int = 12,
        max_width: int = 12000,
        max_height: int = 12000,
        max_pixels: int = 64_000_000,
        max_dimension_inches: float = 100.0,
        max_records: int = 250_000,
        max_input_bytes: int = 64 * 1024 * 1024,
    ):
        self.dpi = dpi
        self.supersample = supersample
        self.padding = padding
        self.max_width = max_width
        self.max_height = max_height
        self.max_pixels = max_pixels
        self.max_dimension_inches = max_dimension_inches
        self.max_records = max_records
        self.max_input_bytes = max_input_bytes

    def render(
        self,
        source: str | Path,
        target: str | Path,
        *,
        extent_emu: tuple[int, int] | None = None,
    ) -> WmfRenderMetadata:
        return self.render_bytes(Path(source).read_bytes(), target, extent_emu=extent_emu)

    def render_bytes(
        self,
        data: bytes,
        target: str | Path,
        *,
        extent_emu: tuple[int, int] | None = None,
    ) -> WmfRenderMetadata:
        import time

        started = time.perf_counter()
        target = Path(target)
        inspection = WmfInspector(
            max_records=self.max_records,
            max_input_bytes=self.max_input_bytes,
        ).inspect(data)
        if not inspection.valid:
            status = (
                "resource_limit"
                if inspection.error and "limit exceeded" in inspection.error
                else "failed"
            )
            return WmfRenderMetadata(
                status=status,
                logical_bounds=inspection.logical_bounds,
                render_dpi=self.dpi,
                supersample=self.supersample,
                wall_seconds=time.perf_counter() - started,
                error=inspection.error,
            )
        geometry = self._geometry(inspection, extent_emu)
        if geometry is None:
            return WmfRenderMetadata(
                status="unsupported_geometry",
                logical_bounds=inspection.logical_bounds,
                render_dpi=self.dpi,
                supersample=self.supersample,
                wall_seconds=time.perf_counter() - started,
                error="WMF has no reliable placeable or Word extent geometry",
            )
        bounds_source, width_in, height_in = geometry
        if (
            width_in > self.max_dimension_inches
            or height_in > self.max_dimension_inches
        ):
            return WmfRenderMetadata(
                status="resource_limit",
                bounds_source=bounds_source,
                logical_bounds=inspection.logical_bounds,
                natural_aspect_ratio=width_in / height_in,
                render_dpi=self.dpi,
                supersample=self.supersample,
                wall_seconds=time.perf_counter() - started,
                error=(
                    "WMF physical dimension limit exceeded: "
                    f"{width_in:.3f}x{height_in:.3f} inches"
                ),
            )
        aspect = width_in / height_in
        base_width = max(1, math.ceil(width_in * self.dpi))
        base_height = max(1, math.ceil(height_in * self.dpi))
        if (
            base_width > self.max_width
            or base_height > self.max_height
            or base_width * base_height > self.max_pixels
        ):
            return WmfRenderMetadata(
                status="resource_limit",
                bounds_source=bounds_source,
                logical_bounds=inspection.logical_bounds,
                natural_aspect_ratio=aspect,
                render_dpi=self.dpi,
                supersample=self.supersample,
                raw_size=(base_width, base_height),
                wall_seconds=time.perf_counter() - started,
                error=(
                    "WMF bitmap bounds exceed renderer limits: "
                    f"{base_width}x{base_height}, {base_width * base_height} pixels"
                ),
            )
        sample = max(1, self.supersample)
        raw_width, raw_height, sample = self._guard_size(
            base_width, base_height, sample
        )
        scale = raw_width / max(1, inspection.logical_bounds[2] - inspection.logical_bounds[0]) if inspection.logical_bounds else raw_width / base_width
        try:
            image = self._render_gdi(data, raw_width, raw_height, width_in, height_in)
        except Exception as exc:  # noqa: BLE001 - isolate one malformed metafile
            return WmfRenderMetadata(
                status="failed",
                bounds_source=bounds_source,
                logical_bounds=inspection.logical_bounds,
                natural_aspect_ratio=aspect,
                render_dpi=self.dpi,
                supersample=sample,
                raw_size=(raw_width, raw_height),
                scale_x=scale,
                scale_y=scale,
                wall_seconds=time.perf_counter() - started,
                error=str(exc),
            )

        difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
        foreground = difference.convert("L")
        bbox = foreground.getbbox()
        nonzero = sum(count for value, count in enumerate(foreground.histogram()) if value)
        ratio = nonzero / (raw_width * raw_height)
        if bbox is None:
            target.unlink(missing_ok=True)
            return WmfRenderMetadata(
                status="empty",
                bounds_source=bounds_source,
                logical_bounds=inspection.logical_bounds,
                natural_aspect_ratio=aspect,
                render_dpi=self.dpi,
                supersample=sample,
                raw_size=(raw_width, raw_height),
                foreground_pixel_ratio=0.0,
                scale_x=scale,
                scale_y=scale,
                wall_seconds=time.perf_counter() - started,
            )

        crop = image.crop(bbox)
        raw_padding = self.padding * sample
        crop = ImageOps.expand(crop, border=raw_padding, fill="white")
        if sample > 1:
            final_size = (
                max(1, round(crop.width / sample)),
                max(1, round(crop.height / sample)),
            )
            crop = crop.resize(final_size, Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        crop.save(target, format="PNG", optimize=False)
        return WmfRenderMetadata(
            status="success",
            bounds_source=bounds_source,
            logical_bounds=inspection.logical_bounds,
            natural_aspect_ratio=aspect,
            render_dpi=self.dpi,
            supersample=sample,
            raw_size=(raw_width, raw_height),
            crop_box=bbox,
            final_size=crop.size,
            foreground_pixel_ratio=ratio,
            padding=self.padding,
            scale_x=scale,
            scale_y=scale,
            wall_seconds=time.perf_counter() - started,
        )

    def _geometry(self, inspection, extent_emu):
        if (
            inspection.placeable_header
            and inspection.logical_bounds
            and inspection.inch
            and inspection.inch > 0
        ):
            left, top, right, bottom = inspection.logical_bounds
            if right > left and bottom > top:
                return (
                    "placeable_header",
                    (right - left) / inspection.inch,
                    (bottom - top) / inspection.inch,
                )
        if extent_emu and extent_emu[0] > 0 and extent_emu[1] > 0:
            return "word_extent", extent_emu[0] / 914400, extent_emu[1] / 914400
        return None

    def _guard_size(self, width, height, sample):
        raw_width, raw_height = width * sample, height * sample
        limit = min(
            1.0,
            self.max_width / raw_width,
            self.max_height / raw_height,
            math.sqrt(self.max_pixels / (raw_width * raw_height)),
        )
        if limit < 1:
            raw_width = max(1, math.floor(raw_width * limit))
            raw_height = max(1, math.floor(raw_height * limit))
            sample = max(1, min(sample, math.floor(sample * limit)))
        return raw_width, raw_height, sample

    @staticmethod
    def _render_gdi(data, width, height, width_in, height_in):
        if os.name != "nt":
            raise RuntimeError("Win32 GDI renderer is only available on Windows")

        class METAFILEPICT(ctypes.Structure):
            _fields_ = [
                ("mm", wintypes.LONG),
                ("xExt", wintypes.LONG),
                ("yExt", wintypes.LONG),
                ("hMF", wintypes.HANDLE),
            ]

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class RGBQUAD(ctypes.Structure):
            _fields_ = [
                ("rgbBlue", ctypes.c_ubyte),
                ("rgbGreen", ctypes.c_ubyte),
                ("rgbRed", ctypes.c_ubyte),
                ("rgbReserved", ctypes.c_ubyte),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", RGBQUAD * 1)]

        gdi = ctypes.WinDLL("gdi32", use_last_error=True)
        gdi.SetWinMetaFileBits.argtypes = [
            wintypes.UINT,
            ctypes.c_void_p,
            wintypes.HDC,
            ctypes.POINTER(METAFILEPICT),
        ]
        gdi.SetWinMetaFileBits.restype = wintypes.HANDLE
        gdi.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi.CreateCompatibleDC.restype = wintypes.HDC
        gdi.CreateDIBSection.argtypes = [
            wintypes.HDC,
            ctypes.POINTER(BITMAPINFO),
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        gdi.CreateDIBSection.restype = wintypes.HBITMAP
        gdi.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi.SelectObject.restype = wintypes.HGDIOBJ
        gdi.PlayEnhMetaFile.argtypes = [
            wintypes.HDC,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.RECT),
        ]
        gdi.PlayEnhMetaFile.restype = wintypes.BOOL
        gdi.DeleteEnhMetaFile.argtypes = [wintypes.HANDLE]
        gdi.DeleteEnhMetaFile.restype = wintypes.BOOL
        gdi.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi.DeleteObject.restype = wintypes.BOOL
        gdi.DeleteDC.argtypes = [wintypes.HDC]
        gdi.DeleteDC.restype = wintypes.BOOL

        placeable = len(data) >= 22 and int.from_bytes(data[:4], "little") == 0x9AC6CDD7
        wmf = data[22:] if placeable else data
        buffer = ctypes.create_string_buffer(wmf)
        pict = METAFILEPICT(
            8,
            max(1, round(width_in * 2540)),
            max(1, round(height_in * 2540)),
            None,
        )
        hemf = gdi.SetWinMetaFileBits(len(wmf), buffer, None, ctypes.byref(pict))
        if not hemf:
            raise ctypes.WinError(ctypes.get_last_error())
        dc = bitmap = old = None
        try:
            dc = gdi.CreateCompatibleDC(None)
            if not dc:
                raise ctypes.WinError(ctypes.get_last_error())
            info = BITMAPINFO()
            info.bmiHeader = BITMAPINFOHEADER(
                ctypes.sizeof(BITMAPINFOHEADER),
                width,
                -height,
                1,
                32,
                0,
                width * height * 4,
                0,
                0,
                0,
                0,
            )
            bits = ctypes.c_void_p()
            bitmap = gdi.CreateDIBSection(dc, ctypes.byref(info), 0, ctypes.byref(bits), None, 0)
            if not bitmap or not bits.value:
                raise ctypes.WinError(ctypes.get_last_error())
            ctypes.memset(bits, 0xFF, width * height * 4)
            old = gdi.SelectObject(dc, bitmap)
            rect = wintypes.RECT(0, 0, width, height)
            if not gdi.PlayEnhMetaFile(dc, hemf, ctypes.byref(rect)):
                raise ctypes.WinError(ctypes.get_last_error())
            raw = ctypes.string_at(bits, width * height * 4)
            return Image.frombytes("RGB", (width, height), raw, "raw", "BGRX")
        finally:
            if dc and old:
                gdi.SelectObject(dc, old)
            if bitmap:
                gdi.DeleteObject(bitmap)
            if dc:
                gdi.DeleteDC(dc)
            gdi.DeleteEnhMetaFile(hemf)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wmf_renderer")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--extent-emu", nargs=2, type=int, metavar=("CX", "CY"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = WmfRenderer(dpi=args.dpi, supersample=args.supersample).render(
        args.input,
        args.output,
        extent_emu=tuple(args.extent_emu) if args.extent_emu else None,
    )
    print(json.dumps(metadata.to_dict(), ensure_ascii=False))
    return 0 if metadata.status in {"success", "empty"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
