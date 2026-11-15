from __future__ import annotations

import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

PLACEABLE_KEY = 0x9AC6CDD7
META_ESCAPE = 0x0626
MFCOMMENT = 15

# Records that directly create visible pixels.  This list follows MS-WMF
# record names; anything outside the visible and known-state sets is treated
# as uncertain, never as proof that a preview is empty.
_TEXT_RECORDS = {0x0521, 0x0A32}  # TEXTOUT, EXTTEXTOUT
_DRAWING_RECORDS = {
    0x0213,  # LINETO
    0x0418,  # ELLIPSE
    0x0419,  # FLOODFILL
    0x041B,  # RECTANGLE
    0x041F,  # SETPIXEL
    0x061C,  # ROUNDRECT
    0x061D,  # PATBLT
    0x0817,  # ARC
    0x081A,  # PIE
    0x0830,  # CHORD
    0x0922,  # BITBLT
    0x0B23,  # STRETCHBLT
    0x0324,  # POLYGON
    0x0325,  # POLYLINE
    0x0228,  # FILLREGION
    0x0429,  # FRAMEREGION
    0x012A,  # INVERTREGION
    0x012B,  # PAINTREGION
    0x0538,  # POLYPOLYGON
    0x0940,  # DIBBITBLT
    0x0B41,  # DIBSTRETCHBLT
    0x0F43,  # STRETCHDIB
    0x0548,  # EXTFLOODFILL
}
_STATE_RECORDS = {
    0x0000,  # EOF
    0x001E,  # SAVEDC
    0x0102,  # SETBKMODE
    0x0103,  # SETMAPMODE
    0x0104,  # SETROP2
    0x0105,  # SETRELABS
    0x0106,  # SETPOLYFILLMODE
    0x0107,  # SETSTRETCHBLTMODE
    0x0108,  # SETTEXTCHAREXTRA
    0x0109,  # SETTEXTCOLOR
    0x0127,  # RESTOREDC
    0x012D,  # SELECTOBJECT
    0x012E,  # SETTEXTALIGN
    0x012F,  # RESIZEPALETTE
    0x0139,  # SELECTPALETTE
    0x0142,  # DIBCREATEPATTERNBRUSH
    0x01F0,  # DELETEOBJECT
    0x0201,  # SETBKCOLOR
    0x0209,  # SETTEXTCOLOR (alternate encoded parameter count)
    0x020A,  # SETTEXTJUSTIFICATION
    0x020B,  # SETWINDOWORG
    0x020C,  # SETWINDOWEXT
    0x020D,  # SETVIEWPORTORG
    0x020E,  # SETVIEWPORTEXT
    0x020F,  # OFFSETWINDOWORG
    0x0211,  # OFFSETVIEWPORTORG
    0x0214,  # MOVETO
    0x0220,  # OFFSETCLIPRGN
    0x022C,  # SELECTCLIPREGION
    0x0231,  # SETMAPPERFLAGS
    0x0234,  # SELECTPALETTE
    0x0235,  # REALIZEPALETTE
    0x02FA,  # CREATEPENINDIRECT
    0x02FB,  # CREATEFONTINDIRECT
    0x02FC,  # CREATEBRUSHINDIRECT
    0x02FD,  # CREATEBITMAPINDIRECT
    0x02FE,  # CREATEBITMAP
    0x0300,  # SETPALENTRIES
    0x0320,  # OFFSETCLIPRGN (alternate)
    0x0415,  # EXCLUDECLIPRECT
    0x0416,  # INTERSECTCLIPRECT
    0x0436,  # ANIMATEPALETTE
    0x0437,  # SETPALENTRIES
    0x04F7,  # CREATEPALETTE
    0x06FF,  # CREATEREGION
}


@dataclass(frozen=True)
class WmfInspection:
    file_type: str
    valid: bool
    placeable_header: bool
    logical_bounds: tuple[int, int, int, int] | None
    inch: int | None
    record_count: int
    drawing_record_count: int
    text_record_count: int
    unknown_record_count: int
    embedded_comment_presence: bool
    mathtype_comment_presence: bool
    embedded_mtef_presence: bool
    embedded_mathml_presence: bool
    visual_state: str
    embedded_mtef: bytes | None = field(default=None, repr=False)
    embedded_mathml: str | None = field(default=None, repr=False)
    warnings: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self, *, include_payloads: bool = False) -> dict:
        result = asdict(self)
        if include_payloads:
            result["embedded_mtef_hex"] = (
                self.embedded_mtef.hex() if self.embedded_mtef else None
            )
        result.pop("embedded_mtef", None)
        if not include_payloads:
            result.pop("embedded_mathml", None)
        return result


class WmfInspector:
    """Conservative structural inspector for classic Windows metafiles."""

    def __init__(
        self,
        *,
        max_input_bytes: int = 64 * 1024 * 1024,
        max_records: int = 250_000,
        max_comment_bytes: int = 16 * 1024 * 1024,
    ):
        self.max_input_bytes = max_input_bytes
        self.max_records = max_records
        self.max_comment_bytes = max_comment_bytes

    def inspect(self, source: bytes | str | Path) -> WmfInspection:
        data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
        warnings: list[str] = []
        if len(data) > self.max_input_bytes:
            return self._invalid(
                False,
                None,
                None,
                warnings,
                f"WMF input byte limit exceeded: {len(data)} > {self.max_input_bytes}",
            )
        placeable = len(data) >= 22 and struct.unpack_from("<I", data, 0)[0] == PLACEABLE_KEY
        bounds = None
        inch = None
        header_offset = 22 if placeable else 0
        if placeable:
            _, _, left, top, right, bottom, inch, _, checksum = struct.unpack_from(
                "<IHhhhhHIH", data, 0
            )
            bounds = (left, top, right, bottom)
            calculated = 0
            for word in struct.unpack_from("<10H", data, 0):
                calculated ^= word
            if calculated != checksum:
                warnings.append("placeable header checksum mismatch")
            if inch == 0 or right <= left or bottom <= top:
                warnings.append("placeable header has invalid geometry")

        if len(data) < header_offset + 18:
            return self._invalid(placeable, bounds, inch, warnings, "truncated METAHEADER")
        try:
            file_type, header_words, version, _, _, _, _ = struct.unpack_from(
                "<HHHIHIH", data, header_offset
            )
        except struct.error as exc:
            return self._invalid(placeable, bounds, inch, warnings, str(exc))
        if header_words < 9 or file_type not in {1, 2} or version not in {0x0100, 0x0300}:
            return self._invalid(
                placeable, bounds, inch, warnings, "invalid WMF METAHEADER"
            )

        offset = header_offset + header_words * 2
        record_count = drawing = text = unknown = 0
        comments: list[bytes] = []
        saw_eof = False
        error = None
        while offset + 6 <= len(data):
            size_words, function = struct.unpack_from("<IH", data, offset)
            size_bytes = int(size_words) * 2
            if size_words < 3 or offset + size_bytes > len(data):
                error = "truncated or invalid WMF record table"
                break
            record_count += 1
            if record_count > self.max_records:
                error = f"WMF record limit exceeded: {record_count} > {self.max_records}"
                break
            if function in _TEXT_RECORDS:
                text += 1
            elif function in _DRAWING_RECORDS:
                drawing += 1
            elif function == META_ESCAPE:
                if size_bytes < 10:
                    unknown += 1
                else:
                    escape, byte_count = struct.unpack_from("<HH", data, offset + 6)
                    available = size_bytes - 10
                    payload = data[
                        offset + 10 : offset + 10 + min(int(byte_count), available)
                    ]
                    if escape == MFCOMMENT:
                        if sum(map(len, comments)) + len(payload) > self.max_comment_bytes:
                            error = (
                                "WMF embedded comment byte limit exceeded: "
                                f"> {self.max_comment_bytes}"
                            )
                            break
                        comments.append(payload)
                    else:
                        unknown += 1
            elif function not in _STATE_RECORDS:
                unknown += 1
            offset += size_bytes
            if function == 0:
                saw_eof = True
                break
        if not saw_eof and error is None:
            error = "WMF record table has no META_EOF"

        mtef, mathml, mathtype = self._embedded_payloads(comments)
        valid = error is None
        if drawing or text:
            visual_state = "visible"
        elif valid and unknown == 0:
            visual_state = "empty"
        else:
            visual_state = "uncertain"
        return WmfInspection(
            file_type="wmf",
            valid=valid,
            placeable_header=placeable,
            logical_bounds=bounds,
            inch=inch,
            record_count=record_count,
            drawing_record_count=drawing,
            text_record_count=text,
            unknown_record_count=unknown,
            embedded_comment_presence=bool(comments),
            mathtype_comment_presence=mathtype,
            embedded_mtef_presence=mtef is not None,
            embedded_mathml_presence=mathml is not None,
            visual_state=visual_state,
            embedded_mtef=mtef,
            embedded_mathml=mathml,
            warnings=tuple(warnings),
            error=error,
        )

    @staticmethod
    def _invalid(placeable, bounds, inch, warnings, error) -> WmfInspection:
        return WmfInspection(
            "wmf",
            False,
            placeable,
            bounds,
            inch,
            0,
            0,
            0,
            0,
            False,
            False,
            False,
            False,
            "uncertain",
            warnings=tuple(warnings),
            error=error,
        )

    @classmethod
    def _embedded_payloads(
        cls, comments: list[bytes]
    ) -> tuple[bytes | None, str | None, bool]:
        mathtype = any(
            marker in comment
            for comment in comments
            for marker in (b"MathType", b"AppsMFCC", b"MathTypeUU")
        )
        application_chunks: dict[bytes, list[bytes]] = {}
        old_mtef = None
        for comment in comments:
            if comment.startswith(b"AppsMFCC") and len(comment) >= 18:
                _, _, chunk_size = struct.unpack_from("<HII", comment, 8)
                name_end = comment.find(b"\x00", 18)
                if name_end > 18:
                    app_name = comment[18:name_end]
                    chunk = comment[name_end + 1 :]
                    if chunk_size <= len(chunk):
                        chunk = chunk[:chunk_size]
                    application_chunks.setdefault(app_name, []).append(chunk)
            marker = comment.find(b"MathType")
            magic = comment.find(b"\x55\x55", marker + 8 if marker >= 0 else 0)
            if marker >= 0 and magic >= 0 and magic + 4 <= len(comment):
                length = struct.unpack_from("<H", comment, magic + 2)[0]
                candidate = comment[magic + 4 : magic + 4 + length]
                if candidate[:1] in {b"\x03", b"\x05"}:
                    old_mtef = candidate

        preferred = b"".join(application_chunks.get(b"Design Science, Inc.", []))
        all_payloads = [preferred] if preferred else []
        all_payloads.extend(b"".join(chunks) for chunks in application_chunks.values())
        mathml = next((cls._extract_mathml(item) for item in all_payloads if item), None)
        mtef = next(
            (item for item in all_payloads if item[:1] in {b"\x03", b"\x05"}),
            old_mtef,
        )
        return mtef, mathml, mathtype

    @staticmethod
    def _extract_mathml(payload: bytes) -> str | None:
        for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
            try:
                text = payload.decode(encoding)
            except UnicodeDecodeError:
                continue
            match = re.search(
                r"<(?:[A-Za-z_][\w.-]*:)?math\b.*?</(?:[A-Za-z_][\w.-]*:)?math\s*>",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if match:
                return match.group(0)
            empty = re.search(
                r"<(?:[A-Za-z_][\w.-]*:)?math\b[^>]*/\s*>",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if empty:
                return empty.group(0)
        return None
