from __future__ import annotations

import posixpath
import re
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from lxml import etree
from PIL import Image, UnidentifiedImageError

from .namespaces import CONTENT_TYPES_NS, REL_NS

MIB = 1024 * 1024


class InvalidDocxError(ValueError):
    """The source is not a structurally valid, safe DOCX package."""


class ResourceLimitError(InvalidDocxError):
    """A DOCX exceeds an explicitly configured production resource limit."""


@dataclass(frozen=True)
class DocxResourceLimits:
    """Production defaults derived from the frozen 169-DOCX corpus plus headroom."""

    max_archive_bytes: int = 512 * MIB
    max_part_count: int = 10_000
    max_total_uncompressed_bytes: int = 512 * MIB
    max_single_xml_bytes: int = 32 * MIB
    max_single_media_bytes: int = 128 * MIB
    max_single_ole_bytes: int = 64 * MIB
    max_single_other_bytes: int = 128 * MIB
    max_compression_ratio: float = 200.0
    max_image_width: int = 50_000
    max_image_height: int = 50_000
    max_image_pixels: int = 250_000_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"DOCX resource limit {name} must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class DocxResourceProfile:
    archive_bytes: int
    part_count: int
    total_uncompressed_bytes: int
    largest_xml_bytes: int
    largest_media_bytes: int
    largest_ole_bytes: int
    largest_other_bytes: int
    maximum_compression_ratio: float
    largest_image_width: int = 0
    largest_image_height: int = 0
    largest_image_pixels: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class Relationship:
    relationship_id: str
    relationship_type: str
    target: str
    target_part: str | None
    external: bool


def _safe_xml(data: bytes, name: str) -> etree._Element:
    if re.search(br"<!DOCTYPE\b", data, flags=re.IGNORECASE):
        raise InvalidDocxError(f"DTD is forbidden in DOCX XML part: {name}")
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        recover=False,
        huge_tree=False,
    )
    try:
        return etree.fromstring(data, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise InvalidDocxError(f"Invalid XML in DOCX part {name}: {exc}") from exc


def _validate_member_name(name: str) -> None:
    if not name or "\\" in name:
        raise InvalidDocxError(f"unsafe ZIP member path: {name!r}")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or name.startswith("//")
        or any(part in {"", ".", ".."} for part in path.parts)
        or re.match(r"^[A-Za-z]:", name)
    ):
        raise InvalidDocxError(f"unsafe ZIP member path: {name!r}")


def _member_kind(name: str) -> str:
    lower = name.lower()
    if lower == "[content_types].xml" or lower.endswith((".xml", ".rels")):
        return "xml"
    if lower.startswith("word/media/"):
        return "media"
    if lower.startswith("word/embeddings/") or lower.endswith(".bin"):
        return "ole"
    return "other"


def validate_docx_source(
    source: str | Path, *, limits: DocxResourceLimits | None = None
) -> DocxResourceProfile:
    source = Path(source)
    limits = limits or DocxResourceLimits()
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file():
        raise InvalidDocxError(f"DOCX source is not a regular file: {source}")
    if source.suffix.lower() != ".docx":
        raise InvalidDocxError(f"Unsupported source type (expected DOCX): {source}")
    archive_bytes = source.stat().st_size
    if archive_bytes > limits.max_archive_bytes:
        raise ResourceLimitError(
            f"DOCX archive size {archive_bytes} exceeds {limits.max_archive_bytes}"
        )
    with source.open("rb") as handle:
        signature = handle.read(4)
    if signature != b"PK\x03\x04":
        raise InvalidDocxError("DOCX does not have a ZIP local-file signature")

    try:
        with zipfile.ZipFile(source, "r") as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise InvalidDocxError("DOCX contains duplicate ZIP member names")
            for name in names:
                _validate_member_name(name)
            required = {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
            missing = sorted(required.difference(names))
            if missing:
                raise InvalidDocxError(
                    "DOCX is missing required package part(s): " + ", ".join(missing)
                )
            if len(infos) > limits.max_part_count:
                raise ResourceLimitError(
                    f"DOCX part count {len(infos)} exceeds {limits.max_part_count}"
                )

            largest = {"xml": 0, "media": 0, "ole": 0, "other": 0}
            total_uncompressed = 0
            maximum_ratio = 1.0
            single_limits = {
                "xml": limits.max_single_xml_bytes,
                "media": limits.max_single_media_bytes,
                "ole": limits.max_single_ole_bytes,
                "other": limits.max_single_other_bytes,
            }
            for info in infos:
                kind = _member_kind(info.filename)
                total_uncompressed += info.file_size
                largest[kind] = max(largest[kind], info.file_size)
                if info.file_size > single_limits[kind]:
                    raise ResourceLimitError(
                        f"DOCX {kind} part {info.filename!r} is {info.file_size} bytes; "
                        f"limit is {single_limits[kind]}"
                    )
                ratio = (
                    info.file_size / info.compress_size
                    if info.compress_size
                    else (float("inf") if info.file_size else 1.0)
                )
                maximum_ratio = max(maximum_ratio, ratio)
                if ratio > limits.max_compression_ratio:
                    raise ResourceLimitError(
                        f"DOCX compression ratio {ratio:.3f} for {info.filename!r} "
                        f"exceeds {limits.max_compression_ratio:.3f}"
                    )
            if total_uncompressed > limits.max_total_uncompressed_bytes:
                raise ResourceLimitError(
                    "DOCX total uncompressed bytes "
                    f"{total_uncompressed} exceed {limits.max_total_uncompressed_bytes}"
                )

            content_types = _safe_xml(
                archive.read("[Content_Types].xml"), "[Content_Types].xml"
            )
            override = content_types.find(
                f"{{{CONTENT_TYPES_NS}}}Override[@PartName='/word/document.xml']"
            )
            if override is None or "wordprocessingml.document.main+xml" not in (
                override.get("ContentType", "")
            ):
                raise InvalidDocxError(
                    "[Content_Types].xml does not declare the DOCX main document part"
                )
            root_rels = _safe_xml(archive.read("_rels/.rels"), "_rels/.rels")
            office_targets = [
                node.get("Target", "")
                for node in root_rels.findall(f"{{{REL_NS}}}Relationship")
                if node.get("Type", "").endswith("/officeDocument")
                and node.get("TargetMode") != "External"
            ]
            if "word/document.xml" not in {
                posixpath.normpath(target.lstrip("/")) for target in office_targets
            }:
                raise InvalidDocxError(
                    "_rels/.rels has no internal officeDocument relationship to "
                    "word/document.xml"
                )
            document = _safe_xml(
                archive.read("word/document.xml"), "word/document.xml"
            )
            if etree.QName(document).localname != "document" or not document.xpath(
                "./*[local-name()='body']"
            ):
                raise InvalidDocxError(
                    "word/document.xml has no valid document/body root structure"
                )
    except zipfile.BadZipFile as exc:
        raise InvalidDocxError(f"DOCX ZIP is invalid or truncated: {exc}") from exc

    return DocxResourceProfile(
        archive_bytes=archive_bytes,
        part_count=len(infos),
        total_uncompressed_bytes=total_uncompressed,
        largest_xml_bytes=largest["xml"],
        largest_media_bytes=largest["media"],
        largest_ole_bytes=largest["ole"],
        largest_other_bytes=largest["other"],
        maximum_compression_ratio=maximum_ratio,
    )


class PackageIndex:
    """Read one preflighted DOCX ZIP and index safe parts and relationships."""

    def __init__(
        self, source: str | Path, *, limits: DocxResourceLimits | None = None
    ):
        started = time.perf_counter()
        self.source = Path(source)
        self.limits = limits or DocxResourceLimits()
        self.resource_profile = validate_docx_source(self.source, limits=self.limits)
        self.warnings: list[str] = []
        try:
            with zipfile.ZipFile(self.source, "r") as archive:
                self.parts = {
                    info.filename: archive.read(info)
                    for info in archive.infolist()
                    if not info.is_dir()
                }
        except zipfile.BadZipFile as exc:
            raise InvalidDocxError(f"DOCX ZIP is invalid or truncated: {exc}") from exc
        image_width, image_height, image_pixels = self._validate_image_dimensions()
        self.resource_profile = replace(
            self.resource_profile,
            largest_image_width=image_width,
            largest_image_height=image_height,
            largest_image_pixels=image_pixels,
        )
        self.xml_parts: dict[str, etree._Element] = {}
        self.relationships: dict[tuple[str, str], Relationship] = {}
        self.content_types: dict[str, str] = {}
        self._index_content_types()
        self._index_relationships()
        self.read_seconds = time.perf_counter() - started

    def _validate_image_dimensions(self) -> tuple[int, int, int]:
        largest_width = largest_height = largest_pixels = 0
        raster_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
        for name, data in self.parts.items():
            if not name.startswith("word/media/"):
                continue
            if PurePosixPath(name).suffix.lower() not in raster_suffixes:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                    with Image.open(BytesIO(data)) as image:
                        width, height = image.size
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                self.warnings.append(f"Could not inspect image dimensions for {name}: {exc}")
                continue
            pixels = width * height
            largest_width = max(largest_width, width)
            largest_height = max(largest_height, height)
            largest_pixels = max(largest_pixels, pixels)
            if (
                width > self.limits.max_image_width
                or height > self.limits.max_image_height
                or pixels > self.limits.max_image_pixels
            ):
                raise ResourceLimitError(
                    f"DOCX image dimensions exceed limits for {name!r}: "
                    f"{width}x{height} ({pixels} pixels)"
                )
        return largest_width, largest_height, largest_pixels

    def _parse_xml(self, name: str) -> etree._Element:
        if name not in self.xml_parts:
            data = self.parts.get(name)
            if data is None:
                raise KeyError(f"DOCX part not found: {name}")
            self.xml_parts[name] = _safe_xml(data, name)
        return self.xml_parts[name]

    def xml(self, name: str) -> etree._Element:
        return self._parse_xml(name)

    def data(self, name: str) -> bytes:
        return self.parts[name]

    def has_part(self, name: str) -> bool:
        return name in self.parts

    def _index_content_types(self) -> None:
        root = self._parse_xml("[Content_Types].xml")
        defaults = {
            node.get("Extension", "").lower(): node.get("ContentType", "")
            for node in root.findall(f"{{{CONTENT_TYPES_NS}}}Default")
        }
        for part in self.parts:
            override = root.find(f"{{{CONTENT_TYPES_NS}}}Override[@PartName='/{part}']")
            if override is not None:
                self.content_types[part] = override.get("ContentType", "")
            else:
                self.content_types[part] = defaults.get(
                    PurePosixPath(part).suffix[1:].lower(), ""
                )

    def _index_relationships(self) -> None:
        for rels_name in [name for name in self.parts if name.endswith(".rels")]:
            owner = self._owner_part(rels_name)
            try:
                root = self._parse_xml(rels_name)
            except InvalidDocxError as exc:
                if rels_name == "_rels/.rels":
                    raise
                self.warnings.append(str(exc))
                continue
            for node in root.findall(f"{{{REL_NS}}}Relationship"):
                rid = node.get("Id")
                target = node.get("Target", "")
                external = node.get("TargetMode") == "External"
                if not rid:
                    continue
                target_part = None
                if not external:
                    target_part = self._resolve_target(owner, target)
                    if target_part is None:
                        self.warnings.append(
                            f"Unsafe internal relationship target ignored: "
                            f"{rels_name} {rid} -> {target!r}"
                        )
                self.relationships[(owner, rid)] = Relationship(
                    relationship_id=rid,
                    relationship_type=node.get("Type", ""),
                    target=target,
                    target_part=target_part,
                    external=external,
                )

    @staticmethod
    def _owner_part(rels_name: str) -> str:
        path = PurePosixPath(rels_name)
        if path.name == ".rels" and str(path.parent) == "_rels":
            return ""
        parent = path.parent
        if parent.name != "_rels":
            return ""
        return str(parent.parent / path.name.removesuffix(".rels"))

    @staticmethod
    def _resolve_target(owner: str, target: str) -> str | None:
        if (
            not target
            or "\\" in target
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target)
            or target.startswith("//")
        ):
            return None
        if target.startswith("/"):
            resolved = posixpath.normpath(target.lstrip("/"))
        else:
            base = posixpath.dirname(owner)
            resolved = posixpath.normpath(posixpath.join(base, target))
        if resolved in {"", ".", ".."} or resolved.startswith("../"):
            return None
        try:
            _validate_member_name(resolved)
        except InvalidDocxError:
            return None
        return resolved

    def relationship(
        self, owner_part: str, relationship_id: str
    ) -> Relationship | None:
        return self.relationships.get((owner_part, relationship_id))

    def related_parts(
        self, owner_part: str, relationship_type_suffix: str
    ) -> list[str]:
        return [
            rel.target_part
            for (owner, _), rel in self.relationships.items()
            if owner == owner_part
            and rel.relationship_type.endswith(relationship_type_suffix)
            and rel.target_part is not None
        ]

    def non_body_inventory(self) -> list[dict[str, int | str]]:
        candidates: list[dict[str, Any]] = []
        for name in sorted(self.parts):
            if not (
                name.startswith(("word/header", "word/footer"))
                or name
                in {
                    "word/footnotes.xml",
                    "word/endnotes.xml",
                    "word/comments.xml",
                }
            ):
                continue
            if not name.endswith(".xml"):
                continue
            try:
                root = self._parse_xml(name)
            except InvalidDocxError:
                candidates.append({"part": name, "xml_error": 1})
                continue
            candidates.append(
                {
                    "part": name,
                    "paragraphs": len(root.xpath(".//*[local-name()='p']")),
                    "formulas": len(root.xpath(".//*[local-name()='oMath']")),
                    "images": len(
                        root.xpath(
                            ".//*[local-name()='blip' or local-name()='imagedata']"
                        )
                    ),
                    "tables": len(root.xpath(".//*[local-name()='tbl']")),
                }
            )
        return candidates
