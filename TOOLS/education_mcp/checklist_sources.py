"""Content-addressed inventory of explicitly selected checklist inputs."""

from hashlib import file_digest
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from zipfile import BadZipFile, ZipFile


FORMATS = {".pdf", ".docx", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def digest(stream):
    return file_digest(stream, "sha256").hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return digest(stream)


def authorized(path, roots):
    requested = Path(path)
    if not requested.is_absolute():
        raise ValueError("Use absolute input paths")
    resolved = requested.resolve(strict=True)
    if not any(resolved.is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError("Source outside allowed roots")
    return resolved


def input_files(path, skipped):
    try:
        if path.resolve(strict=True) != path:
            skipped.append({"path": str(path), "reason": "Linked input skipped"})
        elif path.is_dir():
            for item in sorted(path.iterdir()):
                yield from input_files(item, skipped)
        elif path.is_file():
            yield path
    except OSError as exc:
        skipped.append({"path": str(path), "reason": str(exc)})


def inventory(paths, roots, encoding=None):
    """Report all matching provenance for each distinct set of bytes."""
    groups = {}
    skipped = []
    visited = set()
    for request in paths:
        try:
            path = authorized(request, roots)
        except OSError as exc:
            skipped.append({"path": request, "reason": str(exc)})
            continue
        for path in input_files(path, skipped):
            if path in visited:
                continue
            visited.add(path)
            try:
                if path.suffix.lower() in FORMATS:
                    groups.setdefault(file_sha(path), []).append({"path": str(path)})
                elif path.suffix.lower() == ".zip":
                    archive_sha = file_sha(path)
                    with ZipFile(path, metadata_encoding=encoding) as archive:
                        for index, member in enumerate(archive.infolist()):
                            if member.is_dir():
                                continue
                            origin = {"path": str(path), "archive_sha256": archive_sha,
                                      "member": member.filename, "index": index}
                            if encoding:
                                origin["zip_encoding"] = encoding
                            if PurePosixPath(member.filename).suffix.lower() not in FORMATS:
                                skipped.append({**origin, "reason": "Not a convertible member"})
                                continue
                            try:
                                with archive.open(member) as stream:
                                    groups.setdefault(digest(stream), []).append(origin)
                            except (OSError, RuntimeError, ValueError, BadZipFile) as exc:
                                skipped.append({**origin, "reason": str(exc)})
                else:
                    skipped.append({"path": str(path), "reason": "Unsupported format"})
            except (OSError, RuntimeError, ValueError, LookupError, BadZipFile) as exc:
                skipped.append({"path": str(path), "reason": str(exc)})
    return {"unique": [{"sha256": key, "sources": sources} for key, sources in sorted(groups.items())],
            "skipped": skipped}


@contextmanager
def archive_member(roots, origin):
    path = authorized(origin["path"], roots)
    if file_sha(path) != origin["archive_sha256"]:
        raise ValueError("Archive changed since inventory")
    with ZipFile(path, metadata_encoding=origin.get("zip_encoding")) as archive:
        index = origin["index"]
        if type(index) is not int or not 0 <= index < len(archive.infolist()):
            raise ValueError("Unknown archive member")
        member = archive.infolist()[index]
        if member.is_dir() or member.filename != origin["member"]:
            raise ValueError("Archive member changed")
        suffix = PurePosixPath(member.filename).suffix.lower()
        if suffix not in FORMATS:
            raise ValueError("Unsupported archive member")
        with archive.open(member) as stream:
            yield stream, suffix


def verify_member(roots, origin, sha256):
    with archive_member(roots, origin) as (stream, _):
        if digest(stream) != sha256:
            raise ValueError("Archive member fingerprint mismatch")


def stage(ws, roots, origin, sha256):
    """Write the selected member to a derived path, without extracting ZIP paths."""
    with archive_member(roots, origin) as (source, suffix):
        base = (ws / "tmp/checklist-import/inputs").resolve()
        if not base.is_relative_to(ws):
            raise ValueError("Staging outside workspace")
        base.mkdir(parents=True, exist_ok=True)
        target = base / (sha256 + suffix)
        if target.resolve() != target:
            raise ValueError("Staging alias is not a regular path")
        if target.exists():
            if file_sha(target) != sha256:
                raise ValueError("Staged member fingerprint mismatch")
        else:
            with tempfile.TemporaryDirectory(dir=base) as folder:
                staged = Path(folder) / "input"
                with staged.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if file_sha(staged) != sha256:
                    raise ValueError("Archive member fingerprint mismatch")
                staged.replace(target)
    return {"source": str(target), "sha256": sha256}
