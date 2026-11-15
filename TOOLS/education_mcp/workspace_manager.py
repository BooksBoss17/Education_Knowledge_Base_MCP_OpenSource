"""Idempotent workspace bootstrap, inspect-only checks and explicit additive repair."""
from collections import Counter
from pathlib import Path, PurePosixPath
import csv
import io
import json
import os
import stat
import subprocess
import uuid

STATE_DIR = '.education-mcp'
STATE_NAME = 'workspace.json'
PROTECTED = {'tmp', 'tmp/bemarkdown'}


def linked_path(path):
    try:
        info = path.lstat()
        return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)
    except FileNotFoundError:
        return False


def default_desktop():
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as key:
                value = winreg.QueryValueEx(key, 'Desktop')[0]
                return Path(os.path.expandvars(value))
        except OSError:
            pass
    return Path.home() / 'Desktop'


def select_profile(total_mib):
    if total_mib is None:
        return dict(profile=None, recommended=False, reason='GPU_MEMORY_UNKNOWN')
    if total_mib < 6144:
        return dict(profile=None, recommended=False, reason='BELOW_6GB_NOT_RECOMMENDED')
    name = '10gb' if total_mib > 10240 else '6gb'
    return dict(profile=name, recommended=True, budget_mib=10240 if name == '10gb' else 6144,
                reason='DETECTED_DEDICATED_GPU_MEMORY')


def detect_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
        rows = [dict(index=int(r[0]), uuid=r[1].strip(), name=r[2].strip(), total_mib=int(float(r[3])))
                for r in csv.reader(io.StringIO(result.stdout)) if len(r) == 4]
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is not None:
            first = visible.split(',')[0].strip()
            rows = [r for r in rows if str(r['index']) == first or r['uuid'] == first]
        device = rows[0] if rows else None
        return dict(device=device, **select_profile(device['total_mib'] if device else None))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return dict(device=None, **select_profile(None), detection_error=type(exc).__name__)


def gpu_used_mib(device):
    if not device:
        return None
    result = subprocess.run(['nvidia-smi', '--id=' + str(device.get('uuid') or device.get('index', 0)),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=3, check=True)
    return int(result.stdout.strip())


def directory_paths(values):
    if not isinstance(values, list) or len(values) > 1000:
        raise ValueError('Expected at most 1000 relative directory names')
    paths = set(PROTECTED)
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    for value in values:
        if not isinstance(value, str) or not value or len(value) > 240:
            raise ValueError('Invalid relative directory name')
        value = value.replace('\\', '/')
        parts = value.split('/')
        if any(not p or p in {'.', '..'} or p.rstrip(' .') != p or any(c in p for c in ':*?"<>|')
               or any(ord(c) < 32 for c in p) or p.split('.')[0].upper() in reserved for p in parts):
            raise ValueError('Directory must be a safe relative path')
        if parts[0].casefold() == STATE_DIR:
            raise ValueError('Workspace metadata is reserved')
        for i in range(1, len(parts) + 1):
            paths.add('/'.join(parts[:i]))
    folded = {}
    for p in paths:
        if p.casefold() in folded and folded[p.casefold()] != p:
            raise ValueError('Case-colliding directory names')
        folded[p.casefold()] = p
    return sorted(paths, key=lambda p: (p.count('/'), p))


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class WorkspaceManager:
    def __init__(self, root=None, *, template=None, gpu_detector=detect_gpu):
        raw = Path(root) if root is not None else default_desktop() / 'Education_Knowledge_Base'
        # Refuse redirected roots rather than adopting a different workspace silently.
        if linked_path(raw):
            raise ValueError('Workspace root cannot be a link or junction')
        self.root = raw.absolute()
        self.template = Path(template) if template else Path(__file__).with_name('workspace_template.json')
        self.gpu_detector = gpu_detector
        self.state_path = self.root / STATE_DIR / STATE_NAME

    def safe_path(self, relative):
        relative = relative.replace('\\', '/')
        if linked_path(self.root):
            raise ValueError('Workspace root cannot be a link or junction')
        path = self.root
        for part in PurePosixPath(relative).parts:
            path = path / part
            if linked_path(path):
                raise ValueError('Linked workspace path cannot be followed: ' + relative)
            if path.exists() and not path.is_dir() and path != self.state_path:
                raise ValueError('Directory blocked by file: ' + relative)
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Path escapes workspace')
        return path

    def bootstrap(self):
        existed = self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = self.safe_path(STATE_DIR)
        metadata.mkdir(exist_ok=True)
        lock = metadata / 'bootstrap.lock'
        if linked_path(lock):
            raise ValueError('Linked bootstrap lock')
        with lock.open('a+b') as handle:
            handle.seek(0, 2)
            if not handle.tell():
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                if linked_path(self.state_path):
                    raise ValueError('Linked workspace state')
                if self.state_path.exists():
                    return self.inspect()
                template = json.loads(self.template.read_text(encoding='utf-8'))
                directories = directory_paths(template['directories'])
                created = []
                if not existed:
                    for relative in directories:
                        path = self.safe_path(relative)
                        if not path.exists():
                            path.mkdir(); created.append(relative)
                state = dict(schema='education-workspace-v1', mode='CHECK', directories=directories,
                             gpu=self.gpu_detector(), adopted_existing=existed, initially_created=created)
                atomic_json(self.state_path, state)
                return {**self.inspect(), 'first_run': True, 'created': created}
            finally:
                handle.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def state(self):
        self.safe_path(STATE_DIR)
        if linked_path(self.state_path):
            raise ValueError('Linked workspace state')
        state = json.loads(self.state_path.read_text(encoding='utf-8'))
        if state.get('schema') != 'education-workspace-v1':
            raise ValueError('Unsupported workspace state')
        state['directories'] = directory_paths(state['directories'])
        return state

    def inspect(self):
        state = self.state()
        missing, conflicts = [], []
        for relative in state['directories']:
            try:
                if not self.safe_path(relative).is_dir():
                    missing.append(relative)
            except ValueError as exc:
                conflicts.append(dict(path=relative, reason=str(exc)))
        return dict(mode='CHECK', workspace=str(self.root), complete=not missing and not conflicts,
                    missing_directories=missing, conflicts=conflicts, gpu=state['gpu'],
                    repair_policy='Only explicitly requested directories are created; never delete or move content',
                    default_directories=state['directories'])

    def repair(self, directories):
        requested = directory_paths(directories)
        # Protected paths are template invariants, not implicit repair requests.
        explicit = set()
        for value in directories:
            parts = value.replace('\\', '/').split('/')
            explicit.update('/'.join(parts[:i]) for i in range(1, len(parts)+1))
        requested = [p for p in requested if p in explicit]
        allowed = set(self.state()['directories'])
        if not set(requested) <= allowed:
            raise ValueError('Repair paths must belong to the configured framework')
        targets = [(p, self.safe_path(p)) for p in requested]
        created = []
        for relative, path in targets:
            if not path.exists():
                path.mkdir(); created.append(relative)
        return {**self.inspect(), 'created': created}

    def configure(self, directories):
        state = self.state()
        state['directories'] = directory_paths(directories)
        atomic_json(self.state_path, state)
        return {**self.inspect(), 'framework_updated': True, 'existing_content_unchanged': True}

    def inventory(self, prefix='', offset=0, limit=200):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Invalid inventory pagination')
        if prefix:
            directory_paths([prefix])
        root = self.safe_path(prefix)
        if not root.is_dir():
            raise ValueError('Inventory directory does not exist')
        files, skipped, errors, folders = [], [], [], 0
        def onerror(exc):
            errors.append(dict(path=str(exc.filename), error=type(exc).__name__))
        for current, dirs, names in os.walk(root, followlinks=False, onerror=onerror):
            kept = []
            for name in dirs:
                p = Path(current) / name
                if linked_path(p):
                    skipped.append(p.relative_to(self.root).as_posix())
                else:
                    kept.append(name); folders += 1
            dirs[:] = sorted(kept)
            for name in sorted(names):
                p = Path(current) / name
                if linked_path(p):
                    skipped.append(p.relative_to(self.root).as_posix()); continue
                files.append(dict(name=name, path=p.relative_to(self.root).as_posix(),
                                  type=p.suffix.lower() or '[no extension]'))
        files.sort(key=lambda r: r['path'])
        return dict(workspace=str(self.root), prefix=prefix, total_files=len(files), total_directories=folders,
                    counts_by_type=dict(sorted(Counter(r['type'] for r in files).items())),
                    files=files[offset:offset+limit], offset=offset,
                    next_offset=offset+limit if offset+limit < len(files) else None,
                    skipped_links=skipped, errors=errors, complete=not errors and not skipped,
                    scope='All regular files under selected folder, including temporary and metadata files; no content read')

    def call(self, action='inspect', directories=None, prefix='', offset=0, limit=200):
        if action == 'inspect': return self.inspect()
        if action == 'inventory': return self.inventory(prefix, offset, limit)
        if action == 'repair': return self.repair(directories)
        if action == 'configure': return self.configure(directories)
        raise ValueError('Unknown workspace action')

    def sample_gpu_usage(self):
        return gpu_used_mib(self.state()['gpu'].get('device'))
