"""Installer regression tests; fixture doctors do not claim GPU validation."""
from pathlib import Path
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import runtime_setup as runtime

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

installer = load('tested_installer', ROOT / 'scripts/setup.py')
resolver = load('tested_runtime_selection', ROOT / 'TOOLS/education_mcp/runtime_selection.py')

class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_contract_independent_of_tool_wheel_and_changes_with_dependencies(self):
        lock = self.root / 'scripts/requirements-pypi.lock'
        lock.parent.mkdir()
        lock.write_text('Pillow==12.1.0\n', encoding='utf-8')
        first = runtime.dependency_contract(self.root)
        (self.root / 'new-tool.whl').write_bytes(b'new code')
        self.assertEqual(first, runtime.dependency_contract(self.root))
        lock.write_text('Pillow==12.2.0\n', encoding='utf-8')
        self.assertNotEqual(first['fingerprint'], runtime.dependency_contract(self.root)['fingerprint'])

    def test_donor_requires_abi_versions_and_archive_hash(self):
        contract = dict(python=[3, 11], bits=64, platform='win32', pins={'pillow': '12.1.0'}, archives={'source': 'a' * 64})
        snapshot = dict(python=[3, 11], bits=64, platform='win32', packages={
            'pillow': {'version': '12.1.0'}, 'source': {'direct_url': {'archive_info': {'hashes': {'sha256': 'a' * 64}}}}})
        self.assertTrue(runtime.matches_contract(snapshot, contract))
        for key, bad in [('python', [3, 12]), ('bits', 32), ('platform', 'linux')]:
            self.assertFalse(runtime.matches_contract(dict(snapshot, **{key: bad}), contract))
        snapshot['packages']['source']['direct_url']['archive_info']['hashes']['sha256'] = 'b' * 64
        self.assertFalse(runtime.matches_contract(snapshot, contract))

    def test_reused_donor_never_installs_or_creates_environment(self):
        donor = self.root / 'donor'
        with patch.object(runtime, 'inspect_dependencies', return_value={'ok': True}), \
             patch.object(runtime.venv, 'EnvBuilder') as builder:
            def no_install(*args, **kwargs):
                self.fail('No installation allowed into a compatible donor')
            actual, reused = runtime.ensure_dependencies(self.root, self.root, {'fingerprint': 'f' * 64}, [donor], None, no_install, self.root)
            self.assertTrue(reused)
            self.assertEqual(actual, donor.resolve())
            builder.assert_not_called()

    def test_explicit_incompatible_donor_fails_without_modification(self):
        with patch.object(runtime, 'inspect_dependencies', return_value=None), patch.object(runtime.venv, 'EnvBuilder') as builder:
            with self.assertRaises(ValueError):
                runtime.ensure_dependencies(self.root, self.root, {'fingerprint': 'f' * 64}, [], self.root / 'old', None, self.root)
            builder.assert_not_called()

    def test_fresh_dependency_install_is_separate_from_incompatible_donor(self):
        donor = self.root / 'old'; donor.mkdir()
        keep = donor / 'keep'; keep.write_text('unchanged')
        calls = []
        with patch.object(runtime, 'inspect_dependencies', side_effect=[None, None, {'validated': True}]), \
             patch.object(runtime.venv, 'EnvBuilder') as builder:
            new, reused = runtime.ensure_dependencies(self.root, self.root / 'home', {'fingerprint': 'f' * 64},
                [donor], None, lambda command, log: calls.append(command), self.root)
        self.assertFalse(reused); self.assertNotEqual(new, donor)
        self.assertEqual(keep.read_text(), 'unchanged')
        self.assertEqual(len(calls), 3)
        self.assertIn('paddlepaddle-gpu==3.2.2', calls[1])
        self.assertIn('torch==2.13.0+cu130', calls[2])
        self.assertTrue((new / 'dependency-contract.json').is_file())
        builder.assert_called_once_with(with_pip=True)

    def test_existing_target_is_preserved(self):
        target = self.root / 'old'
        target.mkdir()
        marker = target / 'keep'; marker.write_text('unchanged')
        new = runtime.unused_target(target)
        self.assertNotEqual(new, target)
        self.assertEqual(marker.read_text(), 'unchanged')

    def test_reservation_never_shares_an_in_progress_destination(self):
        first = runtime.reserve_target(self.root / 'shared')
        second = runtime.reserve_target(self.root / 'shared')
        self.assertTrue(first.is_dir() and second.is_dir())
        self.assertNotEqual(first, second)

    @unittest.skipUnless(os.name == 'nt', 'PowerShell preflight')
    def test_powershell_preflight_rejects_corrupt_and_empty_release(self):
        script = self.root / 'setup.ps1'
        script.write_bytes((ROOT / 'setup.ps1').read_bytes())
        data = self.root / 'sample.txt'; data.write_bytes(b'good')
        manifest = self.root / 'RELEASE_MANIFEST.json'
        runtime.write(manifest, {'schema': 'education-public-file-inventory-v1', 'files': [
            {'path': 'sample.txt', 'bytes': 4, 'sha256': runtime.sha(data)}]})
        command = ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(script), '-VerifyOnly']
        checked = subprocess.run(command, capture_output=True)
        self.assertEqual(checked.returncode, 0, checked.stderr.decode(errors='replace'))
        data.write_bytes(b'bad!')
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        runtime.write(manifest, {'schema': 'education-public-file-inventory-v1', 'files': []})
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_workspace_selection_preserves_legacy_custom_workspace(self):
        client = {'mcpServers': {'education': {'args': ['launch.py', '--workspace', 'custom-workspace']}}}
        self.assertEqual(runtime.select_workspace(None, {}, client), 'custom-workspace')
        self.assertEqual(runtime.select_workspace(None, {'workspace': 'stored'}, client), 'stored')
        self.assertEqual(runtime.select_workspace(self.root, {'workspace': 'stored'}, client), str(self.root.resolve()))

    def test_repeating_custom_directory_reuses_activated_suffix_directory(self):
        requested = (self.root / 'custom').resolve()
        actual = (self.root / 'custom-upgraded').resolve()
        active = dict(wheel_sha256='new', python=str(actual / 'Scripts/python.exe'), requested_runtime_root=str(requested))
        for argument in (None, requested, actual):
            self.assertEqual(runtime.choose_tool_target(argument, self.root / 'default', active, 'new'), (requested, actual))
        different = (self.root / 'explicit-new-location').resolve()
        self.assertEqual(runtime.choose_tool_target(different, self.root / 'default', active, 'new'), (different, different))
        self.assertEqual(runtime.choose_tool_target(requested, self.root / 'default', active, 'next-version'), (requested, requested))

    def test_activation_redirects_registered_legacy_override_but_not_explicit_test_runtime(self):
        tool = self.root / 'TOOLS/bemarkdown'; tool.mkdir(parents=True)
        runtime.write(tool / 'TOOL_MANIFEST.json', {'wheel': {'sha256': 'b' * 64}})
        old, new, override = [self.root / name / 'Scripts/python.exe' for name in ('old', 'new', 'explicit')]
        runtime.write(self.root / '.local/runtime-selection.json', dict(status='READY_FULL', wheel_sha256='b' * 64,
            python=str(new), legacy_runtime_pythons=[str(old)]))
        self.assertEqual(resolver.resolve_runtime(self.root), new.resolve())
        self.assertEqual(resolver.resolve_runtime(self.root, old), new.resolve())
        self.assertEqual(resolver.resolve_runtime(self.root, override), override.resolve())
        runtime.write(tool / 'TOOL_MANIFEST.json', {'wheel': {'sha256': 'c' * 64}})
        with self.assertRaisesRegex(ValueError, 'not activated'):
            resolver.resolve_runtime(self.root)
        with self.assertRaisesRegex(ValueError, 'not activated'):
            resolver.resolve_runtime(self.root, old)

    def test_preflight_runs_before_model_download_or_runtime_changes(self):
        with patch.object(installer, 'ROOT', self.root), patch.object(sys, 'argv', ['setup.py', '--accept-external-licenses']), \
             patch.object(sys, 'version_info', (3, 11)), patch.object(installer, 'verify', side_effect=ValueError('corrupt release')), \
             patch.object(installer, 'models') as models, patch.object(installer, 'ensure_dependencies') as deps:
            with self.assertRaisesRegex(ValueError, 'corrupt release'):
                installer.main()
            models.assert_not_called(); deps.assert_not_called()
            self.assertFalse((self.root / '.local').exists())

    def setup_fixture(self, doctor_ready):
        tool = self.root / 'TOOLS/bemarkdown'; tool.mkdir(parents=True)
        wheel = tool / 'test.whl'; wheel.write_bytes(b'fixture only')
        runtime.write(tool / 'TOOL_MANIFEST.json', {'wheel': {'path': 'test.whl', 'sha256': runtime.sha(wheel)}})
        local = self.root / '.local'
        old = {'status': 'READY_FULL', 'runtime': str(self.root / 'old'), 'workspace': str(self.root / 'custom'), 'models': 'conversion'}
        activation = {'status': 'READY_FULL', 'wheel_sha256': 'old', 'python': str(self.root / 'old/Scripts/python.exe')}
        runtime.write(local / 'setup-state.json', old)
        runtime.write(local / 'runtime-selection.json', activation)
        before = (local / 'runtime-selection.json').read_bytes()
        def run(command, log, capture=False):
            if 'doctor' in command:
                return json.dumps({'readiness': 'READY_FULL' if doctor_ready else 'FAILED'})
            return json.dumps({'mcp_client_configuration': {'mcpServers': {'education': {
                'command': 'candidate-python', 'args': ['launch.py', '--workspace', str(self.root / 'custom')]}}}})
        patches = [patch.object(installer, 'ROOT', self.root), patch.object(sys, 'argv', ['setup.py', '--accept-external-licenses']),
            patch.object(sys, 'version_info', (3, 11)), patch.object(installer, 'verify'), patch.object(installer, 'models'),
            patch.object(installer.shutil, 'which', return_value=sys.executable), patch.object(installer, 'dependency_contract', return_value={'fingerprint': 'f'}),
            patch.object(installer, 'ensure_dependencies', return_value=(self.root / 'deps', True)),
            patch.object(installer, 'ensure_tool', return_value=(self.root / 'candidate', False)),
            patch.object(installer, 'verify_tool', return_value=True), patch.object(installer, 'run', side_effect=run),
            patch.object(installer.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'', b''))]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches: stack.enter_context(p)
            if doctor_ready:
                installer.main()
            else:
                with self.assertRaisesRegex(RuntimeError, 'validation did not pass'): installer.main()
        return local, before, old

    def test_doctor_failure_preserves_previous_activation_and_state(self):
        local, before, old = self.setup_fixture(False)
        self.assertEqual((local / 'runtime-selection.json').read_bytes(), before)
        self.assertEqual(runtime.read(local / 'setup-state.json'), old)
        self.assertEqual(runtime.read(local / 'last-setup-attempt.json')['status'], 'FAILED')

    def test_success_activates_after_checks_and_generates_stable_client_config(self):
        local, before, old = self.setup_fixture(True)
        activation = runtime.read(local / 'runtime-selection.json')
        self.assertNotEqual((local / 'runtime-selection.json').read_bytes(), before)
        self.assertIn(str((self.root / 'old/Scripts/python.exe').resolve()), activation['legacy_runtime_pythons'])
        config = runtime.read(local / 'mcp-client.json')['mcpServers']['education']
        self.assertNotIn('--runtime-python', config['args'])
        self.assertEqual(config['command'], str(Path(sys._base_executable).resolve()))
        self.assertEqual(runtime.read(local / 'setup-state.json')['models'], 'conversion')
        self.assertEqual(runtime.read(local / 'setup-state.json')['workspace'], old['workspace'])

    @unittest.skipUnless(os.name == 'nt', 'Windows runtime layout')
    def test_real_two_wheel_upgrade_reuses_donor_and_same_wheel_without_reinstall(self):
        donor = self.root / 'deps'
        venv.EnvBuilder(with_pip=True).create(donor)
        sentinel = donor / 'keep.txt'; sentinel.write_text('retain donor')
        donor_pip = donor / 'Lib/site-packages/pip/__init__.py'
        before = runtime.sha(donor_pip)
        def wheel(version):
            path = self.root / ('bemarkdown-' + version + '-py3-none-any.whl')
            info = 'bemarkdown-' + version + '.dist-info/'
            files = {'bemarkdown/__init__.py': ('VERSION=' + repr(version) + '\n').encode(),
                'bemarkdown/static/sample.txt': b'fixture resource',
                info + 'METADATA': ('Metadata-Version: 2.1\nName: bemarkdown\nVersion: ' + version + '\n').encode(),
                info + 'WHEEL': b'Wheel-Version: 1.0\nGenerator: regression\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
            rows = [[n, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).decode().rstrip('='), str(len(b))] for n,b in files.items()]
            rows.append([info + 'RECORD', '', ''])
            out = io.StringIO(); csv.writer(out, lineterminator='\n').writerows(rows)
            files[info + 'RECORD'] = out.getvalue().encode()
            with zipfile.ZipFile(path, 'w') as z:
                for n,b in files.items(): z.writestr(n,b)
            return path
        calls = []
        def run(command, log):
            calls.append(command)
            subprocess.run([str(c) for c in command], check=True, capture_output=True, env=runtime.clean_env())
        first, second = wheel('0.0.1'), wheel('0.0.2')
        target = self.root / 'tool'
        v1, reused = runtime.ensure_tool(target, donor, first, {'fingerprint': 'fixture'}, run, self.root)
        self.assertFalse(reused)
        v2, reused = runtime.ensure_tool(target, donor, second, {'fingerprint': 'fixture'}, run, self.root)
        self.assertNotEqual(v1, v2)
        self.assertTrue(runtime.verify_tool(v1, first)); self.assertTrue(runtime.verify_tool(v2, second))
        same, reused = runtime.ensure_tool(v2, donor, second, {'fingerprint': 'fixture'}, run, self.root)
        self.assertTrue(reused); self.assertEqual(same, v2); self.assertEqual(len(calls), 2)
        self.assertEqual(runtime.sha(donor_pip), before); self.assertEqual(sentinel.read_text(), 'retain donor')

if __name__ == '__main__':
    unittest.main()
