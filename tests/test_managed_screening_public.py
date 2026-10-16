"""Public activation and resource/classifier wiring without model inference."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

class PublicManagedContract(unittest.TestCase):
    def test_registered_public_runtime_is_used_by_queue(self):
        bootstrap=module('public_queue',ROOT/'TOOLS/runtime_manager/bootstrap.py')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);selected=root/'custom-python.exe';selected.write_bytes(b'fixture')
            payload={
                'TOOLS/bemarkdown/TOOL_MANIFEST.json':{'wheel':{'sha256':'a'*64}},
                '.local/runtime-selection.json':{'status':'READY_FULL','wheel_sha256':'a'*64,'python':str(selected)},
                'TOOLS/bemarkdown_optimized/HOST_MEMORY_POLICY.json':{'mode':'auto'},
            }
            for name,value in payload.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
            for name in ['TOOLS/bemarkdown_optimized/core/bemarkdown/host_memory_policy.py','TOOLS/runtime_manager/native_worker.py']:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n',encoding='utf-8')
            target=root/'TOOLS/education_mcp/runtime_selection.py';target.parent.mkdir(parents=True)
            shutil.copyfile(ROOT/'TOOLS/education_mcp/runtime_selection.py',target)
            with patch.object(bootstrap,'runtime',return_value=(root,root/'TOOLS/runtime_manager')),patch.dict(os.environ,{'LOCALAPPDATA':str(root/'local')}):
                _,programs=bootstrap.registry(root,Path('unused.exe'))
            self.assertEqual(Path(programs['native-convert']['python']),selected.resolve())
            self.assertNotIn('semantic-retention',programs)

    def test_semantic_override_is_honored_without_default_install(self):
        bootstrap=module('public_semantic',ROOT/'TOOLS/semantic_retention/bootstrap.py')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);tool=root/'TOOLS/semantic_retention';tool.mkdir(parents=True)
            (tool/'SCHEME.json').write_text('{"runtime_id":"unused"}',encoding='utf-8')
            receipt=root/'custom.json';value={'status':'VERIFIED_HOST_RUNTIME','no_developer_runtime_dependency':True,'source_python':'selected'}
            receipt.write_text(json.dumps(value),encoding='utf-8')
            with patch.dict(os.environ,{'LOCALAPPDATA':str(root/'local'),'SEMANTIC_RETENTION_RUNTIME_CONFIG':str(receipt)}):
                self.assertEqual(bootstrap.runtime_config(tool),value)

    def test_installed_native_inventory_verifies(self):
        gateway=module('public_native',ROOT/'TOOLS/bemarkdown_optimized/convert.py')
        self.assertTrue(gateway.verified_payload(ROOT/'TOOLS/bemarkdown_optimized')['image_screening']['default'])

    def test_managed_entrypoints_compile_on_python_311(self):
        for part in ['runtime_manager','education_mcp','bemarkdown_optimized/core']:
            for path in (ROOT/'TOOLS'/part).rglob('*.py'):
                ast.parse(path.read_text(encoding='utf-8'),filename=str(path))

    def test_private_screening_templates_are_not_distributed(self):
        value=json.loads((ROOT/'TOOLS/education_mcp/screening_templates.json').read_text(encoding='utf-8'))
        self.assertEqual(value['status'],'NOT_CONFIGURED_PUBLIC')
        self.assertEqual(value['templates'],[])

    def test_resource_forecasts_bind_public_model_metadata(self):
        rows=json.loads((ROOT/'MODELS/MODEL_REGISTRY.json').read_text(encoding='utf-8'))['models']
        registry={row['model_id']:row for row in rows}
        def check(value):
            if isinstance(value,dict):
                if {'model_id','model_fingerprint','manifest_sha256'}<=value.keys():
                    row=registry[value['model_id']]
                    self.assertEqual(value['model_fingerprint'],row['model_fingerprint'])
                    self.assertEqual(value['manifest_sha256'],row['manifest_sha256'])
                for child in value.values():check(child)
            elif isinstance(value,list):
                for child in value:check(child)
        for path in (ROOT/'TOOLS/bemarkdown_optimized').rglob('RESOURCE_FORECASTS_6GB.json'):
            check(json.loads(path.read_text(encoding='utf-8')))

if __name__=='__main__':unittest.main()
