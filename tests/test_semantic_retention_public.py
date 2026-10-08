import importlib.util,io,json,os,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('semantic_public_install',ROOT/'scripts/install_semantic_retention.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class InstallContract(unittest.TestCase):
    def test_archive_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);z=p/'bad.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('../outside.dll',b'bad')
            with self.assertRaises(ValueError):module.extract_native(z,p/'target')
            self.assertFalse((p/'outside.dll').exists())

    def test_wrong_runtime_versions_refused_before_download(self):
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/'python.exe';fake.write_bytes(b'fixture')
            with patch.object(module,'query',return_value={'version':'0.0','versions':{}}),patch.object(module,'download',side_effect=AssertionError('must not download')):
                with self.assertRaisesRegex(ValueError,'pinned Python'):module.register_runtime(fake,fake,None,Path(d),True)

    def test_model_verify_only_never_downloads(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scripts=root/'scripts';scripts.mkdir();models=root/'MODELS';models.mkdir()
            authority=models/'manifest.json';authority.write_text('{}',encoding='utf-8')
            plan={'models':[{'model_id':'qwen35-9b-semantic-reviewer','directory':'reviewer','manifest_path':'manifest.json',
                'manifest_sha256':module.sha256(authority),'files':[{'path':'weights.gguf','sha256':'0'*64,'bytes':3}]}]}
            (scripts/'model-downloads.json').write_text(json.dumps(plan),encoding='utf-8')
            (scripts/'retention-model-weights.json').write_text('{"files":[]}',encoding='utf-8')
            with patch.object(module,'ROOT',root),patch.object(module,'download',side_effect=AssertionError('must not download')):
                with self.assertRaisesRegex(ValueError,'Missing or mismatched'):module.install_models(root/'cache',verify_only=True)

    def test_source_mirror_matches_archive(self):
        tool=ROOT/'TOOLS/semantic_retention'
        with zipfile.ZipFile(tool/'dist/engine.zip') as z:
            for name in z.namelist():self.assertEqual(z.read(name),(ROOT/'SOURCE/semantic_retention_engine'/name).read_bytes())

    def test_core_adapter_is_not_auto_connected(self):
        self.assertFalse(json.loads((ROOT/'TOOLS/semantic_retention/SCHEME.json').read_text(encoding='utf-8'))['bemarkdown_auto_integrated'])
        text=(ROOT/'scripts/setup.py').read_text(encoding='utf-8')
        self.assertIn("model.get('distribution') == 'standalone-semantic'",text)

if __name__=='__main__':unittest.main()
