"""Run with: python -m unittest discover -s tests -p test_downloads.py"""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import downloads

class DownloadTests(unittest.TestCase):
    def test_reject_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            for relative in ('../escape', 'C:/escape', '/escape'):
                with self.assertRaises(ValueError):
                    downloads.safe_path(directory, relative)

    def test_existing_cache_is_hash_verified(self):
        data = b'public fixture'
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / digest
            path.write_bytes(data)
            with patch.object(downloads.urllib.request, 'urlopen', side_effect=AssertionError('network must not run')):
                self.assertEqual(downloads.download('https://example.org/file', digest, len(data), directory), path.resolve())

    def test_bad_bytes_never_become_accepted_asset(self):
        digest = hashlib.sha256(b'correct').hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(downloads.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(b'incorrect')):
                with patch.object(downloads.time, 'sleep'):
                    with self.assertRaises(ValueError):
                        downloads.download('https://example.org/file', digest, len(b'correct'), directory)
            self.assertFalse((Path(directory) / digest).exists())

    def test_plain_http_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                downloads.download('http://example.org/file', '0'*64, 1, directory)

if __name__ == '__main__':
    unittest.main()
