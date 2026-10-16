"""Pinned-loader adapter: readonly Windows mappings avoid whole-shard COW commit.

HF5.5 materializes each get_slice()[...] then copies it to the target device.
This adapter never writes checkpoint bytes and never maps with ACCESS_COPY.
"""
import json
import math
import mmap
import pathlib
import struct
import warnings

class ReadOnlySlice:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name
    def get_shape(self):
        return self.owner.header[self.name]['shape']
    def get_dtype(self):
        return self.owner.header[self.name]['dtype']
    def __getitem__(self, key):
        return self.owner.get_tensor(self.name)[key]

class ReadOnlySafeOpen:
    def __init__(self, filename, framework='pt', device='cpu', backend=None):
        assert framework in ('pt', 'pytorch') and str(device) == 'cpu'
        assert backend in (None, 'mmap', 'pread'), 'UNSUPPORTED_SAFETENSORS_BACKEND'
        self.path = pathlib.Path(filename)
        self.file = self.path.open('rb')
        n = struct.unpack('<Q', self.file.read(8))[0]
        assert n <= 16 * 1024 * 1024, 'UNREASONABLE_SAFETENSORS_HEADER'
        self.header = json.loads(self.file.read(n))
        self.start = 8 + n
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        size = self.path.stat().st_size
        for name, meta in self.header.items():
            if name == '__metadata__':
                continue
            a,b = meta['data_offsets']
            assert 0 <= a <= b <= size - self.start
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.file.close()
        try:
            self.mapping.close()
        except BufferError:
            # A returned tensor still owns a readonly buffer view. Its lifetime owns
            # the mmap; there is no committed whole-shard writable copy to release.
            pass
    def keys(self):
        return [k for k in self.header if k != '__metadata__']
    def metadata(self):
        return self.header.get('__metadata__')
    def get_slice(self, name):
        assert name in self.header and name != '__metadata__'
        return ReadOnlySlice(self, name)
    def get_tensor(self, name):
        import torch
        info = self.header[name]
        dtype = {'BF16':torch.bfloat16, 'F16':torch.float16, 'F32':torch.float32, 'F64':torch.float64,
                 'I64':torch.int64, 'I32':torch.int32, 'I16':torch.int16, 'I8':torch.int8,
                 'U8':torch.uint8, 'BOOL':torch.bool}[info['dtype']]
        count = math.prod(info['shape'])
        a,b = info['data_offsets']
        assert b-a == count * torch.empty((),dtype=dtype).element_size(), 'TENSOR_BYTES_SHAPE_MISMATCH'
        if count == 0:
            return torch.empty(info['shape'],dtype=dtype)
        view = memoryview(self.mapping)[self.start+a:self.start+b]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='.*buffer is not writable.*')
            tensor = torch.frombuffer(view,dtype=dtype,count=count)
        return tensor.reshape(info['shape'])
