"""Pinned inference-only readonly safetensors adapter for Windows Job limits.

Preserves checkpoint dtype/shape/value and all model/device/offload arguments.
Only the known Transformers loader alias is replaced, for the load scope.
Do not mutate returned readonly-backed CPU tensors in-place.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib,json,math,mmap
from pathlib import Path
import struct,warnings

EXPECTED_MODELING_UTILS_SHA256='a7392c26dd2f005383cc3c1f8562638a6039d393ca03fb1933513479a6d264e4'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

class ReadOnlySlice:
    def __init__(self,owner,name):self.owner,self.name=owner,name
    def get_shape(self):return self.owner.header[self.name]['shape']
    def get_dtype(self):return self.owner.header[self.name]['dtype']
    def __getitem__(self,key):return self.owner.get_tensor(self.name)[key]

class ReadOnlySafeOpen:
    def __init__(self,filename,framework='pt',device='cpu',backend=None):
        if framework not in ('pt','pytorch') or str(device)!='cpu' or backend not in (None,'mmap','pread'):
            raise ValueError('READONLY_SAFETENSORS_REQUIRES_PINNED_CPU_PYTORCH_LOADER')
        self.path=Path(filename);self.file=None;self.mapping=None
        try:
            self.file=self.path.open('rb');size=self.path.stat().st_size
            raw=self.file.read(8)
            if len(raw)!=8:raise ValueError('SAFETENSORS_HEADER_TRUNCATED')
            n=struct.unpack('<Q',raw)[0]
            if n>16*1024**2 or n>size-8:raise ValueError('SAFETENSORS_HEADER_SIZE_INVALID')
            self.header=json.loads(self.file.read(n));self.start=8+n
            for name,info in self.header.items():
                if name=='__metadata__':continue
                a,b=info['data_offsets']
                if not 0<=a<=b<=size-self.start:raise ValueError('SAFETENSORS_DATA_OFFSETS_INVALID')
                if any(type(v) is not int or v<0 for v in info['shape']):raise ValueError('SAFETENSORS_SHAPE_INVALID')
            self.mapping=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        except BaseException:
            self.close();raise
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        if self.file:self.file.close();self.file=None
        if self.mapping:
            try:self.mapping.close()
            except BufferError:
                # A tensor's Python buffer retains the mmap until that tensor
                # dies. Closing its original file does not invalidate the view.
                return
            self.mapping=None
    def keys(self):return sorted(k for k in self.header if k!='__metadata__')
    def metadata(self):return self.header.get('__metadata__')
    def get_slice(self,name):return ReadOnlySlice(self,name)
    def get_tensor(self,name):
        import torch
        info=self.header[name]
        dtype={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32,'F64':torch.float64,
            'I64':torch.int64,'I32':torch.int32,'I16':torch.int16,'I8':torch.int8,'U8':torch.uint8,'BOOL':torch.bool}[info['dtype']]
        a,b=info['data_offsets'];count=math.prod(info['shape'])
        if b-a!=count*torch.empty((),dtype=dtype).element_size():raise ValueError('SAFETENSORS_BYTES_SHAPE_MISMATCH')
        if count==0:return torch.empty(info['shape'],dtype=dtype)
        view=memoryview(self.mapping)[self.start+a:self.start+b]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='.*buffer is not writable.*')
            value=torch.frombuffer(view,dtype=dtype,count=count)
        return value.reshape(info['shape'])

@contextmanager
def readonly_safe_open_scope(modeling_module,*,expected_sha256=EXPECTED_MODELING_UTILS_SHA256):
    actual=sha(modeling_module.__file__)
    if actual!=expected_sha256:raise RuntimeError('UNQUALIFIED_TRANSFORMERS_LOADER:'+actual)
    original=modeling_module.safe_open
    modeling_module.safe_open=ReadOnlySafeOpen
    receipt=dict(adapter_sha256=sha(__file__),modeling_utils_sha256=actual,
        checkpoint_dtype_unchanged=True,device_map_unchanged=True,allocator_warmup_unchanged=True,
        mapping_access='ACCESS_READ',global_safetensors_module_changed=False)
    try:yield receipt
    finally:modeling_module.safe_open=original
