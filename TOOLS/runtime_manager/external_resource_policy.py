"""RAM guard for Windows clients of separately managed API services, no local GPU."""
import ctypes
import hashlib
import json
import time

GiB=1024**3


def sample_device(**_):
    from bemarkdown.host_memory_policy import _Memory
    m=_Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('Memory observation failed')
    return dict(total_physical_bytes=m.total,physical_free_bytes=m.available,
        commit_limit_bytes=m.total_pagefile,commit_headroom_bytes=m.available_pagefile,
        gpu_total_bytes=0,gpu_free_bytes=0,gpu_used_bytes=0,gpu_uuid='EXTERNAL_API',gpu_index=0,gpu_count=0,
        mapping_supported=True,group_current_bytes=0,group_working_bytes=0,sample_epoch=time.time(),
        observer='WINDOWS_RAM_EXTERNAL_API_NO_GPU_SAMPLING')


def resolve_device_policy(total_physical_bytes,commit_limit_bytes,gpu_total_bytes,**_):
    p={'schema':'public-external-api-ram-v1','project_ram_cap_gib':None,
       'ram_reserve':0,'commit_reserve':0,
       'gpu_uuid':'EXTERNAL_API','gpu_total_bytes':0,'mode':'monitor','supported':True,
       'total_physical_bytes':total_physical_bytes}
    p['policy_fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    return p


def evaluate_admission(p,o,r,reservations=None):
    if r['extra_gpu_mib']:
        return {'decision':'CAPACITY_INSUFFICIENT','reasons':['EXTERNAL_API_CANNOT_OWN_LOCAL_GPU']}
    return {'decision':'ADMIT','reasons':[],'policy_fingerprint':p['policy_fingerprint']}


def check_current(p,o,raise_on_violation=False):
    return {'allowed':True,'decision':'CONTINUE','reasons':[],'policy_fingerprint':p['policy_fingerprint']}
