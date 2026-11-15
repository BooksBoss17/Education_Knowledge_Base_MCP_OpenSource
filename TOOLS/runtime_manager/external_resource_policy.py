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
    cap=min(8*GiB,total_physical_bytes//2,commit_limit_bytes//2)
    p={'schema':'public-external-api-ram-v1','project_ram_cap_gib':cap/GiB,
       'ram_reserve':min(2*GiB,total_physical_bytes//8),'commit_reserve':min(2*GiB,commit_limit_bytes//8),
       'gpu_uuid':'EXTERNAL_API','gpu_total_bytes':0,'mode':'external','supported':True,
       'total_physical_bytes':total_physical_bytes,'commit_limit_bytes':commit_limit_bytes}
    p['policy_fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    return p


def evaluate_admission(p,o,r,reservations=None):
    reasons=[]
    if r['extra_gpu_mib']:return {'decision':'CAPACITY_INSUFFICIENT','reasons':['EXTERNAL_API_CANNOT_OWN_LOCAL_GPU']}
    if r['extra_private_bytes']>p['project_ram_cap_gib']*GiB:
        return {'decision':'CAPACITY_INSUFFICIENT','reasons':['CLIENT_RAM_REQUIREMENT_TOO_LARGE']}
    if o['physical_free_bytes']-r['extra_physical_bytes']<p['ram_reserve']:reasons.append('PHYSICAL_RESERVE')
    if o['commit_headroom_bytes']-r['extra_private_bytes']<p['commit_reserve']:reasons.append('COMMIT_RESERVE')
    return {'decision':'WAIT' if reasons else 'ADMIT','reasons':reasons,'policy_fingerprint':p['policy_fingerprint']}


def check_current(p,o,raise_on_violation=False):
    reasons=[]
    if o.get('group_current_bytes',0)>p['project_ram_cap_gib']*GiB:reasons.append('CLIENT_RAM_CAP')
    if o['physical_free_bytes']<p['ram_reserve']:reasons.append('PHYSICAL_RESERVE')
    if o['commit_headroom_bytes']<p['commit_reserve']:reasons.append('COMMIT_RESERVE')
    return {'allowed':not reasons,'decision':'STOP' if reasons else 'CONTINUE','reasons':reasons,'policy_fingerprint':p['policy_fingerprint']}
