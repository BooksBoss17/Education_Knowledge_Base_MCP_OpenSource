"""Standalone stdlib device policy; import this file without importing bemarkdown.

RAM commit, physical availability, and whole-card VRAM are distinct quantities.
Reservations represent ONLY not-yet-started allocations. Running tasks are in
live observations and must not also appear in reservations.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import time

GiB=1024**3
MiB=1024**2
ENV='BEMARKDOWN_HOST_MEMORY_POLICY_JSON'
SCHEMA='bemarkdown-device-resource-policy-v2'

class ResourcePolicyError(MemoryError):
    def __init__(self,code,*,decision='WAIT',details=None):
        super().__init__(code);self.code=code;self.decision=decision;self.details=details

def _integer(value,name,positive=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (positive and value<=0):
        raise ValueError('RESOURCE_OBSERVATION_INVALID:'+name)
    return int(value)

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def resolve_policy(total_bytes,mode='auto'):
    """Compatible RAM-only resolver. No sampling, environment mutation or imports."""
    total_bytes=_integer(total_bytes,'total_physical_bytes',True)
    installed=max(1,round(total_bytes/GiB))
    if mode=='legacy6':cap,physical,commit=6.,8.,9.
    elif mode=='auto':
        cap=12. if installed>=24 else 8. if installed>=16 else 6. if installed>=8 else max(1.,installed/2)
        physical=min(8.,max(2.,installed/4))
        commit=min(9.,max(3.,min(cap,installed/2)*.75))
    else:raise ValueError('HOST_RAM_MODE_MUST_BE_AUTO_OR_LEGACY6')
    return dict(schema=SCHEMA,mode=mode,total_physical_bytes=total_bytes,project_ram_cap_gib=cap,
        physical_free_floor_gib=physical,commit_headroom_floor_gib=commit,gpu_policy_changed=mode=='auto')

def resolve_device_policy(total_physical_bytes,commit_limit_bytes,gpu_total_bytes,*,gpu_uuid,gpu_index=0,gpu_count=1,mode='auto',mapping_supported=True):
    p=resolve_policy(total_physical_bytes,mode)
    commit=_integer(commit_limit_bytes,'commit_limit_bytes',True)
    total=_integer(gpu_total_bytes,'gpu_total_bytes',True)
    if not isinstance(gpu_uuid,str) or not gpu_uuid:raise ValueError('GPU_IDENTITY_REQUIRED')
    if type(gpu_index) is not int or type(gpu_count) is not int or gpu_index<0 or gpu_count<1:raise ValueError('GPU_SELECTION_INVALID')
    nominal=p['project_ram_cap_gib']
    # Commit capacity is not physical RAM. Never promise a private-commit cap
    # that cannot coexist with the configured system commit reserve.
    if mode=='auto':p['project_ram_cap_gib']=max(0.,min(nominal,commit/GiB-p['commit_headroom_floor_gib']))
    if mode=='legacy6':reserve=1024*MiB;cap=6144*MiB;startup=7168*MiB
    else:
        reserve=max(768*MiB,math.ceil(total*.15/(256*MiB))*256*MiB)
        cap=max(0,total-reserve);startup=0
    supported=(gpu_index==0 and gpu_count==1 and mapping_supported)
    p.update(commit_limit_bytes=commit,nominal_ram_cap_gib=nominal,gpu_total_bytes=total,
        gpu_uuid=gpu_uuid,gpu_index=gpu_index,gpu_count=gpu_count,gpu_reserve_bytes=reserve,
        gpu_whole_card_cap_bytes=cap,gpu_startup_free_floor_bytes=startup,
        supported=supported,unsupported_reason=None if supported else 'SINGLE_GPU_ZERO_MAPPING_REQUIRED',
        stage_wait_timeout_seconds=60.,stage_wait_poll_seconds=1.,
        model_execution_profile='6gb',model_loading_parameters_changed=False,
        reservation_semantics='UNSTARTED_UNCONSUMED_ONLY; running allocations already in live measurements')
    p['policy_fingerprint']=_digest(p)
    return p

def _validated_policy(p):
    expected=resolve_device_policy(p['total_physical_bytes'],p['commit_limit_bytes'],p['gpu_total_bytes'],
        gpu_uuid=p['gpu_uuid'],gpu_index=p['gpu_index'],gpu_count=p['gpu_count'],mode=p['mode'],mapping_supported=p['supported'])
    if p!=expected:raise ValueError('HOST_RAM_POLICY_SNAPSHOT_INVALID')
    return p

def _obs(policy,observation):
    s=dict(observation)
    for key in ('physical_free_bytes','commit_headroom_bytes','gpu_free_bytes','gpu_used_bytes'):
        s[key]=_integer(s[key],key)
    for key in ('group_current_bytes','group_working_bytes'):s[key]=_integer(s.get(key,0),key)
    if s.get('gpu_uuid',policy['gpu_uuid'])!=policy['gpu_uuid'] or s.get('gpu_index',0)!=policy['gpu_index']:
        raise ResourcePolicyError('GPU_IDENTITY_CHANGED',decision='CAPACITY_INSUFFICIENT')
    if s.get('gpu_total_bytes',policy['gpu_total_bytes'])!=policy['gpu_total_bytes']:
        raise ResourcePolicyError('GPU_CAPACITY_CHANGED',decision='CAPACITY_INSUFFICIENT')
    if s.get('gpu_count',policy['gpu_count'])!=policy['gpu_count'] or (policy['supported'] and not s.get('mapping_supported',True)):
        raise ResourcePolicyError('GPU_TOPOLOGY_OR_MAPPING_CHANGED',decision='CAPACITY_INSUFFICIENT')
    if s['physical_free_bytes']>policy['total_physical_bytes']:raise ValueError('PHYSICAL_OBSERVATION_INVALID')
    if s['gpu_free_bytes']>policy['gpu_total_bytes'] or s['gpu_used_bytes']>policy['gpu_total_bytes']:
        raise ValueError('GPU_OBSERVATION_INVALID')
    if s['gpu_free_bytes']+s['gpu_used_bytes']>policy['gpu_total_bytes']+MiB:raise ValueError('GPU_OBSERVATION_INCONSISTENT')
    return s

def check_current(policy,observation,*,raise_on_violation=True):
    _validated_policy(policy);s=_obs(policy,observation)
    limits=(('GROUP_CURRENT_PROJECT_RAM_CAP',s['group_current_bytes']>policy['project_ram_cap_gib']*GiB),
        ('GROUP_WORKING_PROJECT_RAM_CAP',s['group_working_bytes']>policy['project_ram_cap_gib']*GiB),
        ('PHYSICAL_FREE_POLICY_FLOOR',s['physical_free_bytes']<policy['physical_free_floor_gib']*GiB),
        ('COMMIT_HEADROOM_POLICY_FLOOR',s['commit_headroom_bytes']<policy['commit_headroom_floor_gib']*GiB),
        ('GPU_WHOLE_CARD_POLICY_CAP',s['gpu_used_bytes']>policy['gpu_whole_card_cap_bytes']),
        ('GPU_FREE_POLICY_RESERVE',s['gpu_free_bytes']<policy['gpu_reserve_bytes']))
    failed=[name for name,bad in limits if bad]
    if not policy['supported']:failed.insert(0,policy['unsupported_reason'])
    result=dict(decision='STOP' if failed else 'CONTINUE',allowed=not failed,reasons=failed,
        policy_fingerprint=policy['policy_fingerprint'],observation=s)
    if failed and raise_on_violation:raise ResourcePolicyError(';'.join(failed),decision='CAPACITY_INSUFFICIENT' if not policy['supported'] else 'WAIT',details=result)
    return result

def evaluate_admission(policy,observation,requirement,reservations=None):
    """Pure queue decision. Never sleep, launch, change model settings or reserve.

    reservation rows MUST have state UNSTARTED. Including a RUNNING row is an
    error rather than silently charging its already measured allocations twice.
    """
    _validated_policy(policy);s=_obs(policy,observation)
    private=_integer(requirement['extra_private_bytes'],'extra_private_bytes')
    physical=_integer(requirement['extra_physical_bytes'],'extra_physical_bytes')
    gpu=_integer(requirement['extra_gpu_mib']*MiB,'extra_gpu_mib')
    unstarted=list(reservations or [])
    rp=rw=rg=0
    for row in unstarted:
        if row.get('state')!='UNSTARTED':raise ValueError('RESERVATION_MUST_BE_UNSTARTED_UNCONSUMED')
        if row.get('gpu_uuid',policy['gpu_uuid'])!=policy['gpu_uuid']:raise ValueError('RESERVATION_GPU_IDENTITY_MISMATCH')
        rp+=_integer(row['extra_private_bytes'],'reservation_private')
        rw+=_integer(row['extra_physical_bytes'],'reservation_physical')
        rg+=_integer(row['extra_gpu_mib']*MiB,'reservation_gpu')
    reserve=_integer(requirement.get('host_reserve_bytes',math.ceil(.03*GiB)),'host_reserve_bytes')
    projected=dict(group_current_bytes=s['group_current_bytes']+private+rp+reserve,
        physical_free_bytes=s['physical_free_bytes']-physical-rw,
        commit_headroom_bytes=s['commit_headroom_bytes']-private-rp,
        gpu_free_bytes=s['gpu_free_bytes']-gpu-rg,gpu_used_bytes=s['gpu_used_bytes']+gpu+rg)
    capacity=[]
    if not policy['supported']:capacity.append(policy['unsupported_reason'])
    if private+reserve>policy['project_ram_cap_gib']*GiB:capacity.append('MODEL_PRIVATE_REQUIREMENT_EXCEEDS_PROJECT_CAP')
    if physical>policy['total_physical_bytes']-policy['physical_free_floor_gib']*GiB:capacity.append('MODEL_PHYSICAL_REQUIREMENT_EXCEEDS_CAPACITY')
    if gpu>policy['gpu_whole_card_cap_bytes']:capacity.append('MODEL_GPU_REQUIREMENT_EXCEEDS_CAPACITY')
    if policy['gpu_startup_free_floor_bytes']>policy['gpu_total_bytes']:capacity.append('LEGACY_GPU_STARTUP_EXCEEDS_DEVICE_CAPACITY')
    failed=[]
    if projected['group_current_bytes']>policy['project_ram_cap_gib']*GiB:failed.append('STAGE_PROJECTED_FAMILY_COMMIT_CAP')
    if projected['physical_free_bytes']<policy['physical_free_floor_gib']*GiB:failed.append('STAGE_PROJECTED_PHYSICAL_FREE_FLOOR')
    if projected['commit_headroom_bytes']<policy['commit_headroom_floor_gib']*GiB:failed.append('STAGE_PROJECTED_COMMIT_HEADROOM_FLOOR')
    if (projected['gpu_used_bytes']>policy['gpu_whole_card_cap_bytes'] or projected['gpu_free_bytes']<policy['gpu_reserve_bytes']):failed.append('STAGE_PROJECTED_GPU_CAP_OR_RESERVE')
    if s['gpu_free_bytes']<policy['gpu_startup_free_floor_bytes']:failed.append('LEGACY_GPU_STARTUP_FREE_FLOOR')
    return dict(decision='CAPACITY_INSUFFICIENT' if capacity else 'WAIT' if failed else 'ADMIT',
        reasons=capacity or failed,projected=projected,policy_fingerprint=policy['policy_fingerprint'],
        reservations_count=len(unstarted),observed_whole_card_used_bytes=s['gpu_used_bytes'])

class _Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_uint64) for n in ('total','available','total_pagefile','available_pagefile','total_virtual','available_virtual','extended')]
class _Perf(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('commit_total','commit_limit','commit_peak','physical_total','physical_available','system_cache','kernel_total','kernel_paged','kernel_nonpaged','page_size')]+[(n,ctypes.c_ulong) for n in ('handles','processes','threads')]
class _Gpu(ctypes.Structure):
    _fields_=[(n,ctypes.c_uint64) for n in ('total','free','used')]

class DeviceSampler:
    """Windows/NVML, single physical GPU 0 only. No Torch, Paddle or contexts."""
    def __init__(self,gpu_index=0,gpu_uuid=None):
        if os.name!='nt':raise OSError('WINDOWS_RESOURCE_SAMPLER_REQUIRED')
        if gpu_index!=0:raise ResourcePolicyError('SINGLE_GPU_ZERO_MAPPING_REQUIRED',decision='CAPACITY_INSUFFICIENT')
        self.lib=None;self.closed=False
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.psapi=ctypes.WinDLL('psapi',use_last_error=True)
        try:
            self.lib=ctypes.WinDLL('nvml.dll')
            if self.lib.nvmlInit_v2()!=0:raise OSError('NVML_INIT_FAILED')
            count=ctypes.c_uint()
            if self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count))!=0:raise OSError('NVML_COUNT_FAILED')
            self.count=count.value
            self.handle=ctypes.c_void_p()
            self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p)]
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))!=0:raise OSError('NVML_DEVICE_ZERO_UNAVAILABLE')
            self.lib.nvmlDeviceGetUUID.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_uint]
            value=ctypes.create_string_buffer(128)
            if self.lib.nvmlDeviceGetUUID(self.handle,value,len(value))!=0:raise OSError('NVML_UUID_FAILED')
            self.uuid=value.value.decode('ascii')
            if gpu_uuid and gpu_uuid!=self.uuid:raise ResourcePolicyError('GPU_IDENTITY_CHANGED',decision='CAPACITY_INSUFFICIENT')
            visible=os.environ.get('CUDA_VISIBLE_DEVICES')
            self.mapping_supported=self.count==1 and (visible is None or visible in ('0',self.uuid))
            self.lib.nvmlDeviceGetMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(_Gpu)]
            self.psapi.GetPerformanceInfo.argtypes=[ctypes.POINTER(_Perf),ctypes.c_ulong]
            self.kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(_Memory)]
        except BaseException:
            self.close();raise
    def sample(self):
        if self.closed:raise OSError('DEVICE_SAMPLER_CLOSED')
        memory=_Memory();memory.length=ctypes.sizeof(memory)
        perf=_Perf();perf.cb=ctypes.sizeof(perf)
        gpu=_Gpu()
        count=ctypes.c_uint()
        if self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count))!=0:raise OSError('NVML_COUNT_FAILED')
        if not self.kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):raise ctypes.WinError(ctypes.get_last_error())
        if not self.psapi.GetPerformanceInfo(ctypes.byref(perf),perf.cb):raise ctypes.WinError(ctypes.get_last_error())
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(gpu))!=0:raise OSError('NVML_SAMPLE_FAILED')
        return dict(total_physical_bytes=int(memory.total),physical_free_bytes=int(memory.available),
            commit_limit_bytes=int(perf.commit_limit*perf.page_size),commit_headroom_bytes=int((perf.commit_limit-perf.commit_total)*perf.page_size),
            gpu_total_bytes=int(gpu.total),gpu_free_bytes=int(gpu.free),gpu_used_bytes=int(gpu.used),
            gpu_uuid=self.uuid,gpu_index=0,gpu_count=count.value,mapping_supported=self.mapping_supported and count.value==1,
            observer='STDLIB_WINDOWS_NVML_NO_MODEL_IMPORT',sample_epoch=time.time())
    def close(self):
        if self.lib:
            try:self.lib.nvmlShutdown()
            except (OSError,AttributeError):pass
            self.lib=None
        self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def sample_device(gpu_index=0,gpu_uuid=None):
    with DeviceSampler(gpu_index,gpu_uuid) as sampler:return sampler.sample()

def observation_from_legacy(host,gpu,*,group_private_gib=0,group_working_gib=0,policy=None):
    p=policy or current_memory_policy()
    return dict(physical_free_bytes=host['PhysicalAvailable_gib']*GiB,
        commit_headroom_bytes=host['commit_headroom_gib']*GiB,
        group_current_bytes=group_private_gib*GiB,group_working_bytes=group_working_gib*GiB,
        gpu_free_bytes=gpu.get('gpu_free_bytes',gpu['free']*MiB),
        gpu_used_bytes=gpu.get('gpu_used_bytes',gpu['used']*MiB),
        gpu_total_bytes=gpu.get('gpu_total_bytes',p['gpu_total_bytes']),
        gpu_uuid=gpu.get('gpu_uuid',p['gpu_uuid']),gpu_index=gpu.get('gpu_index',p['gpu_index']))

def physical_total_bytes():
    return sample_device()['total_physical_bytes']

def configured_policy(tool_root):
    path=Path(tool_root)/'HOST_MEMORY_POLICY.json'
    config=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {'schema':SCHEMA,'mode':'auto'}
    if config.get('schema') not in (SCHEMA,'bemarkdown-host-memory-policy-v1'):raise ValueError('HOST_RAM_CONFIG_SCHEMA_INVALID')
    obs=sample_device(config.get('gpu_index',0),config.get('gpu_uuid'))
    return resolve_device_policy(obs['total_physical_bytes'],obs['commit_limit_bytes'],obs['gpu_total_bytes'],
        gpu_uuid=obs['gpu_uuid'],gpu_index=obs['gpu_index'],gpu_count=obs['gpu_count'],mode=config.get('mode','auto'),mapping_supported=obs['mapping_supported'])
def activate_host_policy(tool_root):
    if os.environ.get(ENV):
        policy=current_memory_policy()
        observed=sample_device(policy['gpu_index'],policy['gpu_uuid'])
        if (observed['total_physical_bytes']!=policy['total_physical_bytes'] or
            observed['gpu_total_bytes']!=policy['gpu_total_bytes'] or observed['gpu_count']!=policy['gpu_count'] or
            observed['mapping_supported']!=policy['supported']):
            raise ValueError('MANAGED_POLICY_DEVICE_IDENTITY_CHANGED')
    else:policy=configured_policy(tool_root)
    os.environ[ENV]=json.dumps(policy,separators=(',',':'));return policy
def current_memory_policy(env=None):
    raw=(os.environ if env is None else env).get(ENV)
    return _validated_policy(json.loads(raw)) if raw else configured_policy(Path(__file__).resolve().parents[2])
def replace_with_retry(source,target):
    for attempt in range(6):
        try:os.replace(source,target);return
        except OSError as exc:
            if getattr(exc,'winerror',None) not in {5,32,33} or attempt==5:raise
            time.sleep(.05*2**attempt)
