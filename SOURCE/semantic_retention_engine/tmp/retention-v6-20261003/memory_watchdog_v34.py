"""Track host/process peaks and terminate only identity-bound task processes."""

import ctypes, json, pathlib, time

from ctypes import wintypes

K=ctypes.windll.kernel32;P=ctypes.windll.psapi

K=ctypes.windll.kernel32;P=ctypes.windll.psapi

K.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]

K.OpenProcess.restype=wintypes.HANDLE

K.CloseHandle.argtypes=[wintypes.HANDLE]

class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]

K.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]

K.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4

K.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]

P.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]

def process_info(pid):
    handle=K.OpenProcess(0x0410,False,pid)
    if not handle:return None
    try:
        size=wintypes.DWORD(32768);buf=ctypes.create_unicode_buffer(size.value)
        if not K.QueryFullProcessImageNameW(handle,0,buf,ctypes.byref(size)):return None
        creation=wintypes.FILETIME();exit_time=wintypes.FILETIME();kernel=wintypes.FILETIME();user=wintypes.FILETIME()
        if not K.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(kernel),ctypes.byref(user)):return None
        c=Counters();c.cb=ctypes.sizeof(c)
        if not P.GetProcessMemoryInfo(handle,ctypes.byref(c),c.cb):return None
        ticks=(creation.dwHighDateTime<<32)|creation.dwLowDateTime
        return {'pid':pid,'exe':buf.value,'created_ticks':ticks,
                'working_gib':c.WorkingSetSize/2**30,'peak_working_gib':c.PeakWorkingSetSize/2**30,
                'private_commit_gib':c.PrivateUsage/2**30,'peak_pagefile_gib':c.PeakPagefileUsage/2**30}
    finally:K.CloseHandle(handle)

def save(path,value):
    path=pathlib.Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8')
    for attempt in range(5):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1*(attempt+1))
