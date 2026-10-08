"""Windows RAM/commit guard, independently usable before any model launch."""

import ctypes, time

from ctypes import wintypes

class Info(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(n,wintypes.DWORD) for n in ('HandleCount','ProcessCount','ThreadCount')]

def sample():
    p=Info();p.cb=ctypes.sizeof(p)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p),p.cb):
        raise OSError('Cannot verify Windows memory; fail closed')
    result={k+'_gib':getattr(p,k)*p.PageSize/2**30 for k in ('CommitTotal','CommitLimit','PhysicalTotal','PhysicalAvailable','KernelTotal','KernelNonpaged')}
    result['commit_headroom_gib']=(p.CommitLimit-p.CommitTotal)*p.PageSize/2**30
    result['timestamp']=time.time()
    return result
