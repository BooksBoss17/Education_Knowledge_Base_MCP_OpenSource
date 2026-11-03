"""Windows RAM/commit guard, independently usable before any model launch."""
import ctypes,json,pathlib,time
from ctypes import wintypes

LIMITS={'project_memory_cap_gib':6.0,
        'startup_commit_headroom_gib':3.0,'startup_physical_available_gib':2.0,
        'running_commit_reserve_gib':3.0,'running_physical_reserve_gib':2.0,
        'measured_model_commit_gib':8.0,'estimated_model_working_gib':5.0,
        'runner_memory_budget_gib':0.5,
        'poll_seconds':2.0}
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
def check(snapshot,phase='startup',limits=LIMITS):
    if phase not in ('startup','running'):raise ValueError('Unknown guard phase')
    commit=limits['startup_commit_headroom_gib' if phase=='startup' else 'running_commit_reserve_gib']
    physical=limits['startup_physical_available_gib' if phase=='startup' else 'running_physical_reserve_gib']
    problems=[]
    if snapshot['commit_headroom_gib']<commit:problems.append('INSUFFICIENT_SYSTEM_COMMIT_HEADROOM')
    if snapshot['PhysicalAvailable_gib']<physical:problems.append('INSUFFICIENT_PHYSICAL_RAM_HEADROOM')
    if phase=='startup':
        # Latest user cap is project usage <=6 GiB, independent of host reserves.
        commit_budget=limits['measured_model_commit_gib']+limits['runner_memory_budget_gib']
        working_budget=limits['estimated_model_working_gib']+limits['runner_memory_budget_gib']
        if commit_budget>limits['project_memory_cap_gib']:problems.append('PROJECT_COMMIT_BUDGET_EXCEEDS_CAP')
        if working_budget>limits['project_memory_cap_gib']:problems.append('PROJECT_RAM_BUDGET_EXCEEDS_CAP')
        if snapshot['commit_headroom_gib']-commit_budget<limits['running_commit_reserve_gib']:
            problems.append('INSUFFICIENT_SYSTEM_SPACE_FOR_PROJECT_BUDGET')
        if snapshot['PhysicalAvailable_gib']-working_budget<limits['running_physical_reserve_gib']:
            problems.append('INSUFFICIENT_PHYSICAL_SPACE_FOR_PROJECT_BUDGET')
    else:
        if snapshot.get('project_private_commit_gib',0)>limits['project_memory_cap_gib']:problems.append('PROJECT_PRIVATE_COMMIT_EXCEEDS_CAP')
        if snapshot.get('project_working_gib',0)>limits['project_memory_cap_gib']:problems.append('PROJECT_WORKING_SET_EXCEEDS_CAP')
    return {'allowed':not problems,'phase':phase,'problems':problems,'snapshot':snapshot,'limits':limits}
def require_startup(output):
    result=check(sample())
    path=pathlib.Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    if not result['allowed']:raise MemoryError('Model launch refused by RAM/commit safety guard')
    return result
def controls():
    low={**LIMITS,'measured_model_commit_gib':4.25,'estimated_model_working_gib':4.25}
    good={'commit_headroom_gib':8,'PhysicalAvailable_gib':10}
    assert check(good,limits=low)['allowed']
    assert not check(good)['allowed'] # original 8 GiB model violates project cap
    assert not check({**good,'commit_headroom_gib':1.97},limits=low)['allowed']
    assert not check({**good,'PhysicalAvailable_gib':1.9},'running')['allowed']
    assert not check({**good,'project_private_commit_gib':6.01},'running')['allowed']
    assert not check({**good,'project_working_gib':6.01},'running')['allowed']
    assert check({**good,'project_private_commit_gib':6,'project_working_gib':6},'running')['allowed']
    assert not check({**good,'commit_headroom_gib':2.9},'running')['allowed']
    return 8
if __name__=='__main__':
    print(json.dumps({'controls_passed':controls(),'preflight':check(sample())}))
