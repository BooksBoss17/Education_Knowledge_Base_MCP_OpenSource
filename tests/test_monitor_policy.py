import runpy
from pathlib import Path

def test_external_monitor_records_without_ram_caps():
    policy = runpy.run_path(str(Path(__file__).parents[1]/'TOOLS/runtime_manager/external_resource_policy.py'))
    p = policy['resolve_device_policy'](8*1024**3,16*1024**3,0)
    assert p == policy['resolve_device_policy'](8*1024**3,20*1024**3,0)
    assert 'commit_limit_bytes' not in p
    assert p['project_ram_cap_gib'] is None
    obs = dict(group_current_bytes=12*1024**3,physical_free_bytes=1,commit_headroom_bytes=1)
    assert policy['check_current'](p,obs)['allowed']
    assert policy['evaluate_admission'](p,obs,dict(extra_gpu_mib=0))['decision']=='ADMIT'
