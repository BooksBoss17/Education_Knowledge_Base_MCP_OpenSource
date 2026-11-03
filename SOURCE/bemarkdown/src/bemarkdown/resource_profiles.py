"""Inference scheduling profiles; models and output contracts stay fixed."""
from dataclasses import asdict, dataclass
import os
from .host_memory_policy import current_memory_policy


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    target_mib: int
    hard_limit_mib: int
    got_allocator_mib: int
    got_batch_size: int
    got_vision_batch_size: int
    overlap_got_dependencies: bool

    def to_dict(self):
        return {**asdict(self),
                'native_process_isolation':self.name=='6gb',
                'serial_heavy_model_stages':self.name=='6gb',
                'model_execution_profile':self.name,
                'device_resource_policy':current_memory_policy() if self.name=='6gb' else None,
                'process_group_private_commit_cap_mib':int(current_memory_policy()['project_ram_cap_gib']*1024) if self.name=='6gb' else None,
                'hard_limit_scope':'LEGACY_PROFILE_FIELD_NOT_PROCESS_GROUP_CAP'}


PROFILES = {
    '6gb': ResourceProfile('6gb', 6144, 8192, 2816, 32, 1, False),
    '10gb': ResourceProfile('10gb', 10240, 10240, 6144, 32, 1, True),
}


def resource_profile(name=None):
    name = name if name is not None else os.environ.get('BEMARKDOWN_RESOURCE_PROFILE', '6gb')
    if name not in PROFILES:
        raise ValueError('BEMARKDOWN_RESOURCE_PROFILE must be 6gb or 10gb')
    return PROFILES[name]
