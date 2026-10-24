import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_common import HSDPMeshInfo


def set_partial_reshard_mesh(module: FSDPModule, mesh: DeviceMesh) -> int:
    if mesh.ndim != 2:
        raise ValueError("partial reshard requires a two-dimensional mesh")

    state = module._get_fsdp_state()
    group = state._fsdp_param_group
    if group is None:
        raise ValueError("partial reshard requires an FSDP parameter group")
    if group.mesh_info.shard_mesh_size <= mesh.size(1):
        raise ValueError("partial reshard mesh must use fewer shard ranks than FSDP")

    post_forward_mesh_info = HSDPMeshInfo(
        mesh,
        shard_mesh_dim=1,
        replicate_mesh_dim=0,
    )
    group.post_forward_mesh_info = post_forward_mesh_info
    state._auto_reshard_after_forward = False

    for fsdp_param in group.fsdp_params:
        if fsdp_param.is_dtensor:
            raise ValueError("partial reshard does not support DTensor parameters")
        fsdp_param.post_forward_mesh_info = post_forward_mesh_info
        shape_only = torch.empty(fsdp_param._orig_size, device="meta")
        fsdp_param._init_sharded_post_forward_param_metadata(shape_only)

    return len(group.fsdp_params)
