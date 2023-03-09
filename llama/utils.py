import os
import torch
import contextlib
import functools

from typing import Optional, Union, Tuple


@functools.lru_cache()
def get_torch_device_name() -> str:
    """
    Returns the torch device name
    """
    if torch.cuda.is_available(): return 'cuda'
    with contextlib.suppress(Exception):
        if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'


@functools.lru_cache()
def get_torch_device(
    name: Optional[Union[str, torch.device]] = None,
) -> 'torch.device':
    """
    Returns the torch device
    """
    if isinstance(name, torch.device): return name
    return torch.device(name or get_torch_device_name())

@functools.lru_cache()
def get_torch_dist_method(
    name: Optional[str] = None,
) -> str:
    """
    Returns the torch distributed method
    """
    if name is None:
        name = get_torch_device_name()
    return 'nccl' if name == 'cuda' else 'gloo'


@functools.lru_cache()
def get_torch_default_tensor_type(
    name: Optional[str] = None,
) -> 'torch.Tensor':
    """
    Returns the torch default tensor type
    """
    if name is None:
        name = get_torch_device_name()
    return torch.cuda.HalfTensor if name == 'cuda' else torch.BFloat16Tensor
    # return 'torch.cuda.FloatTensor' if name == 'cuda' else 'torch.FloatTensor'



def setup_model_parallel(
    seed: int,
    device_name: Optional[str] = None,
) -> Tuple[int, int]:

    from fairscale.nn.model_parallel.initialize import initialize_model_parallel
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    device_name = device_name or get_torch_device_name()

    torch.distributed.init_process_group(get_torch_dist_method(device_name))
    
    initialize_model_parallel(world_size)
    if device_name == 'cuda':
        torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(seed)
    return local_rank, world_size
