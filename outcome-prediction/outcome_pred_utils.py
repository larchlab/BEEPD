import torch

from typing import Dict, Any

CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
if MPS_AVAILABLE:
    MPS_DEVICE = torch.device('mps')


def make_tensor_cuda_if_available(t: torch.Tensor) -> torch.Tensor:
    if CUDA_AVAILABLE:
        return t.cuda()
    elif MPS_AVAILABLE:
        return t.to(MPS_DEVICE)
    else:
        return t
    
def make_dict_vals_cuda_if_available(d: Dict[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
    if CUDA_AVAILABLE:
        return {k: v.cuda() for k,v in d.items()}
    elif MPS_AVAILABLE:
        return {k: v.to(MPS_DEVICE) for k,v in d.items()}
    else:
        return d