import torch
from torch import Tensor

__all__ = ["cusparsemm"]

library_name = "mamtorch_kernel_sparsev1"
K = torch.ops.mamtorch_kernel_sparsev1

def cusparsemm(a: Tensor, b: Tensor) -> Tensor:
    return K.cusparsemm.default(a, b)

@torch.library.register_fake(f"{library_name}::cusparsemm")
def _(a, b):
    torch._check(a.size(1) == b.size(0))
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty((a.size(0), b.size(1)), dtype=torch.float, device=a.device)