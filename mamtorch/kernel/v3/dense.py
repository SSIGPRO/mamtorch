import torch
from torch import Tensor

__all__ = ["dense"]

library_name = "mamtorch_kernel_v3"
K = torch.ops.mamtorch_kernel_v3

def dense(a: Tensor, b: Tensor) -> Tensor:
    return K.dense.default(a, b)

@torch.library.register_fake(f"{library_name}::dense")
def _(a, b):
    torch._check(a.size(1) == b.size(0))
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    torch._check(a.is_contiguous())
    torch._check(b.is_contiguous())
    return torch.empty((a.size(0), b.size(1)), dtype=torch.float, device=a.device)