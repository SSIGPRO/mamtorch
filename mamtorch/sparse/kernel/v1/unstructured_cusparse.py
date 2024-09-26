import torch
from torch import Tensor

__all__ = ["unstructured_cusparse"]

library_name = "mamtorch_kernel_sparsev1"
K = torch.ops.mamtorch_kernel_sparsev1

def unstructured_cusparse(a: Tensor, b: Tensor) -> Tensor:
    return K.unstructured_cusparse.default(a, b)

@torch.library.register_fake(f"{library_name}::unstructured_cusparse")
def _(a, b):
    torch._check(a.size(1) == b.size(0))
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    torch._check(a.is_contiguous())
    torch._check(b.is_contiguous())
    return torch.empty((a.size(0), b.size(1)), dtype=torch.float, device=a.device)