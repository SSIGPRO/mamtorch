import torch
from torch import Tensor

__all__ = ["fullyconnected_fast"]

library_name = "mamtorch_kernel_v4"
K = torch.ops.mamtorch_kernel_v4

def fullyconnected_fast(a: Tensor, b: Tensor) -> Tensor:
    return K.fullyconnected_fast.default(a, b)

@torch.library.register_fake(f"{library_name}::fullyconnected_fast")
def _(a, b):
    torch._check(a.size(1) == b.size(0))
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    torch._check(a.is_contiguous())
    torch._check(b.is_contiguous())
    return torch.empty((a.size(0), b.size(1)), dtype=torch.float, device=a.device)

def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    a_grad, b_grad = None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        a_grad = grad@b.T
        b_grad = a.T@grad
    return a_grad, b_grad

def _setup_context(ctx, inputs, output):
    a, b = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        saved_a = a
        saved_b = b
    ctx.save_for_backward(saved_a, saved_b)

torch.library.register_autograd(
    f"{library_name}::fullyconnected_fast", _backward, setup_context=_setup_context)