import torch
from torch import Tensor

__all__ = ["fullyconnected_fast"]

library_name = "mamtorch_kernel_v3"
K = torch.ops.mamtorch_kernel_v3

def fullyconnected_fast(a: Tensor, b: Tensor, bias: Tensor, beta: float) -> Tensor:
    return K.fullyconnected_fast.default(a, b, bias, beta)

@torch.library.register_fake(f"{library_name}::fullyconnected_fast")
def _(a, b, bias, beta):
    torch._check(a.size(1) == b.size(0))
    torch._check(b.size(1) == bias.size(0))
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(bias.dtype == torch.float)
    torch._check(a.device == b.device)
    torch._check(a.device == bias.device)
    torch._check(a.is_contiguous())
    torch._check(b.is_contiguous())
    torch._check(bias.is_contiguous())
    return torch.empty((a.size(0), b.size(1)), dtype=torch.float, device=a.device)

def _backward(ctx, grad):
    # use standard dense gradient evaluation (no argmax and argmin provided)
    a, b = ctx.saved_tensors
    a_grad, b_grad, bias_grad = None, None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]  or ctx.needs_input_grad[2]:
        bias_grad_tmp = grad
        a_grad_tmp = grad@b.T
        b_grad_tmp = a.T@grad
    if ctx.needs_input_grad[0]:
        a_grad = a_grad_tmp
    if ctx.needs_input_grad[1]:
        b_grad = b_grad_tmp
    if ctx.needs_input_grad[2]:
        bias_grad = bias_grad_tmp
    return a_grad, b_grad, bias_grad, None

def _setup_context(ctx, inputs, output):
    a, b, _, _ = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_a = a
    if ctx.needs_input_grad[1]:
        saved_b = b
    ctx.save_for_backward(saved_a, saved_b)

torch.library.register_autograd(
    f"{library_name}::fullyconnected_fast", _backward, setup_context=_setup_context)