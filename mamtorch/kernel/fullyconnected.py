import torch
from torch import Tensor

__all__ = ["fullyconnected"]

def fullyconnected(a: Tensor, b: Tensor) -> list[Tensor]:
    return torch.ops.mamtorch.fullyconnected.default(a, b)

@torch.library.register_fake("mamtorch::fullyconnected")
def _(a, b):
    torch._check(a.size(1) == b.size(0))
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    torch._check(a.is_contiguous())
    torch._check(b.is_contiguous())
    return torch.empty((a.size(0), b.size(1)), dtype=torch.float, device=a.device), \
           torch.empty((a.size(0), b.size(1)), dtype=torch.int, device=a.device), \
           torch.empty((a.size(0), b.size(1)), dtype=torch.int, device=a.device)

def _backward(ctx, grad):
    a, b, argmax, argmin = ctx.saved_tensors
    a_grad, b_grad = None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        a_grad_tmp, b_grad_tmp = torch.ops.mamtorch.fullyconnected_backward.default(a, b, grad[0], argmax, argmin)
    if ctx.needs_input_grad[0]:
        a_grad = a_grad_tmp
    if ctx.needs_input_grad[1]:
        b_grad = b_grad_tmp
    return a_grad, b_grad

def _setup_context(ctx, inputs, output):
    a, b = inputs
    c, argmax, argmin = output
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_a = a
    if ctx.needs_input_grad[1]:
        saved_b = b
    ctx.save_for_backward(saved_a, saved_b, argmin, argmax)

torch.library.register_autograd(
    "mamtorch::fullyconnected", _backward, setup_context=_setup_context)

@torch.library.register_fake("mamtorch::fullyconnected_backward")
def _(a, b, grad, argmax, argmin):
    torch._check(a.size(1) == b.size(0))
    torch._check(grad.size(0) == a.size(0))
    torch._check(grad.size(1) == b.size(1))
    torch._check(grad.shape == argmax.shape)
    torch._check(grad.shape == argmin.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(grad.dtype == torch.float)
    torch._check(argmax.dtype == torch.int)
    torch._check(argmin.dtype == torch.int)
    torch._check(a.device == b.device)
    torch._check(a.device == grad.device)
    torch._check(a.device == argmax.device)
    torch._check(a.device == argmin.device)
    torch._check(a.is_contiguous())
    torch._check(b.is_contiguous())
    torch._check(grad.is_contiguous())
    torch._check(argmax.is_contiguous())
    torch._check(argmin.is_contiguous())
    return torch.empty_like(a), torch.empty_like(b)