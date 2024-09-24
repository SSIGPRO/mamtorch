import torch
from torch import Tensor

__all__ = ["fullyconnected"]

library_name = "mamtorch_kernel_v2"
K = torch.ops.mamtorch_kernel_v2

def fullyconnected(a: Tensor, b: Tensor, bias: Tensor, beta: float) -> list[Tensor]:
    return K.fullyconnected.default(a, b, bias, beta)

@torch.library.register_fake(f"{library_name}::fullyconnected")
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
    return torch.empty((a.size(0), b.size(1)), dtype=torch.float, device=a.device), \
           torch.empty((a.size(0), b.size(1)), dtype=torch.int, device=a.device), \
           torch.empty((a.size(0), b.size(1)), dtype=torch.int, device=a.device)

def _backward(ctx, grad):
    a, b, argmax, argmin = ctx.saved_tensors
    beta = ctx.beta
    a_grad, b_grad, bias_grad = None, None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        a_grad, b_grad = K.fullyconnected_backward.default(a, b, grad[0], argmax, argmin, beta)
    if ctx.needs_input_grad[2]:
        bias_grad = grad[0]
    return a_grad, b_grad, bias_grad, None

def _setup_context(ctx, inputs, output):
    a, b, _, beta = inputs
    _, argmax, argmin = output
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
        saved_a = a
        saved_b = b
    ctx.beta = beta
    ctx.save_for_backward(saved_a, saved_b, argmin, argmax)

torch.library.register_autograd(
    f"{library_name}::fullyconnected", _backward, setup_context=_setup_context)

@torch.library.register_fake(f"{library_name}::fullyconnected_backward")
def _(a, b, grad, argmax, argmin, beta):
    print("I'm checking")
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