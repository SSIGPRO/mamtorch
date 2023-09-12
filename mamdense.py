import torch
import mamtorchkernel

class MAMDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        C, argmax, argmin = mamtorchkernel.mamdense_forward(A, B)
        ctx.save_for_backward(A, B, argmax, argmin)
        return C, argmax, argmin

    @staticmethod
    def backward(ctx, C_grad):
        A, B, argmax, argmin = ctx.saved_tensors
        A_grad, B_grad = mamtorchkernel.mamdense_backward(A, B, C_grad, argmax, argmin)
        return A_grad, B_grad