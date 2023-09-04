import torch
import mamtorchkernel

def python_mam_backward(A, B, Cgrad, argmax, argmin):
    Aaranged_indices = torch.arange(A.shape[0]).reshape(-1, 1)
    Baranged_indices = torch.tile(torch.arange(B.shape[1]).reshape(1, -1), (1, 2))  
    arg = torch.concatenate([argmax, argmin], axis=-1).to(int)
    
    AzeroedT = torch.zeros((A.shape[1], A.shape[0]), device=A.device)
    AzeroedT[arg, Aaranged_indices] = A[Aaranged_indices, arg]
    Bgrad = torch.zeros_like(B)
    Bgrad[arg, Baranged_indices] = (AzeroedT@Cgrad)[arg, Baranged_indices]
    del AzeroedT
    
    BzeroedT = torch.zeros((B.shape[1], B.shape[0]), device=B.device)
    BzeroedT[Baranged_indices, arg] = B[arg, Baranged_indices]
    Agrad = torch.zeros_like(A)
    Agrad[Aaranged_indices, arg] = (Cgrad@BzeroedT)[Aaranged_indices, arg]
    del BzeroedT
    
    return Agrad, Bgrad

class MAMDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        C, argmax, argmin = mamtorchkernel.mam_forward(A, B)
        ctx.save_for_backward(A, B, argmax, argmin)
        return C, argmax, argmin

    @staticmethod
    def backward(ctx, C_grad):
        A, B, argmax, argmin = ctx.saved_tensors
        A_grad, B_grad = mam_backward(A, B, C_grad, argmax, argmin)
        return A_grad, B_grad