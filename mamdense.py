import torch
import mamtorchkernel

def mam_backward(A, B, Cgrad, argmax, argmin):
    AzeroedT = torch.zeros((A.shape[1], A.shape[0]), device=A.device)
    aranged_indices = np.arange(A.shape[0]).reshape(-1, 1)
    AzeroedT[argmax, aranged_indices] = A[aranged_indices, argmax]
    AzeroedT[argmin, aranged_indices] = A[aranged_indices, argmin]
    Bgrad = AzeroedT@Cgrad
    del AzeroedT
    
    BzeroedT = torch.zeros((B.shape[1], B.shape[0]), device=B.device)
    aranged_indices = np.arange(B.shape[1]).reshape(1, -1)
    BzeroedT[aranged_indices, argmax] = B[argmax, aranged_indices]
    BzeroedT[aranged_indices, argmin] = B[argmin, aranged_indices]
    Agrad = Cgrad@BzeroedT
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