import torch
import mamtorchkernel

class MAMDenseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        C, argmax, argmin = mamtorchkernel.mamdense_forward(A, B)
        ctx.save_for_backward(A, B, argmax, argmin)
        return C

    @staticmethod
    def backward(ctx, C_grad):
        A, B, argmax, argmin = ctx.saved_tensors
        #A_grad, B_grad = mamtorchkernel.mamdense_backward(A, B, C_grad, argmax, argmin)
        #return A_grad, B_grad
        return A, B
    

class MAMDense(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MAMDense, self).__init__()
        self.input_features = in_features
        self.state_size = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        #self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize weight and bias here
        torch.nn.init.xavier_uniform_(self.weight)  # You can use a different initialization method
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        
    def forward(self, input):
        C = MAMDenseFunction.apply(input, self.weight.T.contiguous())
        if self.bias is not None:
            C += self.bias.view(1, -1)  # Add bias
        return C