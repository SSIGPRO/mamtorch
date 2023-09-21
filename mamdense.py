import torch
import mamtorchkernel
import math
import torch.nn.functional as F

class MAMDenseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        C, argmax, argmin = mamtorchkernel.mamdense_forward(A, B)
        ctx.save_for_backward(A, B, argmax, argmin)
        return C

    @staticmethod
    def backward(ctx, C_grad):
        A, B, argmax, argmin = ctx.saved_tensors
        A_grad, B_grad = mamtorchkernel.mamdense_backward(A, B, C_grad, argmax, argmin)
        return A_grad, B_grad
    

class MAMDense(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, beta=False, beta_decay='linear', beta_epochs=0):
        super(MAMDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        if beta:
            self.beta = 1.0
        else:
            self.beta = 0.0
        self.beta_decay = beta_decay
        self.beta_epochs = beta_epochs
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize weight and bias here
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
            
    def adjust_beta(self, epoch):
        assert self.beta_epochs > 0, "Invalid value for beta_epochs. Please use a positive integer."

        if epoch+1 >= self.beta_epochs:
            self.beta = 0
            return
        if self.beta_decay == 'linear':
            delta_beta =  1/self.beta_epochs
            self.beta -= delta_beta
            return
        
    def forward(self, input):
        C = MAMDenseFunction.apply(input, self.weight.T.contiguous())
        # If self.beta is not 0 MAC output is still computed
        if self.beta >= 10e-5:
            D = F.linear(input, self.weight)
            if self.bias is not None:
                return (1-self.beta)*C + self.beta*D + self.bias.view(1, -1)
            return (1-self.beta)*C + self.beta*D
        
        # No need to compute MAC output
        if self.bias is not None:
            return C + self.bias.view(1, -1)
        
        return C
        
                   