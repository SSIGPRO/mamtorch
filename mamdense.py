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
        factory_kwargs = {}
        self.input_features = in_features
        self.state_size = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features), **factory_kwargs)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features), **factory_kwargs)
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
        input_flat = input.view(-1, input.size()[-1])
        C = MAMDenseFunction.apply(input_flat, self.weight.T.contiguous())
        C_shape = list(input.shape[:-1]) + [self.weight.size(0)] 
        #If self.beta is not 0 MAC output is still computed
        if self.beta >= 10e-5:
            D = F.linear(input, self.weight)
            if self.bias is not None:
                C += self.bias.view(1, -1)  # Add bias
                mam_term = (1-self.beta)*C + self.bias.view(1, -1)
                mam_term = mam_term.view(C_shape)
                mac_term = self.beta*D
                return mam_term + mac_term
            mam_term = (1-self.beta)*C
            mam_term = mam_term.view(C_shape)
            mac_term = self.beta*D
            return mam_term + mac_term
        #No need to compute MAC output
        if self.bias is not None:
            C += self.bias.view(1, -1)  # Add bias
            C = C.view(C_shape)
        return C
        
                   