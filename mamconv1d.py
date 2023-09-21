import torch
import mamtorchkernel
import math
import torch.nn.functional as F

class MAMConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, stride):
        Y, argmax, argmin = mamtorchkernel.mamconv1d_forward(X, W, stride)
        ctx.save_for_backward(X, W, argmax, argmin)
        return Y, argmax, argmin

    @staticmethod
    def backward(ctx, Y_grad, argmax_grad, argmin_grad):
        X, W, argmax, argmin = ctx.saved_tensors
        X_grad, W_grad = mamtorchkernel.mamconv1d_backward(X, W, Y_grad, argmax, argmin)
        return X_grad, W_grad, None, None
    

class MAMConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=1, padding=0, padding_mode='zeros', beta=False, beta_decay='linear', beta_epochs=0):
        super(MAMDense, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        
        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        if beta:
            self.beta = 1.0
        else:
            self.beta = 0.0
        self.beta_decay = beta_decay
        self.beta_epochs = beta_epochs
        
        self.reset_parameters()
        
    def reset_parameters(self): # CHECK FOR CONV1D PROPER INITIALIZATION
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
        # apply padding
        if self.padding_mode == 'zeros':
            pad_mode = 'constant'
        else:
            pad_mode = self.padding_mode
            
        if type(self.padding) is tuple or type(self.padding) is list:
            if len(self.padding) <= 2:
                X = torch.nn.functional.pad(input, self.padding, mode=pad_mode)
            else:
                raise("To much elements in padding list or tuple.")
        elif type(self.padding) is int:
            if self.padding == 0:
                X = input
            elif self.padding > 0:
                padl = self.padding//2
                padr = self.padding-padl
                X = torch.nn.functional.pad(input, (padl, padr), mode=pad_mode)
            else:
                raise("Padding value must be equal or greater than 0.")
        elif self.padding == 'same':
            pad_total = self.weight.size(0)-1
            padl = pad_total//2
            padr = pad_total-padl
            X = torch.nn.functional.pad(input, (padl, padr), mode=pad_mode)
        elif self.padding == 'valid':
            X = input
        else:
            raise("Invalid padding value.")
        
        Y, argmax, argmin = MAMConv1dFunction.apply(input, self.weight.contiguous(), self.stride)
        
        # Save argmax and argmin for external use
        self.argmax = argmax
        self.argmin = argmin
        
        # If self.beta is not 0 MAC output is still computed
        if self.beta >= 10e-5:
            Yb = F.conv1d(input, self.weight, stride=self.stride)
            if self.bias is not None:
                return (1-self.beta)*Y + self.beta*Yb + self.bias[:, None]
            return (1-self.beta)*Y + self.beta*Yb
        
        # No need to compute MAC output
        if self.bias is not None:
            return Y + self.bias[:, None]
        
        return Y
        
                   