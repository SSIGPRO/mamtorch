import torch
import mamtorchkernel
import math
import torch.nn.functional as F

class MAMConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, hstride, wstride):
        Y, argmax, argmin = mamtorchkernel.mamconv2d_forward(X, W, hstride, wstride)
        ctx.save_for_backward(X, W, argmax, argmin, hstride, wstride)
        return Y, argmax, argmin

    @staticmethod
    def backward(ctx, Y_grad, argmax_grad, argmin_grad):
        X, W, argmax, argmin, hstride, wstride = ctx.saved_tensors
        X_grad, W_grad = mamtorchkernel.mamconv2d_backward(X, W, Y_grad, argmax, argmin, hstride, wstride)
        return X_grad, W_grad, None, None
    

class MAMConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=1, padding=0, padding_mode='zeros', beta=False, beta_decay='linear', beta_epochs=1):
        super(MAMConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.beta_decay = beta_decay
        self.beta_epochs = beta_epochs
        
        if type(kernel_size) is tuple or type(kernel_size) is list:
            self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        else:
            self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
            
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        if beta:
            self.beta = 1.0
            self.adjust_beta(0)
        else:
            self.beta = 0.0
        
        self.max_selection_count = None
        self.min_selection_count = None
        self.argmax = None
        self.argmin = None
        
        if type(stride) is tuple:
            self.hstride = stride[0]
            self.wstride = stride[1]
        else:
            self.hstride = stride
            self.wstride = stride
        
        self.reset_parameters()
        
    def reset_parameters(self): # CHECK FOR CONV2D PROPER INITIALIZATION
        # Initialize weight and bias here
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
            
    def adjust_beta(self, epoch):
        if self.beta_epochs < 0:
            raise Exception("Invalid value for beta_epochs. Please use a positive integer.")
        
        def linear_decay(x, T):
            if x > T:
                return 0.0
            return -x/T+1
        
        def square_wave(x, T, slope):
            if x%T < T//2:
                return 1.0
            return 0.0
            
        def sawtooth_wave(x, T, pause):
            x_loc = x%T
            return linear_decay(x_loc, T-pause)
        
        if self.beta_decay == 'linear':
            self.beta = linear_decay(epoch, self.beta_epochs)
        elif self.beta_decay == "intermittent":
            self.beta = square_wave(epoch, self.beta_epochs)
        elif self.beta_decay == "linear-intermittent":
            self.beta = sawtooth_wave(epoch, self.beta_epochs, self.beta_epochs//2)
                
        
    def reset_selection_count(self):
        self.max_selection_count = torch.zeros_like(self.weight).to(torch.int32)
        self.min_selection_count = torch.zeros_like(self.weight).to(torch.int32)
        
    def update_selection_count(self):
        # Use the current values of self.argmax and self.argmin to update the selection count
        if self.argmax is None or self.argmin is None:
            raise("No argmax or argmin values have been evaluated yet.")
            
        if self.max_selection_count is None or self.min_selection_count is None:
            self.reset_selection_count()
            
        num_batches, num_channels, num_elements_i, num_elements_j = self.argmax.shape
        num_elements = num_elements_i*num_elements_j
        filter_indices = torch.arange(num_channels).repeat_interleave(num_elements).repeat(num_batches).to(self.weight.device)
        channel_indices = self.argmax.flatten() % (self.weight.shape[1])
        element_indices = self.argmax.flatten() // (self.weight.shape[1])
        element_indices_i = element_indices % self.weight.shape[2]
        element_indices_j = element_indices // self.weight.shape[2]
        self.max_selection_count[filter_indices, channel_indices, element_indices_i, element_indices_j] += 1
        
        num_batches, num_channels, num_elements_i, num_elements_j = self.argmin.shape
        num_elements = num_elements_i*num_elements_j
        filter_indices = torch.arange(num_channels).repeat_interleave(num_elements).repeat(num_batches).to(self.weight.device)
        channel_indices = self.argmin.flatten() % (self.weight.shape[1])
        element_indices = self.argmin.flatten() // (self.weight.shape[1])
        element_indices_i = element_indices % self.weight.shape[2]
        element_indices_j = element_indices // self.weight.shape[2]
        self.min_selection_count[filter_indices, channel_indices, element_indices_i, element_indices_j] += 1
        
    def forward(self, input):
        
        X = input
        
        # apply padding
        if self.padding_mode == 'zeros':
            pad_mode = 'constant'
        else:
            pad_mode = self.padding_mode
            
        if type(self.padding) is tuple or type(self.padding) is list:
            if len(self.padding) <= 2:
                X = torch.nn.functional.pad(input, self.padding, mode=pad_mode)
            elif len(self.padding <= 4):
                X = torch.nn.functional.pad(input, self.padding + self.padding, mode=pad_mode)
            else:
                raise("To much elements in padding list or tuple.")
        elif type(self.padding) is int:
            if self.padding == 0:
                X = input
            elif self.padding > 0:
                padl = self.padding//2
                padr = self.padding-padl
                X = torch.nn.functional.pad(input, (padl, padr, padl, padr), mode=pad_mode)
            else:
                raise("Padding value must be equal or greater than 0.")
        elif self.padding == 'same':
            if self.hstride != 1 or self.wstride != 1:
                raise "Cannot apply 'same' padding with stride > 1."
            pad_total_w = self.weight.size(3)-1
            pad_total_h = self.weight.size(2)-1
            padl = pad_total_w//2
            padr = pad_total_w-padl
            padt = pad_total_h//2
            padb = pad_total_h-padt
            X = torch.nn.functional.pad(input, (padl, padr, padt, padb), mode=pad_mode)
        elif self.padding == 'valid':
            X = input
        else:
            raise("Invalid padding value.")
        
        Y, argmax, argmin = MAMConv2dFunction.apply(X, 
                                                    self.weight.contiguous(), 
                                                    torch.tensor(self.hstride), 
                                                    torch.tensor(self.wstride))
        
        # Save argmax and argmin for external use
        self.argmax = argmax
        self.argmin = argmin
        
        # If self.beta is not 0 MAC output is still computed
        if self.beta >= 10e-5:
            Yb = F.conv2d(X, self.weight, stride=self.stride)
            if self.bias is not None:
                return (1-self.beta)*Y + self.beta*Yb + self.bias[None, :, None, None]
            return (1-self.beta)*Y + self.beta*Yb
        
        # No need to compute MAC output
        if self.bias is not None:
            return Y + self.bias[None, :, None, None]
        
        return Y
        
                   