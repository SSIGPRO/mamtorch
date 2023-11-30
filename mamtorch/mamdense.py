import torch
import mamtorchkernel
import math
import torch.nn.functional as F

class MAMDenseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        C, argmax, argmin = mamtorchkernel.mamdense_forward(A, B)
        ctx.save_for_backward(A, B, argmax, argmin)
        return C, argmax, argmin

    @staticmethod
    def backward(ctx, C_grad, argmax_grad, argmin_grad):
        A, B, argmax, argmin = ctx.saved_tensors
        A_grad, B_grad = mamtorchkernel.mamdense_backward(A, B, C_grad, argmax, argmin)
        return A_grad, B_grad, None, None
    

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
        
        self.max_selection_count = None
        self.min_selection_count = None
        self.argmax = None
        self.argmin = None
        
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
        if self.beta_decay == 'descending-parabola':
            self.beta = 1 - ((epoch+1)/self.beta_epochs)**2
            return
        if self.beta_decay == 'ascending-parabola':
            self.beta = 1 + (1/(self.beta_epochs**2)*((epoch+1)**2)) - ((2/self.beta_epochs)*(epoch+1))
            return
        
    def reset_selection_count(self):
        self.max_selection_count = torch.zeros_like(self.weight).to(torch.int32)
        self.min_selection_count = torch.zeros_like(self.weight).to(torch.int32)
        
    def update_selection_count(self):
        # Use the current values of self.argmax and self.argmin to update the selection count
        if self.argmax is None or self.argmin is None:
            raise("No argmax or argmin values have been evaluated yet.")
            
        if self.max_selection_count is None or self.min_selection_count is None:
            self.reset_selection_count()
            
        num_rows, num_cols = self.argmax.shape
        col_indices = torch.arange(num_cols).repeat(num_rows).to(self.weight.device)
        self.max_selection_count[col_indices, self.argmax.flatten()] += 1
        
        num_rows, num_cols = self.argmin.shape
        col_indices = torch.arange(num_cols).repeat(num_rows).to(self.weight.device)
        self.max_selection_count[col_indices, self.argmin.flatten()] += 1
        
    def forward(self, input):
        # flatten input to 2 dimensions
        input_flat = input.view(-1, input.size()[-1])
        
        # apply mam
        C, argmax, argmin = MAMDenseFunction.apply(input_flat, self.weight.T.contiguous())
        
        # store argmax and argmin for external usage
        self.argmax = argmax
        self.argmin = argmin
        
        # get output shape
        C_shape = list(input.shape[:-1]) + [self.weight.size(0)] 
        
        #If self.beta is not 0 MAC output is still computed
        if self.beta >= 10e-5:
            D = F.linear(input, self.weight)
            if self.bias is not None:
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
        
                   