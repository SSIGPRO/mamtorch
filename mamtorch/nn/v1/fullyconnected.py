import torch
import math
import torch.nn.functional as F
from ...kernel import v1 as K

__all__ = ["FullyConnected"]

class FullyConnected(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, vcon_steps=0, vcon_type='linear', init_mode='uniform', **kwargs):
        super(FullyConnected, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.use_bias = bias
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.vcon_type = vcon_type
        self.vcon_steps = vcon_steps
        self.init_mode = init_mode
        
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        if vcon_steps > 0:
            self.beta = 1.0
        else:
            self.beta = 0.0
        
        self.max_selection_count = None
        self.min_selection_count = None
        self.argmax = None
        self.argmin = None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize using xavier uniform
        gain = 1.0
        #a = gain*math.sqrt(6/(self.in_features+self.out_features))
        if self.init_mode == "uniform":
            a = gain*math.sqrt(6/(2+self.out_features))
            torch.nn.init.uniform_(self.weight, -a, a)
        elif self.init_mode == "normal":
            a = gain*math.sqrt(2/(2+self.out_features))
            torch.nn.init.normal_(self.weight, std=a)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
            
    def vcon_step(self):
        if self.vcon_steps < 0:
            raise Exception("Invalid value for vcon_steps. Please use 0 or a positive integer.")
        elif self.vcon_steps == 0:
            return

        if self.vcon_type == 'linear':
            self.beta -= 1/self.vcon_steps
        #elif self.beta_decay == 'descending-parabola':
        #    self.beta = 1 - (epoch/self.beta_epochs)**2
        #elif self.beta_decay == 'ascending-parabola':
        #    self.beta = 1 + (1/(self.beta_epochs**2)*(epoch**2)) - ((2/self.beta_epochs)*epoch)
        else:
            raise Exception(f"Vanishing contribution type '{self.vcon_type}' is not supported")
        
        if self.beta <= 1e-8:
            self.beta = 0
        
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
        C, argmax, argmin = K.fullyconnected(input_flat, self.weight.T.contiguous())
        
        # store argmax and argmin for external usage
        self.argmax = argmax
        self.argmin = argmin
        
        # get output shape
        C_shape = input.shape[:-1] + (self.weight.size(0),) 
        
        #If self.beta is not 0 MAC output is still computed
        if self.beta > 0:
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

    def __repr__(self):
        return f"MAM@FullyConnected(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}, vcon_steps={self.vcon_steps}, vcon_type={self.vcon_type})"
                   