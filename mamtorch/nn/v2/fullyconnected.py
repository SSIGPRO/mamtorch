import torch
from torch import Tensor
from torch.nn import Module, Parameter
import math
from ...kernel import v2 as K

__all__ = ["FullyConnected"]

class FullyConnected(Module):

    __name__ = "FullyConnected"
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features:int,
        bias: bool = True,
        vcon_steps: int = 0,
        vcon_type: str = 'linear',
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        self.vcon_type = vcon_type
        self.vcon_steps = vcon_steps
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
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
        
    def reset_parameters(self) -> None:
        # Initialize weight and bias here
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
            
    def vcon_step(self) -> None:
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

        if self.beta <= 1e-8:
            self.beta = 0
        
    def reset_selection_count(self) -> None:
        self.max_selection_count = torch.zeros_like(self.weight).to(torch.int32)
        self.min_selection_count = torch.zeros_like(self.weight).to(torch.int32)
        
    def update_selection_count(self) -> None:
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
        
    def forward(self, input: Tensor) -> Tensor:
        # flatten input to 2 dimensions
        input_flat = input.view(-1, input.size()[-1])
        
        # apply mam
        if self.bias is not None:
            C_flat, argmax, argmin = K.fullyconnected(input_flat, self.weight.T.contiguous(), self.bias, self.beta)
        else:
            tbias = torch.zeros(self.out_features)
            C_flat, argmax, argmin = K.fullyconnected(input_flat, self.weight.T.contiguous(), tbias, self.beta)
        
        # store argmax and argmin for external usage
        self.argmax = argmax
        self.argmin = argmin
        
        # restore output shape
        C_shape = input.shape[:-1] + (self.weight.size(0),) 
        C = C_flat.view(C_shape)

        return C

    def __repr__(self) -> str:
        return f"MAM@FullyConnected(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, vcon_steps={self.vcon_steps}, vcon_type={self.vcon_type})"