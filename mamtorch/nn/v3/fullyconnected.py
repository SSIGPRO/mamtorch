import torch
from torch import Tensor
from torch.nn import Module, Parameter
import math
from ... import kernel as K

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
        splits: int = 1,
        relu_in: bool = False,
        vcon_steps: int = 0,
        vcon_type: str = 'linear',
        vcon_eps: float = 1e-3,
        wdrop_rate: float = 0,
        drop_rate: float = 0,
        compute_exact = False, # if False, use the approximate computing fast kernel (K.v3), if True, use the exact slower kernel (K.v2)
        store_args = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.splits = splits
        self.relu_in = relu_in
        self.vcon_type = vcon_type
        self.vcon_steps = vcon_steps
        self.vcon_eps = vcon_eps
        self.wdrop_rate = wdrop_rate
        self.drop_rate = drop_rate
        self.compute_exact = compute_exact
        self.store_args = store_args

        self.weight = Parameter(torch.empty(self.out_features, self.in_features, **factory_kwargs))
        if self.splits > 1:
            self.in_subfeatures = math.ceil(self.in_features/self.splits)
            self.in_subfeatures_last = self.in_features-self.in_subfeatures*(self.splits-1)
            print(f"Splits are: {self.in_subfeatures} x {self.splits-1} + {self.in_subfeatures_last} = {self.in_features}")
        
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
        elif self.vcon_type == 'exponential':
            gamma = math.exp(math.log(self.vcon_eps)/self.vcon_steps)
            self.beta *= gamma
        else:
            raise Exception(f"vcon_type '{self.vcon_type}' is not supported.")

        if self.beta <= self.vcon_eps:
            self.beta = 0
        
    def reset_selection_count(self) -> None:
        if self.store_args:
            self.max_selection_count = torch.zeros_like(self.weight).to(torch.int32)
            self.min_selection_count = torch.zeros_like(self.weight).to(torch.int32)
        else:
            raise Exception("MAM layer has not been set to store max and min arguments (store_args is False). Do not use reset_selection_count()")
        
    def update_selection_count(self) -> None:
        if self.store_args:
            # Use the current values of self.argmax and self.argmin to update the selection count
            if self.argmax is None or self.argmin is None:
                raise Exception("No argmax or argmin values have been evaluated yet.")
                
            if self.max_selection_count is None or self.min_selection_count is None:
                self.reset_selection_count()
                
            num_rows, num_cols = self.argmax.shape
            col_indices = torch.arange(num_cols).repeat(num_rows).to(self.weight.device)
            self.max_selection_count[col_indices, self.argmax.flatten()] += 1
            
            num_rows, num_cols = self.argmin.shape
            col_indices = torch.arange(num_cols).repeat(num_rows).to(self.weight.device)
            self.max_selection_count[col_indices, self.argmin.flatten()] += 1
        else:
            raise Exception("MAM layer has not been set to store max and min arguments (store_args is False). Do not use update_selection_count()")
        
    def forward(self, input: Tensor) -> Tensor:
        # apply relu to input if requested
        if self.relu_in:
            input = torch.nn.functional.relu(input)

        # flatten input to 2 dimensions
        input_flat = input.view(-1, input.size()[-1])
        
        w = self.weight.T.contiguous()
        
        if self.training:
            if self.wdrop_rate != 0:
                if self.wdrop_rate > 0 and self.wdrop_rate < 1:
                    w = torch.bernoulli(torch.full_like(w, 1-self.wdrop_rate))*w
                else: 
                    raise Exception("wdrop_rate must be between 0 and 1.")            

        # default noargs computation function
        def compute_noargs(input, weight):
            tbias = torch.zeros(weight.size(-1), device=input.device)
            if self.training:
                if self.compute_exact:
                    out = K.v2.fullyconnected(input, weight, tbias, self.beta)[0]
                else:
                    out = K.v3.fullyconnected(input, weight, tbias, self.beta)[0]
            else:
                out = K.v3.fullyconnected_fast(input, weight, tbias, self.beta) # fast computation is always exact
            return out
        
        if self.splits > 1:
            input_flat_split = input_flat.narrow(-1, 0, self.in_subfeatures) # cut the first input slice
            w_split = w.narrow(0, 0, self.in_subfeatures) # cut the first weight slice
            C_flat = compute_noargs(input_flat_split, w_split)

            for i in range(1, self.splits-1):
                input_flat_split = input_flat.narrow(-1, i*self.in_subfeatures, self.in_subfeatures) # cut the i-th input slice
                w_split = w.narrow(0, i*self.in_subfeatures, self.in_subfeatures) # cut the i-th weight slice
                C_flat += compute_noargs(input_flat_split, w_split)
                
            input_flat_split = input_flat.narrow(-1, self.in_features-self.in_subfeatures_last, self.in_subfeatures_last) # cut the last input slice
            w_split = w.narrow(0, self.in_features-self.in_subfeatures_last, self.in_subfeatures_last) # cut the last weight slice
            C_flat += compute_noargs(input_flat_split, w_split)
        else:
            tbias = torch.zeros(w.size(-1), device=input.device)
            if self.store_args:
                if self.compute_exact:
                    C_flat, argmax, argmin = K.v2.fullyconnected(input_flat, w, tbias, self.beta)
                else:
                    C_flat, argmax, argmin = K.v3.fullyconnected(input_flat, w, tbias, self.beta)
                # store argmax and argmin for external usage
                self.argmax = argmax
                self.argmin = argmin
            else:
                C_flat = compute_noargs(input_flat, w)
         
        # restore output shape
        if self.bias is not None: # Add bias
            C_flat += self.bias.view(1, -1)  
        C_shape = input.shape[:-1] + (self.weight.size(0),) 
        C = C_flat.view(C_shape)

        # output dropout
        if self.drop_rate > 0:
            C = torch.nn.functional.dropout(C, self.drop_rate, self.training)

        return C

    def __repr__(self) -> str:
        return f"MAM@FullyConnected(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, splits={self.splits}, relu_in={self.relu_in}, vcon_steps={self.vcon_steps}, vcon_type={self.vcon_type})"