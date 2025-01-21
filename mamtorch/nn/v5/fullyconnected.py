import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F
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
        accblock_size: int = 1,
        relu_in: bool = False,
        vcon_steps: int = 0,
        vcon_type: str = 'linear',
        vcon_eps: float = 1e-3,
        beta_zero_gain: float = 1.0, # maximum output gain, that is beta_zero_gain*(1-beta)
        wdrop_rate: float = 0,
        drop_rate: float = 0,
        train_mam_only = False, # if True, during vanishing contribution, gradient is evaluated ONLY on the selected max and min interconnections
        store_args = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.splits = splits
        self.accblock_size = accblock_size
        self.relu_in = relu_in
        self.vcon_type = vcon_type
        self.vcon_steps = vcon_steps
        self.vcon_eps = vcon_eps
        self.beta_zero_gain = beta_zero_gain
        self.wdrop_rate = wdrop_rate
        self.drop_rate = drop_rate
        self.train_mam_only = train_mam_only
        self.store_args = store_args

        self.weight = Parameter(torch.empty(self.out_features, self.in_features, **factory_kwargs))
        if self.splits > 1:
            self.in_subfeatures = math.ceil(self.in_features/self.splits)
            self.in_subfeatures_last = self.in_features-self.in_subfeatures*(self.splits-1)
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        if vcon_steps > 0:
            self.beta = 1.0
            self.alpha = 0.0
        else:
            self.beta = 0.0 
            self.alpha = 1.0         
        
        self.max_selection_count = None
        self.min_selection_count = None
        self.argmax = None
        self.argmin = None
        
        self.norm_mean = None
        self.norm_var = None
        
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

        self.alpha = self.beta_zero_gain*(1.0 - self.beta)
        
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
        
    def set_fixated_norm(self, mean, var):
        self.norm_mean = mean
        self.norm_var = var
        
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
            if self.beta < 1:
                if self.training:
                    out = K.v5.fullyconnected(input, weight, self.accblock_size)[0]
                else:
                    if self.accblock_size > 1:
                        out = K.v5.fullyconnected(input, weight, self.accblock_size)[0]
                    else:
                        out = K.v5.fullyconnected_fast(input, weight) # fast computation is always exact
            else:
                out = torch.zeros((input.shape[0], weight.shape[1]), device=weight.device)
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
            if self.store_args:
                C_flat, argmax, argmin = K.v5.fullyconnected(input_flat, w, self.accblock_size)
                # store argmax and argmin for external usage
                self.argmax = argmax
                self.argmin = argmin
            else:
                C_flat = compute_noargs(input_flat, w)

        # multiply MAM term by a gain
        C_flat *= self.alpha
        
        # add MAC term
        if self.beta > 0:
            if self.train_mam_only:
                C_flat = C_flat + self.beta*F.linear(input_flat, self.weight.detach())
            else:
                C_flat = C_flat + self.beta*F.linear(input_flat, self.weight)

        # add bias
        if self.bias is not None:
            C_flat += self.bias.view(1, -1)  

        # restore output shape
        C_shape = input.shape[:-1] + (self.weight.size(0),) 
        C = C_flat.view(C_shape)

        # output dropout
        if self.drop_rate > 0:
            C = torch.nn.functional.dropout(C, self.drop_rate, self.training)

        return C

    def __repr__(self) -> str:
        description_string = f"MAM@FullyConnected(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        if self.vcon_steps > 0:
            description_string += f", vcon_steps={self.vcon_steps}, vcon_type={self.vcon_type}"
        if self.splits > 1:
            description_string += f", splits={self.splits}"
        if self.accblock_size > 1:
            description_string += f", accblock_size={self.accblock_size}"
        if self.beta_zero_gain != 1:
            description_string += f", beta_zero_gain={self.beta_zero_gain}"
        if self.relu_in:
            description_string += ", relu_in=True"
        description_string += ")"
        return description_string