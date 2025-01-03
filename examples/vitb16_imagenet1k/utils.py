import random
import math
import numpy as np
import torch
import torch.nn as nn
import mamtorch
from transformers.models.vit.modeling_vit import ViTConfig, VIT_ATTENTION_CLASSES, ViTIntermediate, ViTOutput, Optional, Union, Tuple

# Base functions

def set_seed(seed: int):
    # Set the Python random seed
    random.seed(seed)
    # Set the NumPy random seed
    np.random.seed(seed)
    # Set the PyTorch random seed
    torch.manual_seed(seed)
    # If you are using GPUs, you also need to set the seed for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        
def generate_paths(qkw: bool, input: bool, intermediate: bool, output: bool, layers_number=12) -> list[str]:
    list_paths=[]
    if layers_number >= 0:
        if qkw:
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.query' for i in range(layers_number)]
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.key' for i in range(layers_number)]
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.value' for i in range(layers_number)]
        if input:
            list_paths += [f'vit.encoder.layer.{i}.attention.output.dense' for i in range(layers_number)]
        if intermediate:
            list_paths += [f'vit.encoder.layer.{i}.intermediate.dense' for i in range(layers_number)]
        if output:
            list_paths += [f'vit.encoder.layer.{i}.output.dense' for i in range(layers_number)]
    else:
        if qkw:
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.query' for i in range(12+layers_number, 12)]
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.key' for i in range(12+layers_number, 12)]
            list_paths += [f'vit.encoder.layer.{i}.attention.attention.value' for i in range(12+layers_number, 12)]
        if input:
            list_paths += [f'vit.encoder.layer.{i}.attention.output.dense' for i in range(12+layers_number, 12)]
        if intermediate:
            list_paths += [f'vit.encoder.layer.{i}.intermediate.dense' for i in range(12+layers_number, 12)]
        if output:
            list_paths += [f'vit.encoder.layer.{i}.output.dense' for i in range(12+layers_number, 12)]  
    
    return list_paths

def get_layers(model: nn.Module, layers_str: list[str], split_attributes=False) -> list[torch.tensor]:
    paths = [layer.split('.') for layer in layers_str]
    if split_attributes:
        attributes = [layer[-1] for layer in paths]
        paths = [layer[:-1] for layer in paths]

    layers_list = []
    for layer in paths:
        tmp_layer = model
        for sub_layer in layer:
            tmp_layer = getattr(tmp_layer, sub_layer)
        layers_list.append(tmp_layer)
    if split_attributes:
        return layers_list, attributes
    else:
        return layers_list
           
def add_mam_to_ViT(model, layers_str, vcon_steps=0, layer_splits=1, vcon_type='linear', train_mam_only=False, beta_zero_gain=1.0):
    layers_list, attributes = get_layers(model, layers_str, True)
    for layer, attr in zip(layers_list, attributes):
        layer_attr = getattr(layer, attr)
        tmp_weight = layer_attr.weight.data
        tmp_bias = layer_attr.bias.data
        mam = mamtorch.nn.FullyConnected(layer_attr.in_features, layer_attr.out_features, bias=True, vcon_steps=vcon_steps, vcon_type=vcon_type, splits=layer_splits, train_mam_only=train_mam_only, beta_zero_gain=beta_zero_gain)
        mam.weight.data = tmp_weight
        mam.bias.data = tmp_bias
        setattr(layer, attr, mam)
    return model

class EmptyLayer(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, vcon_steps=0, vcon_type='linear', vcon_eps=1e-3):
        super().__init__(in_features, out_features, bias)
        self.vcon_steps = vcon_steps
        self.vcon_type = vcon_type
        self.vcon_eps = vcon_eps
        if vcon_steps == 0:
            self.beta = 0.0
        else:
            self.beta = 1.0

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)*self.beta
        if self.beta == 0:
            output = torch.zeros((*input.shape[:-1], self.out_features)).to("cuda:0")
        return output
    
def add_empty_to_ViT(model, layers_str, vcon_steps=0, vcon_type='linear'):
    layers_list, attributes = get_layers(model, layers_str, True)
    for layer, attr in zip(layers_list, attributes):
        layer_attr = getattr(layer, attr)
        tmp_weight = layer_attr.weight.data
        tmp_bias = layer_attr.bias.data
        empty = EmptyLayer(layer_attr.in_features, layer_attr.out_features, bias=True, vcon_steps=vcon_steps, vcon_type=vcon_type)
        empty.weight.data = tmp_weight
        empty.bias.data = tmp_bias
        setattr(layer, attr, empty)
    return model
    
# MODIFIED VIT OUTPUT CLASS FROM TRANSFORMERS PACKAGE
class ViTOutputNoSkip(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        #hidden_states = hidden_states + input_tensor

        return hidden_states
    
def generate_noskip_paths(layers_number=12) -> list[str]:
    list_paths=[]
    if layers_number >= 0:
        list_paths += [f'vit.encoder.layer.{i}.output' for i in range(layers_number)]
    else:
        list_paths += [f'vit.encoder.layer.{i}.output' for i in range(12+layers_number, 12)]
    
    return list_paths
    
def apply_noskip(model, layers_str):
    layers_list, attributes = get_layers(model, layers_str, True)
    for layer, attr in zip(layers_list, attributes):
        layer_attr = getattr(layer, attr)
        tmp_config = model.config
        newViTLayer = ViTOutputNoSkip(tmp_config)
        setattr(layer, attr, newViTLayer)
    return model

# MODIFIED VIT OUTPUT CLASS FROM TRANSFORMERS PACKAGE
class ViTLayerNewSkip(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VIT_ATTENTION_CLASSES[config._attn_implementation](config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        #hidden_states = attention_output + hidden_states
        intermediate_input = attention_output + hidden_states # DETACH THIS FROM THE OUTPUT

        # in ViT, layernorm is also applied after self-attention
        #layer_output = self.layernorm_after(hidden_states)
        layer_output = self.layernorm_after(intermediate_input)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states) # RESIDUAL CONNECTION TAKES FROM THE INPUT (NOT FROM THE ATTENTION OUTPUT)

        outputs = (layer_output,) + outputs

        return outputs
    
def generate_newskip_paths(layers_number=12) -> list[str]:
    list_paths=[]
    if layers_number >= 0:
        list_paths += [f'vit.encoder.layer.{i}' for i in range(layers_number)]
    else:
        list_paths += [f'vit.encoder.layer.{i}' for i in range(12+layers_number, 12)]
    
    return list_paths
    
def apply_newskip(model, layers_str):
    layers_list, attributes = get_layers(model, layers_str, True)
    for layer, attr in zip(layers_list, attributes):
        layer_attr = getattr(layer, attr)
        tmp_config = model.config
        newViTLayer = ViTLayerNewSkip(tmp_config)
        setattr(layer, attr, newViTLayer)
    return model

def combine_config(config_dict_list):
    combined_config = {}
    for config in config_dict_list.values():
        combined_config.update(config)
    return combined_config

def convert_string(value):
    # Handle boolean conversion
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    
    # Try integer conversion
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float conversion
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string if no other conversion is valid
    return value