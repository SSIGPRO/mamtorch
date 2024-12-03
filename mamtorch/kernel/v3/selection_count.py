import torch
from torch import Tensor

__all__ = ["selection_count"]

library_name = "mamtorch_kernel_v3"
K = torch.ops.mamtorch_kernel_v3


def selection_count(a: Tensor, argmax: Tensor, argmin: Tensor) -> list[Tensor]:
    return K.selection_count.default(a, argmax, argmin)
