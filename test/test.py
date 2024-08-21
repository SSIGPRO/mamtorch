import torch
import random
import mamtorch
from reference import fullyconnected_reference

device = "cuda:0"

for i in range(10):
    n = random.randint(10, 1000)
    m = random.randint(10, 1000)
    l = random.randint(10, 1000)
    a = torch.randn((n, m), device=torch.device(device))
    b = torch.randn((m, l), device=torch.device(device))

    print(a.shape)
    print(b.shape)
    print(torch.library.opcheck(torch.ops.mamtorch_kernel_v1.fullyconnected, (a, b)))
    c, argmax, argmin = torch.ops.mamtorch_kernel_v1.fullyconnected(a, b)
    c1, argmax1, argmin1 = fullyconnected_reference(a, b)
    errc = float(torch.max(torch.abs(c-c1)))
    errargmax = float(torch.max(torch.abs(argmax-argmax1)))
    errargmin = float(torch.max(torch.abs(argmin-argmin1)))

    print(f"Iteration {i}: errors {errc} {errargmax} {errargmin}")