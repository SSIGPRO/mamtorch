import torch
import random
import mamtorch
from reference import fullyconnected_reference

device = torch.device("cuda:0")
test_iterations = 1

for i in range(test_iterations):
    n = random.randint(10, 1000)
    m = random.randint(10, 1000)
    l = random.randint(10, 1000)
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    bias = torch.randn((l,), device=device)
    beta = random.uniform(0, 1)
    
    print("Shape of A:", a.shape)
    print("Shape of B:", b.shape)
    print("Shape of bias:", bias.shape)
    print("Beta value:", beta)

    print("Test kernel v1")
    print("Operation check")
    print(torch.library.opcheck(torch.ops.mamtorch_kernel_v1.fullyconnected, (a, b)))
    print("Functionality check")
    res, argmax, argmin = torch.ops.mamtorch_kernel_v1.fullyconnected(a, b)
    res_ref, argmax_ref, argmin_ref = fullyconnected_reference(a, b)
    res_err = float(torch.max(torch.abs(res-res_ref)))
    argmax_err = float(torch.max(torch.abs(argmax-argmax_ref)))
    argmin_err = float(torch.max(torch.abs(argmin-argmin_ref)))
    print(f"Iteration {i}: errors {res_err} {argmax_err} {argmin_err}")

    print("Test kernel v2")
    print("Operation check")
    print(torch.library.opcheck(torch.ops.mamtorch_kernel_v2.fullyconnected, (a, b, bias, beta)))
    print("Functionality check")
    res, argmax, argmin = torch.ops.mamtorch_kernel_v2.fullyconnected(a, b, bias, beta)
    res_ref, argmax_ref, argmin_ref = fullyconnected_reference(a, b, bias, beta)
    res_err = float(torch.max(torch.abs(res-res_ref)))
    argmax_err = float(torch.max(torch.abs(argmax-argmax_ref)))
    argmin_err = float(torch.max(torch.abs(argmin-argmin_ref)))
    print(f"Iteration {i}: errors {res_err} {argmax_err} {argmin_err}")