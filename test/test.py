import torch
import random
import mamtorch
import time
from reference import fullyconnected_reference

device = torch.device("cuda:1")

print("__________________________")
print("Random functionality check")

n = random.randint(10, 1000)
m = random.randint(10, 1000)
l = random.randint(10, 1000)
a = torch.randn((n, m), dtype=torch.float32, device=device)
b = torch.randn((m, l), dtype=torch.float32, device=device)
bias = torch.randn((l,), dtype=torch.float32, device=device)
beta = random.uniform(0, 1)
print("Shape of A:", a.shape)
print("Shape of B:", b.shape)
print("Shape of bias:", bias.shape)
print("Beta value:", beta)

print()
print("Test kernel v1: fullyconnected")
print("Operation check")
print(torch.library.opcheck(torch.ops.mamtorch_kernel_v1.fullyconnected, (a, b)))
print("Functionality check")
res, argmax, argmin = torch.ops.mamtorch_kernel_v1.fullyconnected(a, b)
res_ref, argmax_ref, argmin_ref = fullyconnected_reference(a, b)
res_err = float(torch.max(torch.abs(res-res_ref)))
argmax_err = float(torch.max(torch.abs(argmax-argmax_ref)))
argmin_err = float(torch.max(torch.abs(argmin-argmin_ref)))
print(f"Errors {res_err} {argmax_err} {argmin_err}")

print()
print("Test kernel v2: fullyconnected")
print("Operation check")
print(torch.library.opcheck(torch.ops.mamtorch_kernel_v2.fullyconnected, (a, b, bias, beta)))
print("Functionality check")
res, argmax, argmin = torch.ops.mamtorch_kernel_v2.fullyconnected(a, b, bias, beta)
res_ref, argmax_ref, argmin_ref = fullyconnected_reference(a, b, bias, beta)
res_err = float(torch.max(torch.abs(res-res_ref)))
argmax_err = float(torch.max(torch.abs(argmax-argmax_ref)))
argmin_err = float(torch.max(torch.abs(argmin-argmin_ref)))
print(f"Errors {res_err} {argmax_err} {argmin_err}")

print()
print("Test kernel v2: dense")
print("Functionality check")
res = torch.ops.mamtorch_kernel_v2.dense(a, b)
res_ref = a@b
res_err = float(torch.max(torch.abs(res-res_ref)))
print(f"Errors {res_err}")

print()
print("Test kernel v3: fullyconnected")
print("Operation check")
print(torch.library.opcheck(torch.ops.mamtorch_kernel_v3.fullyconnected, (a, b, bias, beta)))
print("Functionality check")
res, argmax, argmin = torch.ops.mamtorch_kernel_v3.fullyconnected(a, b, bias, beta)
res_ref, argmax_ref, argmin_ref = fullyconnected_reference(a, b, bias, beta)
res_err = float(torch.mean(torch.abs(res-res_ref)))
argmax_err = float(torch.mean(torch.abs(argmax-argmax_ref).to(torch.float)))
argmin_err = float(torch.mean(torch.abs(argmin-argmin_ref).to(torch.float)))
#print(f"Max errors {res_err} {argmax_err} {argmin_err}")
print(f"Mean errors {res_err} {argmax_err} {argmin_err}")
#print(torch.abs(res-res_ref))
#print(torch.abs(res-res_ref)/torch.abs(res_ref))

print()
print("__________________________")
print("Benchmarks")

n, l, m = 128, 1024, 1024

test_iterations = 1000

print("Torch matmul")
total_time = 0
for i in range(test_iterations):
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    tic = time.perf_counter()
    res = torch.matmul(a, b)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")

print("Test kernel v1: fullyconnected")
total_time = 0
for i in range(test_iterations):
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    tic = time.perf_counter()
    res, argmax, argmin = torch.ops.mamtorch_kernel_v1.fullyconnected(a, b)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")
    

print("Test kernel v2: fullyconnected")
total_time = 0
for i in range(test_iterations):
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    bias = torch.randn((l,), device=device)
    beta = 0#random.uniform(0, 1)
    tic = time.perf_counter()
    res, argmax, argmin = torch.ops.mamtorch_kernel_v2.fullyconnected(a, b, bias, beta)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")

print("Test kernel v2: dense")
total_time = 0
for i in range(test_iterations):
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    tic = time.perf_counter()
    res = torch.ops.mamtorch_kernel_v2.dense(a, b)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")


print("Test kernel v3: fullyconnected")
total_time = 0
for i in range(test_iterations):
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    bias = torch.randn((l,), device=device)
    beta = 0#random.uniform(0, 1)
    tic = time.perf_counter()
    res, argmax, argmin = torch.ops.mamtorch_kernel_v3.fullyconnected(a, b, bias, beta)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")
    