import torch
import random
import mamtorch
import time
from reference import fullyconnected_reference, fullyconnected_backward_reference

device = torch.device("cuda:0")

print("__________________________")
print("Random functionality check")

n = random.randint(10, 1000)
m = random.randint(10, 1000)
l = random.randint(10, 1000)
a = torch.randn((n, m), dtype=torch.float32, device=device)
b = torch.randn((m, l), dtype=torch.float32, device=device)
c = torch.randn((n, l), dtype=torch.float32, device=device)
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
print(torch.ops.mamtorch_kernel_v2.fullyconnected(a, b, bias, beta)[0].shape)
print(torch.library.opcheck(torch.ops.mamtorch_kernel_v2.fullyconnected, (a, b, bias, beta)))
print("Functionality check")
res, argmax, argmin = torch.ops.mamtorch_kernel_v2.fullyconnected(a, b, bias, beta)
res_ref, argmax_ref, argmin_ref = fullyconnected_reference(a, b, bias, beta)
res_err = float(torch.max(torch.abs(res-res_ref)))
argmax_err = float(torch.max(torch.abs(argmax-argmax_ref)))
argmin_err = float(torch.max(torch.abs(argmin-argmin_ref)))
print(f"Errors {res_err} {argmax_err} {argmin_err}")

i = 0
for arg, argtrue in zip(argmax.flatten().cpu().numpy(), argmax.flatten().cpu().numpy()):
    if arg > m or arg < 0:
        i += 1
        print(arg, argtrue, i, end=" | ")

print()
i = 0
for arg, argtrue in zip(argmin.flatten().cpu().numpy(), argmin.flatten().cpu().numpy()):
    if arg > m or arg < 0:
        i += 1
        print(arg, argtrue, i, end=" | ")
print()

print()
print("Test kernel v2: dense")
print("Functionality check")
res = torch.ops.mamtorch_kernel_v2.dense(a, b)
res_ref = a@b
res_err = float(torch.max(torch.abs(res-res_ref)))
print(f"Errors {res_err}")

print()
print("Test kernel v2: fullyconnected_backwards")
print("Functionality check")
agrad_res, bgrad_res = torch.ops.mamtorch_kernel_v2.fullyconnected_backward(a, b, c, argmax, argmin, beta)
agrad_res_ref, bgrad_res_ref = fullyconnected_backward_reference(a, b, c, argmax, argmin, beta)
print(f"Max errors {torch.max(torch.abs(agrad_res-agrad_res_ref))} {torch.max(torch.abs(bgrad_res-bgrad_res_ref))}")

print()
print("Test kernel v3: fullyconnected")
print("Operation check")
print(torch.library.opcheck(torch.ops.mamtorch_kernel_v3.fullyconnected, (a, b, bias, beta)))
print("Functionality check")
res, argmax, argmin = torch.ops.mamtorch_kernel_v3.fullyconnected(a, b, bias, beta)
res_ref, argmax_ref, argmin_ref = fullyconnected_reference(a, b, bias, beta)
res_err = torch.abs(res-res_ref)
argmax_err = torch.abs(argmax-argmax_ref).to(torch.float)
argmin_err = torch.abs(argmin-argmin_ref).to(torch.float)
#for re, val in zip(argmax_err.flatten().cpu().numpy(), res_err.flatten().cpu().numpy()):
#    if re > 0:
#        print(f"arg_err {re:5}, val_err {val:5}")
print(f"Max error {float(torch.max(res_err))}")
print(f"Mean error {float(torch.mean(res_err))}")
print(f"Min error {float(torch.min(res_err))}")
#print(f"Mean arg errors {float(torch.mean(argmax_err))} {float(torch.mean(argmin_err))}")
total_occurrencies = argmax_err.shape[0]*argmax_err.shape[1]
print(f"Wrong arg total occurrencies {torch.sum(argmax_err>0)}/{total_occurrencies} {torch.sum(argmin_err>0)}/{total_occurrencies}")

i = 0
for arg, argtrue in zip(argmax.flatten().cpu().numpy(), argmax_ref.flatten().cpu().numpy()):
    if arg > m or arg < 0:
        i += 1
        print(arg, argtrue, i, end=" | ")

print()
i = 0
for arg, argtrue in zip(argmin.flatten().cpu().numpy(), argmin_ref.flatten().cpu().numpy()):
    if arg > m or arg < 0:
        i += 1
        print(arg, argtrue, i, end=" | ")
print()

print()
print("Test kernel v3: fullyconnected_fast")
print("Operation check")
print(torch.library.opcheck(torch.ops.mamtorch_kernel_v3.fullyconnected_fast, (a, b, bias, beta)))
print("Functionality check")
res = torch.ops.mamtorch_kernel_v3.fullyconnected_fast(a, b, bias, beta)
res_ref, _, _ = fullyconnected_reference(a, b, bias, beta)
res_err = float(torch.max(torch.abs(res-res_ref)))
print(f"Max errors {res_err}")

print()
print("Test kernel v3: fullyconnected_backwards")
print("Functionality check")
agrad_res, bgrad_res = torch.ops.mamtorch_kernel_v3.fullyconnected_backward(a, b, c, argmax, argmin, beta)
agrad_res_ref, bgrad_res_ref = fullyconnected_backward_reference(a, b, c, argmax, argmin, beta)
print(f"Max errors {torch.max(torch.abs(agrad_res-agrad_res_ref))} {torch.max(torch.abs(bgrad_res-bgrad_res_ref))}")


print()
print("Test kernel v4: fullyconnected")
print("Operation check")
print(torch.library.opcheck(torch.ops.mamtorch_kernel_v4.fullyconnected, (a, b)))
print("Functionality check")
res, argmax, argmin = torch.ops.mamtorch_kernel_v4.fullyconnected(a, b)
res_ref, argmax_ref, argmin_ref = fullyconnected_reference(a, b)
res_err = torch.abs(res-res_ref)
argmax_err = torch.abs(argmax-argmax_ref).to(torch.float)
argmin_err = torch.abs(argmin-argmin_ref).to(torch.float)
#for re, val in zip(argmax_err.flatten().cpu().numpy(), res_err.flatten().cpu().numpy()):
#    if re > 0:
#        print(f"arg_err {re:5}, val_err {val:5}")
print(f"Max error {float(torch.max(res_err))}")
print(f"Mean error {float(torch.mean(res_err))}")
print(f"Min error {float(torch.min(res_err))}")
#print(f"Mean arg errors {float(torch.mean(argmax_err))} {float(torch.mean(argmin_err))}")
total_occurrencies = argmax_err.shape[0]*argmax_err.shape[1]
print(f"Wrong arg total occurrencies {torch.sum(argmax_err>0)}/{total_occurrencies} {torch.sum(argmin_err>0)}/{total_occurrencies}")

i = 0
for arg, argtrue in zip(argmax.flatten().cpu().numpy(), argmax_ref.flatten().cpu().numpy()):
    if arg > m or arg < 0:
        i += 1
        print(arg, argtrue, i, end=" | ")

print()
i = 0
for arg, argtrue in zip(argmin.flatten().cpu().numpy(), argmin_ref.flatten().cpu().numpy()):
    if arg > m or arg < 0:
        i += 1
        print(arg, argtrue, i, end=" | ")
print()

print()
print("Test kernel v4: fullyconnected_fast")
print("Operation check")
print(torch.library.opcheck(torch.ops.mamtorch_kernel_v4.fullyconnected_fast, (a, b)))
print("Functionality check")
res = torch.ops.mamtorch_kernel_v4.fullyconnected_fast(a, b)
res_ref, _, _ = fullyconnected_reference(a, b)
res_err = float(torch.max(torch.abs(res-res_ref)))
print(f"Max errors {res_err}")

print()
print("Test kernel v4: fullyconnected_backwards")
print("Functionality check")
agrad_res, bgrad_res = torch.ops.mamtorch_kernel_v4.fullyconnected_backward(a, b, c, argmax, argmin)
agrad_res_ref, bgrad_res_ref = fullyconnected_backward_reference(a, b, c, argmax, argmin)
print(f"Max errors {torch.max(torch.abs(agrad_res-agrad_res_ref))} {torch.max(torch.abs(bgrad_res-bgrad_res_ref))}")


print("__________________________")
print("Benchmarks")

n, l, m = 128, 4096, 2048

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

print("Test kernel v3: fullyconnected_fast")
total_time = 0
for i in range(test_iterations):
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    bias = torch.randn((l,), device=device)
    beta = 0#random.uniform(0, 1)
    tic = time.perf_counter()
    res = torch.ops.mamtorch_kernel_v3.fullyconnected_fast(a, b, bias, beta)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")

print("Test kernel v3: fullyconnected_backward")
total_time = 0
for i in range(test_iterations):
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    c = torch.randn((n, l), device=device)
    argmax = torch.randint(0, m-1, (n, l), dtype=torch.int32, device=device)
    argmin = torch.randint(0, m-1, (n, l), dtype=torch.int32, device=device)
    beta = 0#random.uniform(0, 1)
    tic = time.perf_counter()
    agrad_res, bgrad_res = torch.ops.mamtorch_kernel_v3.fullyconnected_backward(a, b, c, argmax, argmin, beta)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")

print("Test kernel v4: fullyconnected")
total_time = 0
for i in range(test_iterations):
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    tic = time.perf_counter()
    res, argmax, argmin = torch.ops.mamtorch_kernel_v4.fullyconnected(a, b)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")

print("Test kernel v4: fullyconnected_fast")
total_time = 0
for i in range(test_iterations):
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    tic = time.perf_counter()
    res = torch.ops.mamtorch_kernel_v4.fullyconnected_fast(a, b)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")

print("Test kernel v4: fullyconnected_backward")
total_time = 0
for i in range(test_iterations):
    a = torch.randn((n, m), device=device)
    b = torch.randn((m, l), device=device)
    c = torch.randn((n, l), device=device)
    argmax = torch.randint(0, m-1, (n, l), dtype=torch.int32, device=device)
    argmin = torch.randint(0, m-1, (n, l), dtype=torch.int32, device=device)
    tic = time.perf_counter()
    agrad_res, bgrad_res = torch.ops.mamtorch_kernel_v4.fullyconnected_backward(a, b, c, argmax, argmin)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")
    