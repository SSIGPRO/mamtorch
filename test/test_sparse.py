import torch
import random
import mamtorch
import time
import numpy as np
import pandas as pd

device = torch.device("cuda:3")

print("__________________________")
print("Random functionality check")

def generate_random_sparse_tensor(n, m, sparse_rate, device):
    a = torch.randn((n, m), dtype=torch.float32, device=device)
    a = a*torch.bernoulli(torch.full_like(a, 1-sparse_rate))
    return a

np.set_printoptions(precision=2, suppress=True, edgeitems=40, linewidth=400)

sparse_rate = 0.9
sparsity_format = torch.sparse_csr

n = random.randint(10, 1000)
l = random.randint(10, 1000)
m = random.randint(10, 1000)
a = generate_random_sparse_tensor(n, l, sparse_rate, device)
b = torch.randn((l, m), dtype=torch.float32, device=device)
print("Shape of A:", a.shape)
print("Shape of B:", b.shape)

print()
print("Result generation")
res_ref = a@b

print("Conversion to COO sparse matrices")
asparse = a.to_sparse(layout=sparsity_format)
print(asparse)

print("Test against torch.sparse")
res = torch.sparse.mm(asparse, b).to_dense()
print(f"Maximum error = {torch.max(torch.abs(res-res_ref)).cpu().numpy()}")

print("Test against mamtorch.cusparsemm")
res = torch.ops.mamtorch_kernel_sparsev1.cusparsemm(asparse, b)
print(f"Maximum error = {torch.max(torch.abs(res-res_ref)).cpu().numpy()}")

print()
print("__________________________")
print("Benchmarks")


sizes = [128, 256, 512, 1024, 2048, 4096]
sparse_rates = [0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
sparsity_formats = [torch.sparse_coo, torch.sparse_csr]

test_iterations = 1000

data = pd.DataFrame()

for size in sizes:
    n, l, m = size, size, size
    for sparse_rate in sparse_rates:
        for sparsity_format in sparsity_formats:
            print(f"Test: matrix size = {size}, sparse rate={sparse_rate}, sparsity format={sparsity_format}")

            print("Torch matmul")
            total_time = 0
            for i in range(test_iterations):
                a = torch.randn((n, l), dtype=torch.float32, device=device)
                b = generate_random_sparse_tensor(l, m, sparse_rate, device)
                tic = time.perf_counter()
                res = torch.matmul(a, b)
                torch.cuda.synchronize()
                toc = time.perf_counter()
                total_time += toc-tic
            dense_time = total_time/test_iterations
            print(f"Average time {dense_time*1000} ms")

            print("cusparse matmul")
            total_time = 0
            for i in range(test_iterations):
                asparse = generate_random_sparse_tensor(l, m, sparse_rate, device).to_sparse(layout=sparsity_format)
                b = torch.randn((n, l), dtype=torch.float32, device=device)
                tic = time.perf_counter()
                res = torch.ops.mamtorch_kernel_sparsev1.cusparsemm(asparse, b)
                torch.cuda.synchronize()
                toc = time.perf_counter()
                total_time += toc-tic
            sparse_time = total_time/test_iterations
            print(f"Average time {sparse_time*1000} ms")

            new_datarow = pd.DataFrame({
                "square matrices size": [size],
                "sparsity rate": [sparse_rate],
                "sparsity format": [sparsity_format],
                "dense matmul time": [dense_time],
                "sparse matmul time": [sparse_time],
            })

            data = pd.concat([data, new_datarow], ignore_index=True)

            data.to_csv("sparse_measures.csv")
    
# print("Torch.sparse matmul")
# total_time = 0
# for i in range(test_iterations):
#     a = torch.randn((n, l), dtype=torch.float32, device=device)
#     sparseb = generate_random_sparse_tensor(l, m, sparse_rate, device).to_sparse(layout=sparsity_format)
#     tic = time.perf_counter()
#     res = torch.sparse.mm(a, sparseb)
#     torch.cuda.synchronize()
#     toc = time.perf_counter()
#     total_time += toc-tic
# print(f"Average time {total_time/test_iterations*1000} ms")