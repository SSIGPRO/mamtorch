import torch
import random
import mamtorch
import time

device = torch.device("cuda:2")

print("__________________________")
print("Random functionality check")

def generate_random_sparse_tensor(n, m, sparse_rate, device):
    a = torch.randn((n, m), dtype=torch.float32, device=device)
    a = a*torch.bernoulli(torch.full_like(a, 1-sparse_rate))
    return a

sparse_rate = 0.999

n = random.randint(10, 1000)
m = random.randint(10, 1000)
l = random.randint(10, 1000)
a = generate_random_sparse_tensor(n, l, sparse_rate, device)
b = generate_random_sparse_tensor(l, m, sparse_rate, device)
print("Shape of A:", a.shape)
print("Shape of B:", b.shape)

print()
print("Result generation")
res_ref = a@b

print("Conversion to COO sparse matrices")
asparse = a.to_sparse()
bsparse = b.to_sparse()

print("Test against torch.sparse")
res = torch.sparse.mm(asparse, bsparse).to_dense()
print(f"Maximum error = {torch.max(torch.abs(res-res_ref)).cpu().numpy()}")

print("Test against mamtorch.unstructured_cusparse")
res = torch.ops.mamtorch_kernel_sparsev1.unstructured_cusparse(asparse, bsparse).to_dense()
print(f"Maximum error = {torch.max(torch.abs(res-res_ref)).cpu().numpy()}")

print()
print("__________________________")
print("Benchmarks")

n, l, m = 1024, 1024, 1024

test_iterations = 1000

print("Torch matmul")
total_time = 0
for i in range(test_iterations):
    a = generate_random_sparse_tensor(n, l, sparse_rate, device)
    b = generate_random_sparse_tensor(l, m, sparse_rate, device)
    tic = time.perf_counter()
    res = torch.matmul(a, b)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")

print("Torch.sparse matmul")
total_time = 0
for i in range(test_iterations):
    sparsea = generate_random_sparse_tensor(n, l, sparse_rate, device).to_sparse()
    sparseb = generate_random_sparse_tensor(l, m, sparse_rate, device).to_sparse()
    tic = time.perf_counter()
    res = torch.sparse.mm(sparsea, sparseb)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    total_time += toc-tic
print(f"Average time {total_time/test_iterations*1000} ms")
    