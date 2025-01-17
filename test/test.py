import torch
import random
import mamtorch
import time
from reference import fullyconnected_reference, fullyconnected_backward_reference

device = torch.device("cuda:0")

def test_function(func, ref_func, args, title="", print_outputs=False):
    print()
    print(title)
    print("Operation check")
    print(torch.library.opcheck(func, args))
    print("Functionality check")
    res = func(*args)
    res_ref = ref_func(*args)

    res = res if isinstance(res, (list, tuple)) else (res,)
    res_ref = res_ref if isinstance(res_ref, (list, tuple)) else (res_ref,)

    if print_outputs:
        print("Outputs")
        for r in res:
            print()
            print(r)
        print("Expected outputs")
        for r in res_ref:
            print()
            print(r)

    print("Maximum errors: ", end="")
    for i in range(len(res)):
        err = float(torch.max(torch.abs(res[i]-res_ref[i])))
        print(err, end=" ")
    print()
    print("Average errors: ", end="")
    for i in range(len(res)):
        err = float(torch.mean(torch.abs(res[i]-res_ref[i]).to(torch.float32)))
        print(err, end=" ")
    print()
    print("Minimum errors: ", end="")
    for i in range(len(res)):
        err = float(torch.min(torch.abs(res[i]-res_ref[i])))
        print(err, end=" ")
    print()

test_iterations = 1000

def benchmark_kernel(func, arg_sizes, arg_types, arg_mins=None, arg_maxs=None, arg_value=None, title=""):
    print()
    print(title)
    print("Elapsed time benchmark")
    total_time = 0
    for i in range(test_iterations):
        args = []
        for i in range(len(arg_sizes)):
            if arg_value is not None and arg_value[i] is not None:
                args += [arg_value[i]]
            elif arg_sizes[i] == 1:
                args += [0]
            else:
                if arg_types[i] == "int":
                    args += [torch.randint(arg_mins[i], arg_maxs[i], arg_sizes[i], dtype=torch.int32, device=device)]
                else:
                    args += [torch.randn(arg_sizes[i], device=device)]
        tic = time.perf_counter()
        res = func(*args)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        total_time += toc-tic
    print(f"Average time {total_time/test_iterations*1000} ms")
    

print("__________________________")
print("Random functionality check")

print_outputs = False
benchmarks = False
n = random.randint(16, 1000)
m = random.randint(16, 1000)
l = random.randint(16, 1000)
#n, m, l = 4, 4, 4
a = torch.randn((n, m), dtype=torch.float32, device=device)
b = torch.randn((m, l), dtype=torch.float32, device=device)
c = torch.randn((n, l), dtype=torch.float32, device=device)
#a = torch.tensor([[1, 2, 3, 4, 2, 4, 6, 8], [2, 4, 6, 8, 1, 2, 3, 4]], dtype=torch.float32, device=device)
#b = torch.tensor([[1, -1],[2, -2],[3, -3],[4, -4],[-1, 1],[-2, 2],[-3, 3],[-4, 4]], dtype=torch.float32, device=device)
argmax = torch.randint(0, m, (n, l), dtype=torch.int32, device=device)
argmin = torch.randint(0, m, (n, l), dtype=torch.int32, device=device)
print("Shape of A:", a.shape)
print("Shape of B:", b.shape)
if print_outputs:
    print("a matrix")
    print(a)
    print("b matrix")
    print(b)
    print("c matrix")
    print(c)
    print("argmax matrix")
    print(argmax)
    print("argmin matrix")
    print(argmin)

# kernel v1
# test_function(torch.ops.mamtorch_kernel_v1.fullyconnected, fullyconnected_reference, (a, b), "Test kernel v1: fullyconnected", print_outputs)

# # kernel v4
# test_function(torch.ops.mamtorch_kernel_v4.fullyconnected, fullyconnected_reference, (a, b), "Test kernel v4: fullyconnected", print_outputs)
# test_function(torch.ops.mamtorch_kernel_v4.fullyconnected_fast, fullyconnected_reference, (a, b), "Test kernel v4: fullyconnected_fast", print_outputs)
# test_function(torch.ops.mamtorch_kernel_v4.fullyconnected_backward, fullyconnected_backward_reference, (a, b, c, argmax, argmin), "Test kernel v4: fullyconnected_backward", print_outputs)

# kernel v5
# accblock_size_list = [1, 4, 8, 16, 32, 64]
# for accblock_size in accblock_size_list:
#     test_function(torch.ops.mamtorch_kernel_v5.fullyconnected, fullyconnected_reference, (a, b, accblock_size), f"Test kernel v5: fullyconnected (accblock_size = {accblock_size})", print_outputs)

test_function(torch.ops.mamtorch_kernel_v5.fullyconnected_backward, fullyconnected_backward_reference, (a, b, c, argmax, argmin, 1), "Test kernel v5: fullyconnected_backward", print_outputs)

if benchmarks:
    # print("__________________________")
    print("Benchmarks")

    n, l, m = 128, 3072, 768

    benchmark_kernel(torch.matmul, [(n, m), (m, l)], ['float', 'float'], title="Baseline: torch.matmul")

    # # kernel v1
    # benchmark_kernel(torch.ops.mamtorch_kernel_v1.fullyconnected, [(n, m), (m, l)], ['float', 'float'], title="Test kernel v1: fullyconnected")

    # # kernel v4
    # benchmark_kernel(torch.ops.mamtorch_kernel_v4.fullyconnected, [(n, m), (m, l)], ['float', 'float'], title="Test kernel v4: fullyconnected")
    # benchmark_kernel(torch.ops.mamtorch_kernel_v4.fullyconnected_fast, [(n, m), (m, l)], ['float', 'float'], title="Test kernel v4: fullyconnected_fast")
    benchmark_kernel(torch.ops.mamtorch_kernel_v4.fullyconnected_backward, [(n, m), (m, l), (n, l), (n, l), (n, l)],
                                                                        ['float', 'float', 'float', 'int', 'int'],
                                                                        [0, 0, 0, 0, 0],
                                                                        [0, 0, 0, m, m],
                                                                        title="Test kernel v4: fullyconnected_backward")

    # kernel v5
    accblock_size_list = [1, 4, 8, 16, 32, 64]
    for accblock_size in accblock_size_list:
        benchmark_kernel(torch.ops.mamtorch_kernel_v5.fullyconnected, [(n, m), (m, l), accblock_size], ['float', 'float', 'int'], [0, 0, 0], [0, 0, 0], [None, None, accblock_size], title=f"Test kernel v5: fullyconnected (accblock_size = {accblock_size})")

    benchmark_kernel(torch.ops.mamtorch_kernel_v5.fullyconnected_backward, [(n, m), (m, l), (n, l), (n, l), (n, l), 1],
                                                                        ['float', 'float', 'float', 'int', 'int', 'int'],
                                                                        [0, 0, 0, 0, 0, 0],
                                                                        [0, 0, 0, m, m, 0],
                                                                        title="Test kernel v5: fullyconnected_backward")

    