import torch
import timeit


def test_GPU_vs_CPU(device):
    if device == "cpu":
        x = torch.rand((10000, 10000), dtype=torch.float32)
        y = torch.rand((10000, 10000), dtype=torch.float32)
    else:
        # by creating the tensors on the GPU we avoid OVERHEAD transfering the tensors from the CPU to the GPU, similar above it will save us a lot of time when traing on the GPU
        x = torch.rand((10000, 10000), dtype=torch.float32, device=device)
        y = torch.rand((10000, 10000), dtype=torch.float32, device=device)
    x * y


numberTimesToRun = 10

""" === Test with CPU """
device = torch.device("cpu")
execution_CPU_time = timeit.timeit(
    lambda: test_GPU_vs_CPU(device), number=numberTimesToRun
)

print(f"\nCPU Execution time: {execution_CPU_time:.5f} seconds")


""" === TEST with GPU (5x faster)"""
device = torch.device("mps")

execution_CPU_time = timeit.timeit(
    lambda: test_GPU_vs_CPU(device), number=numberTimesToRun
)
print(f"\nGPU Execution time: {execution_CPU_time:.5f} seconds")


"""
Example print
    CPU Execution time: 5.75536 seconds

    GPU Execution time: 0.05824 seconds
"""
