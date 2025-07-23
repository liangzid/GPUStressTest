"""
======================================================================
MAIN --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2025, ZiLiang, all rights reserved.
    Created: 23 July 2025
======================================================================
"""


# ------------------------ Code --------------------------------------


import torch
import time
from tqdm import tqdm

def stress_gpu(gpu_id, duration_sec=60*60*24):
    device = torch.device(f'cuda:{gpu_id}')
    
    M = 8000
    N = 8000
    
    A = torch.randn(M, N, device=device)
    B = torch.randn(N, M, device=device)
    
    start_time = time.time()
    iterations = 0
    
    with tqdm(desc=f'GPU {gpu_id} Stress Test', unit='iter') as pbar:
        while time.time() - start_time < duration_sec:
            C = torch.matmul(A, B)
            torch.cuda.synchronize(device)
            
            iterations += 1
            pbar.update(1)
    
    print(f'GPU {gpu_id} completed {iterations} iterations in {duration_sec} seconds')
    return iterations

def multi_gpu_stress_test():
    if not torch.cuda.is_available():
        print("No CUDA-capable GPU found!")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    import threading
    threads = []
    results = [0] * num_gpus
    
    def worker(gpu_id):
        results[gpu_id] = stress_gpu(gpu_id)
    
    for i in range(num_gpus):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print("\nStress test completed!")
    for i in range(num_gpus):
        print(f"GPU {i} performed {results[i]} matrix multiplications")

if __name__ == "__main__":
    multi_gpu_stress_test()
