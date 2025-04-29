#!/usr/bin/env python
"""
Utility functions for GPU management and monitoring.
"""
import os
import torch
import subprocess
import json

def get_gpu_info():
    """Get GPU information using nvidia-smi in a more programmatic way."""
    try:
        nvidia_smi_output = subprocess.check_output(
            ['nvidia-smi', '--format=csv,noheader,nounits', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu'],
            encoding='utf-8'
        )
        
        gpu_info = []
        for line in nvidia_smi_output.strip().split('\n'):
            values = line.split(', ')
            if len(values) >= 6:
                gpu_info.append({
                    'index': int(values[0]),
                    'name': values[1],
                    'memory.total': int(values[2]),
                    'memory.used': int(values[3]),
                    'memory.free': int(values[4]),
                    'utilization.gpu': float(values[5])
                })
        return gpu_info
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

def print_gpu_utilization():
    """Print GPU utilization."""
    gpu_info = get_gpu_info()
    if gpu_info:
        print("\nGPU UTILIZATION:")
        print("----------------")
        for gpu in gpu_info:
            print(f"GPU {gpu['index']} ({gpu['name']}):")
            print(f"  Memory: {gpu['memory.used']} MB / {gpu['memory.total']} MB ({gpu['memory.used']/gpu['memory.total']*100:.1f}%)")
            print(f"  Utilization: {gpu['utilization.gpu']}%")
    else:
        print("\nUnable to get GPU information from nvidia-smi")

def print_pytorch_gpu_memory():
    """Print PyTorch's view of GPU memory usage."""
    if torch.cuda.is_available():
        print("\nPYTORCH GPU MEMORY:")
        print("------------------")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
            print(f"GPU {i}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    else:
        print("\nPyTorch does not detect CUDA")

def optimize_gpu_memory(clear_cache=True):
    """Optimize GPU memory usage."""
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    
    # Other optimizations could be added here
    torch.backends.cudnn.benchmark = True
    
def verify_multi_gpu_setup():
    """Verify multi-GPU setup and configuration."""
    print("\nMULTI-GPU SETUP VERIFICATION:")
    print("----------------------------")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs detected by PyTorch: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Check if all specified GPUs are detected
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            if len(visible_devices) > torch.cuda.device_count():
                print(f"WARNING: {len(visible_devices)} GPUs specified in CUDA_VISIBLE_DEVICES, but PyTorch only detects {torch.cuda.device_count()}")
                
    return torch.cuda.device_count() > 1

if __name__ == "__main__":
    # Simple test run when executed directly
    verify_multi_gpu_setup()
    print_gpu_utilization()
    print_pytorch_gpu_memory() 