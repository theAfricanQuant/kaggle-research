import torch, psutil, os, platform

def detect_hardware():
    gpu = torch.cuda.is_available()
    gpu_name = None
    gpu_mem = 0
    if gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)

    ram = psutil.virtual_memory().total // (1024 ** 3)
    cores = os.cpu_count() or 1
    on_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

    return {
        "gpu": gpu,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem,
        "ram_gb": ram,
        "cores": cores,
        "on_kaggle": on_kaggle,
        "parallel_workers": max(1, cores // 2),
    }
