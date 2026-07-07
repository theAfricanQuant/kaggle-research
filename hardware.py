import os, platform


def detect_hardware():
    try:
        import torch
        gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu else None
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3) if gpu else 0
    except (ImportError, ModuleNotFoundError):
        gpu = False
        gpu_name = None
        gpu_mem = 0

    try:
        import psutil
        ram = psutil.virtual_memory().total // (1024 ** 3)
    except (ImportError, ModuleNotFoundError):
        import subprocess
        ram = int(subprocess.run(
            ["grep", "MemTotal", "/proc/meminfo"],
            capture_output=True, text=True
        ).stdout.split()[1]) // (1024 * 1024) if os.path.exists("/proc/meminfo") else 4

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
