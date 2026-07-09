import os, subprocess


def _detect_nvidia_gpu():
    """Queries nvidia-smi directly — torch isn't one of this project's
    dependencies, so importing it would report 'no GPU' on every machine
    regardless of actual hardware. CatBoost/XGBoost use CUDA without torch.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, mem_mb = result.stdout.strip().split("\n")[0].rsplit(",", 1)
            return True, name.strip(), int(mem_mb.strip()) // 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return False, None, 0


def detect_hardware():
    gpu, gpu_name, gpu_mem = _detect_nvidia_gpu()

    try:
        import psutil
        ram = psutil.virtual_memory().total // (1024 ** 3)
    except (ImportError, ModuleNotFoundError):
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
    }
