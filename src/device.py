"""Device, dtype and peak-FLOPs selection. The same code runs locally, on Colab, on AWS and on CPU."""
import contextlib
import json
import os
import pathlib
import platform
import time

import torch

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

# Dense tensor-core peak (FLOP/s) for the training dtype. These are data-sheet values, not measurements.
# Key: a substring of torch.cuda.get_device_name().
PEAK_FLOPS = {
    "RTX 5070 Ti": (87.9e12, "bf16", "aggregator value (flopper.io), not checked against NVIDIA's whitepaper"),
    "RTX 5090": (209.5e12, "bf16", "aggregator value, not checked"),
    "RTX 4090": (165.2e12, "bf16", "NVIDIA Ada whitepaper, FP32 accumulate"),
    "T4": (65e12, "fp16", "NVIDIA T4 data sheet; T4 has no BF16 tensor cores"),
    "L4": (121e12, "bf16", "NVIDIA L4 data sheet, dense"),
    "A10G": (70e12, "bf16", "approximate; check the AWS / NVIDIA sheet"),
    "A100": (312e12, "bf16", "NVIDIA A100 data sheet, dense (nanoGPT default)"),
    "H100": (989e12, "bf16", "NVIDIA H100 SXM data sheet, dense"),
}


# GPU sharing limits, so that the display (which uses the same GPU) stays responsive.
#   GPU_DUTY          largest fraction of wall time the GPU may be busy with this code (1.0 = no limit)
#   GPU_MEM_FRACTION  largest fraction of GPU memory this process may allocate
GPU_DUTY = float(os.environ.get("GPU_DUTY", "0.6"))
GPU_MEM_FRACTION = float(os.environ.get("GPU_MEM_FRACTION", "0.6"))


class Governor:
    """Run GPU work in short bursts with rests between them.

    Put `with gov.step():` around one unit of GPU work (one training step, one matmul).
    On exit it waits for the GPU to finish, records the busy time, and then sleeps
    busy * (1/GPU_DUTY - 1), so this code keeps the GPU busy at most GPU_DUTY of the wall time.
    Timing code reads `gov.busy` (the sum of busy times), which excludes the rests.
    """

    def __init__(self, device, duty=None):
        self.device = device
        self.duty = GPU_DUTY if duty is None else duty
        self.busy = 0.0
        self.last = 0.0

    @contextlib.contextmanager
    def step(self):
        t0 = time.perf_counter()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last = time.perf_counter() - t0
        self.busy += self.last
        if self.device.type == "cuda" and self.duty < 1.0:
            time.sleep(self.last * (1.0 / self.duty - 1.0))

    def reset(self):
        self.busy = 0.0


def pick_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        if GPU_MEM_FRACTION < 1.0:
            torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION, dev)
        return dev
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_dtype(device):
    """bf16 where the GPU supports it, else fp16 (needs a GradScaler), else fp32 on CPU."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    return torch.float32


def autocast(device, dtype):
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def gpu_name(device):
    return torch.cuda.get_device_name(device) if device.type == "cuda" else f"CPU ({platform.processor() or platform.machine()})"


def peak_flops(device):
    """(peak FLOP/s or None, source note). Override with the PEAK_TFLOPS environment variable."""
    if os.environ.get("PEAK_TFLOPS"):
        return float(os.environ["PEAK_TFLOPS"]) * 1e12, "PEAK_TFLOPS environment override"
    if device.type != "cuda":
        return None, "no GPU: MFU is not defined"
    name = gpu_name(device)
    for key, (flops, _, note) in PEAK_FLOPS.items():
        if key in name:
            return flops, note
    return None, f"unknown GPU '{name}': set PEAK_TFLOPS"


def env_info(device):
    return {
        "device": gpu_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "os": platform.platform(),
        "gpu_duty_limit": GPU_DUTY if device.type == "cuda" else None,
        "gpu_mem_fraction": GPU_MEM_FRACTION if device.type == "cuda" else None,
    }


def save_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path
