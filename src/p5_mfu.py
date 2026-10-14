"""Part 5: compute MFU honestly and explain the distance to 40%.

MFU = (FLOPs the model needs per token x tokens per second) / (peak FLOP/s of the hardware)
FLOPs per token = 6N + 12*L*H*Q*T   (PaLM Appendix B; nanoGPT estimate_mfu)
  6N      : 2N forward + 4N backward for the matmuls of N weights (N excludes position embeddings)
  12LHQT  : the attention score and value matmuls, which do not appear in N
"""
import time

import torch

from src.data import CONFIG_SHAKESPEARE_CHAR, CharData, make_model
from src.device import GPU_DUTY, Governor, autocast, env_info, peak_flops, pick_device, save_json, train_dtype

B = 64            # nanoGPT shakespeare_char batch size
WARMUP, TIMED = 10, 50


def flops_per_token(model):
    cfg = model.config
    N = model.get_num_params()
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
    return 6 * N, 12 * L * H * Q * T, N


def measure(model, data, device, dtype, compiled=False):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=device.type == "cuda")
    step_model = torch.compile(model) if compiled else model
    gen = torch.Generator().manual_seed(0)
    T = model.config.block_size
    batches = [data.get_batch(B, T, gen=gen, device=device) for _ in range(8)]  # made before timing

    def step(i):
        x, y = batches[i % len(batches)]
        with autocast(device, dtype):
            _, loss = step_model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

    # Each step runs inside the Governor: it ends with a GPU sync, and the rest that follows is NOT timed.
    # So dt is the true step time, while the GPU still gets a rest between steps for the display.
    gov = Governor(device)
    for i in range(WARMUP):
        with gov.step():
            step(i)
    gov.reset()
    for i in range(TIMED):
        with gov.step():
            step(i)
    dt = gov.busy / TIMED
    return {"seconds_per_step": dt, "tokens_per_second": B * T / dt,
            "note": "per-step sync; rests between steps are excluded from the time"}, step


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def matmul_ceiling(device, dtype, n=8192, reps=20):
    """The FLOP/s this GPU actually reaches on one large square matmul: a measured ceiling, not a data sheet."""
    if device.type != "cuda":
        return None
    a = torch.randn(n, n, device=device, dtype=dtype); b = torch.randn(n, n, device=device, dtype=dtype)
    gov = Governor(device)
    for _ in range(3):
        with gov.step():
            a @ b
    gov.reset()
    for _ in range(reps):  # one matmul per burst, with a rest after each
        with gov.step():
            a @ b
    return 2 * n ** 3 * reps / gov.busy


def profile(step, device):
    """Share of GPU time per kernel group, and GPU-busy time as a fraction of wall time."""
    if device.type != "cuda":
        return None
    from torch.profiler import ProfilerActivity, profile as tprofile
    gov = Governor(device)
    with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(5):
            with gov.step():
                step(i)
    wall = gov.busy  # step time only; the rests between steps are excluded
    kernels = [e for e in prof.key_averages() if e.device_type.name == "CUDA" and e.self_device_time_total > 0]
    total = sum(e.self_device_time_total for e in kernels)
    groups = {"matmul (gemm)": 0.0, "attention (sdpa / flash)": 0.0, "optimizer (adam)": 0.0,
              "layernorm / softmax / gelu / cross-entropy": 0.0, "other elementwise / copies": 0.0}
    for e in kernels:
        k = e.key.lower()
        if any(s in k for s in ("gemm", "cutlass", "sm80_xmma", "sm90", "sm100", "sm120", "ampere_", "matmul")):
            groups["matmul (gemm)"] += e.self_device_time_total
        elif any(s in k for s in ("flash", "fmha", "attention", "efficient")):
            groups["attention (sdpa / flash)"] += e.self_device_time_total
        elif "adam" in k or "foreach" in k:
            groups["optimizer (adam)"] += e.self_device_time_total
        elif any(s in k for s in ("layer_norm", "softmax", "gelu", "nll", "cross_entropy", "log_softmax")):
            groups["layernorm / softmax / gelu / cross-entropy"] += e.self_device_time_total
        else:
            groups["other elementwise / copies"] += e.self_device_time_total
    top = sorted(kernels, key=lambda e: -e.self_device_time_total)[:10]
    return {"gpu_busy_fraction": (total / 1e6) / wall,  # self_device_time_total is in microseconds
            "kernel_launches_per_step": sum(e.count for e in kernels) / 5,
            "groups_pct": {g: 100 * v / total for g, v in groups.items()},
            "top_kernels": [{"name": e.key[:90], "pct": 100 * e.self_device_time_total / total, "calls": e.count}
                            for e in top]}


def run():
    device = pick_device(); dtype = train_dtype(device)
    data = CharData()
    model = make_model(CONFIG_SHAKESPEARE_CHAR, data.vocab_size, device)
    f6n, fattn, N = flops_per_token(model)
    peak, peak_note = peak_flops(device)
    env = env_info(device)
    print(f"GPU duty limit {GPU_DUTY:.0%} (rests between steps are not counted in any timing)")
    print(f"Device {env['device']}, dtype {dtype}, N = {N:,} (non-embedding), "
          f"FLOPs/token = 6N {f6n:,} + attention {fattn:,}")

    res, step = measure(model, data, device, dtype)
    out = {"env": env, "dtype": str(dtype), "B": B, "T": model.config.block_size, "N": N,
           "flops_per_token_6N": f6n, "flops_per_token_attn": fattn, "peak_flops": peak, "peak_note": peak_note,
           "eager": res}
    achieved = (f6n + fattn) * res["tokens_per_second"]
    out["achieved_flops"] = achieved
    print(f"Eager: {res['tokens_per_second']:,.0f} tokens/s, {1000 * res['seconds_per_step']:.1f} ms/step, "
          f"{achieved / 1e12:.2f} TFLOP/s of model work")
    if peak:
        out["mfu"] = achieved / peak
        out["mfu_without_attention"] = f6n * res["tokens_per_second"] / peak
        out["mfu_nanogpt_estimate"] = model.estimate_mfu(B, res["seconds_per_step"], flops_promised=peak)
        print(f"MFU = {100 * out['mfu']:.1f}% (without attention {100 * out['mfu_without_attention']:.1f}%; "
              f"nanoGPT estimate_mfu {100 * out['mfu_nanogpt_estimate']:.1f}%) against peak "
              f"{peak / 1e12:.1f} TFLOP/s [{peak_note}]")
    else:
        print(f"MFU: n/a ({peak_note})")

    ceiling = matmul_ceiling(device, dtype)
    if ceiling:
        out["measured_matmul_ceiling"] = ceiling
        out["mfu_vs_measured_ceiling"] = achieved / ceiling
        print(f"Measured 8192x8192 matmul ceiling: {ceiling / 1e12:.1f} TFLOP/s; "
              f"model work is {100 * achieved / ceiling:.1f}% of it")

    prof = profile(step, device)
    if prof:
        out["profile"] = prof
        print(f"GPU busy {100 * prof['gpu_busy_fraction']:.0f}% of wall time; "
              f"{prof['kernel_launches_per_step']:.0f} kernel launches per step")
        for g, v in prof["groups_pct"].items():
            print(f"  {g:45s} {v:5.1f}% of GPU time")

    try:  # torch.compile needs Triton; on Windows that can fail. Report it either way.
        model_c = make_model(CONFIG_SHAKESPEARE_CHAR, data.vocab_size, device)
        res_c, _ = measure(model_c, data, device, dtype, compiled=True)
        out["compiled"] = res_c
        if peak:
            out["mfu_compiled"] = (f6n + fattn) * res_c["tokens_per_second"] / peak
        print(f"torch.compile: {res_c['tokens_per_second']:,.0f} tokens/s"
              + (f", MFU {100 * out['mfu_compiled']:.1f}%" if peak else ""))
    except Exception as exc:
        out["compiled_error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        print(f"torch.compile failed: {out['compiled_error']}")

    save_json("p5_mfu.json", out)
    return out


if __name__ == "__main__":
    run()
