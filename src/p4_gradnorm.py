"""Part 4: log the grad norm at every step, then find a step where it moved before the loss did.

The grad norm is the L2 norm of all gradients together, measured BEFORE clipping
(torch.nn.utils.clip_grad_norm_ returns that value). The clip limit is 1.0, a common default.

An "event" is a step s where the grad norm is more than K_NORM times its median over the previous WINDOW steps,
while the loss at s is still inside its normal band, and the loss rises above its band within the next LEAD steps.
Two real runs (no injected data):
  calm : LR 3e-3, a stable run. If it has no event, the script says so. It does not invent one.
  hot  : LR 1e-2, too high for this model. Here the norm jumps and the loss follows a few steps later.
"""
import statistics

import torch

from src.data import CONFIG_S, CharData, make_model
from src.device import RESULTS, Governor, autocast, env_info, pick_device, save_json, train_dtype, logged

STEPS, LR, CLIP, B, T = 1500, 3e-3, 1.0, 32, 128
HOT_LR, HOT_SEED = 1e-2, 6
WINDOW, K_NORM, LEAD = 50, 3.0, 10


def train_and_log(data, device, dtype, lr=LR, steps=STEPS, seed=5):
    model = make_model(CONFIG_S, data.vocab_size, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    loss_log, norm_log, gov = [], [], Governor(device)
    for _ in range(steps):
        x, y = data.get_batch(B, T, gen=gen, device=device)
        with gov.step():  # GPU busy at most GPU_DUTY of the time
            with autocast(device, dtype):
                _, loss = model(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # norm before clipping
            opt.step(); opt.zero_grad(set_to_none=True)
            loss_log.append(loss.item()); norm_log.append(norm.item())
    return loss_log, norm_log


def band(xs):
    """Median and a robust spread (1.4826 x median absolute deviation) of a window."""
    med = statistics.median(xs)
    mad = statistics.median(abs(v - med) for v in xs) * 1.4826
    return med, max(mad, 1e-8)


def find_events(loss_log, norm_log):
    events = []
    for s in range(WINDOW, len(norm_log) - LEAD):
        n_med, _ = band(norm_log[s - WINDOW:s])
        l_med, l_sd = band(loss_log[s - WINDOW:s])
        if norm_log[s] < K_NORM * n_med:
            continue
        loss_now_normal = loss_log[s] < l_med + 3 * l_sd
        after = loss_log[s + 1:s + 1 + LEAD]
        rise = [i + 1 for i, v in enumerate(after) if v > l_med + 3 * l_sd]
        if loss_now_normal and rise:
            events.append({"step": s, "norm": norm_log[s], "norm_median_before": n_med,
                           "norm_ratio": norm_log[s] / n_med, "loss_at_step": loss_log[s],
                           "loss_median_before": l_med, "loss_band": l_med + 3 * l_sd,
                           "loss_rises_after_steps": rise[0], "loss_peak_after": max(after)})
    return events


def one_run(name, data, device, dtype, lr, seed):
    loss_log, norm_log = train_and_log(data, device, dtype, lr=lr, seed=seed)
    events = find_events(loss_log, norm_log)
    print(f"[{name}] {STEPS} steps, lr {lr}, seed {seed}, clip {CLIP}. Grad norm: median {statistics.median(norm_log):.3f}, "
          f"max {max(norm_log):.3f} at step {norm_log.index(max(norm_log))}; "
          f"clip active on {sum(n > CLIP for n in norm_log)} of {STEPS} steps; "
          f"mean loss of last 50 steps {statistics.mean(loss_log[-50:]):.3f}")
    if events:
        e = max(events, key=lambda q: q["norm_ratio"])
        print(f"[{name}] {len(events)} event(s). Clearest: step {e['step']}: grad norm {e['norm']:.3f} "
              f"({e['norm_ratio']:.1f}x its median {e['norm_median_before']:.3f}) while the loss {e['loss_at_step']:.3f} "
              f"was still normal; the loss went above its band ({e['loss_band']:.3f}) {e['loss_rises_after_steps']} "
              f"step(s) later, peak {e['loss_peak_after']:.3f}")
    else:
        e = None
        print(f"[{name}] No event: the grad norm never jumped while the loss stayed normal and then rose.")
    return {"lr": lr, "seed": seed, "events": events, "clearest": e, "loss": loss_log, "grad_norm": norm_log}


@logged("p4_gradnorm")
def run(plot=True):
    device = pick_device(); dtype = train_dtype(device)
    data = CharData()
    runs = {"calm": one_run("calm", data, device, dtype, LR, 5),
            "hot": one_run("hot", data, device, dtype, HOT_LR, HOT_SEED)}
    out = {"env": env_info(device), "dtype": str(dtype), "steps": STEPS, "clip": CLIP,
           "rule": {"window": WINDOW, "k_norm": K_NORM, "lead": LEAD}, "runs": runs}
    save_json("p4_gradnorm.json", out)
    if plot:
        plot_traces(runs)
    return out


def plot_traces(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = []  # (run name, run, start, end)
    for name, r in runs.items():
        panels.append((name, r, 0, len(r["loss"])))
        if r["clearest"]:
            s = r["clearest"]["step"]; panels.append((name, r, max(0, s - 40), s + 40))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)
    for ax, (name, r, a, b) in zip(axes[0], panels):
        loss_log, norm_log, e = r["loss"], r["grad_norm"], r["clearest"]
        xs = range(a, b)
        ax.plot(xs, loss_log[a:b], color="tab:blue", lw=1, label="loss")
        ax.set_xlabel("step"); ax.set_ylabel("loss", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(xs, norm_log[a:b], color="tab:red", lw=1, label="grad norm (before clip)")
        ax2.axhline(CLIP, color="tab:red", ls=":", lw=1)
        ax2.set_ylabel("grad norm", color="tab:red")
        if e and a <= e["step"] < b:
            ax.axvline(e["step"], color="k", ls="--", lw=1)
            ax.axvline(e["step"] + e["loss_rises_after_steps"], color="gray", ls="--", lw=1)
        ax.set_title(f"{name} (lr {r['lr']}): whole run" if a == 0 else
                     f"{name} zoom: norm jumps at step {e['step']},\nloss follows {e['loss_rises_after_steps']} step(s) later")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "p4_gradnorm.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
