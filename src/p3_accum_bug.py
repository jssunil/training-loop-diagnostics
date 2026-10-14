"""Part 3: break gradient accumulation on purpose (the average of the averages) and plot it against the fix.

Three runs start from the same weights and see the same micro-batches in the same order:
  full    : the micro-batches of one step joined into one batch; loss = sum over tokens / all tokens
  correct : accumulation; each micro-batch adds sum(loss) / (tokens in the whole step)      (the 2024 fix)
  bug     : accumulation; each micro-batch adds mean(loss) / K   = the average of the averages
"""
import torch
import torch.nn.functional as F

from src.data import CONFIG_S, PAD, CharData, make_model
from src.device import RESULTS, Governor, autocast, env_info, pick_device, save_json, train_dtype

T = 128
MICRO = [(16, 128), (16, 128), (16, 8)]  # (sequences, real tokens per sequence): 2048, 2048, 128 tokens
STEPS, LR, EVAL_EVERY = 400, 1e-3, 20


def worked_example():
    """Worked example: three micro-batches with 4, 4 and 2 valid tokens, average losses 2, 2, 5."""
    n, avg = [4, 4, 2], [2.0, 2.0, 5.0]
    correct = sum(a * b for a, b in zip(n, avg)) / sum(n)
    bug = sum(avg) / len(avg)
    equal = [2.0, 2.0, 5.0]
    return {"correct": correct, "bug": bug, "error_pct": 100 * (bug - correct) / correct,
            "equal_lengths_correct": sum(4 * a for a in equal) / 12, "equal_lengths_bug": sum(equal) / 3}


def token_losses(model, x, y):
    logits, _ = model(x, all_logits=True)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=PAD, reduction="sum")


def micro_batches(data, gen, device, micro=MICRO):
    return [data.get_ragged_batch(b, T, n, gen=gen, device=device) for b, n in micro]


def step_backward(model, mode, mbs, device, dtype):
    """Run forward and backward for one step in `mode`. Returns the true (token-weighted) loss of the step."""
    total_tokens = sum(int((y != PAD).sum()) for _, y in mbs)
    loss_sum = 0.0
    if mode == "full":  # pad every micro-batch to T and join them
        x = torch.cat([x for x, _ in mbs]); y = torch.cat([y for _, y in mbs])
        with autocast(device, dtype):
            s = token_losses(model, x, y)
        (s / total_tokens).backward()
        return s.item() / total_tokens
    K = len(mbs)
    for x, y in mbs:
        n = int((y != PAD).sum())
        with autocast(device, dtype):
            s = token_losses(model, x, y)
        if mode == "correct":
            (s / total_tokens).backward()   # every token has the same weight 1/total_tokens
        else:  # "bug"
            (s / n / K).backward()          # every micro-batch has the same weight 1/K
        loss_sum += s.item()
    return loss_sum / total_tokens


@torch.no_grad()
def val_loss(model, data, device, dtype, gen):
    model.eval()
    losses, gov = [], Governor(device)
    for _ in range(8):
        x, y = data.get_batch(32, T, split="val", gen=gen, device=device)
        with gov.step(), autocast(device, dtype):
            losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def grad_vector(model):
    return torch.cat([p.grad.flatten().float() for p in model.parameters()])


def first_step_grads(data, device, dtype, micro=MICRO):
    """Gradient of step 0 in all three modes, from the same weights and micro-batches."""
    out = {}
    for mode in ["full", "correct", "bug"]:
        model = make_model(CONFIG_S, data.vocab_size, device)
        mbs = micro_batches(data, torch.Generator().manual_seed(42), device, micro)
        with Governor(device).step():
            step_backward(model, mode, mbs, device, torch.float32)  # fp32 so that only the reduction differs
        out[mode] = grad_vector(model)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    cos = lambda a, b: F.cosine_similarity(a, b, dim=0).item()
    return {"rel_diff_correct_vs_full": rel(out["correct"], out["full"]),
            "rel_diff_bug_vs_full": rel(out["bug"], out["full"]),
            "cosine_bug_vs_full": cos(out["bug"], out["full"]),
            "norm_ratio_bug_over_full": (out["bug"].norm() / out["full"].norm()).item()}


def train(mode, data, device, dtype):
    model = make_model(CONFIG_S, data.vocab_size, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    gen, vgen = torch.Generator().manual_seed(123), torch.Generator().manual_seed(7)
    train_curve, val_curve, gov = [], [], Governor(device)
    for step in range(STEPS):
        mbs = micro_batches(data, gen, device)
        with gov.step():  # GPU busy at most GPU_DUTY of the time
            train_curve.append(step_backward(model, mode, mbs, device, dtype))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            val_curve.append((step, val_loss(model, data, device, dtype, vgen)))
    return train_curve, val_curve


def run(plot=True):
    device = pick_device(); dtype = train_dtype(device)
    data = CharData()
    ex = worked_example()
    print(f"Worked example: correct {ex['correct']:.4f}, average of averages {ex['bug']:.4f} "
          f"({ex['error_pct']:.1f}% too high); with equal lengths both give {ex['equal_lengths_correct']:.4f} "
          f"and {ex['equal_lengths_bug']:.4f}")
    g = first_step_grads(data, device, dtype)
    print(f"Step-0 gradients: |correct - full|/|full| = {g['rel_diff_correct_vs_full']:.2e}; "
          f"|bug - full|/|full| = {g['rel_diff_bug_vs_full']:.3f}; cosine(bug, full) = {g['cosine_bug_vs_full']:.4f}")
    ge = first_step_grads(data, device, dtype, micro=[(16, 128)] * 3)
    print(f"Control, equal lengths {[(16, 128)] * 3}: |bug - full|/|full| = {ge['rel_diff_bug_vs_full']:.2e} "
          f"(the bug hides when every micro-batch holds the same number of tokens)")
    tokens = [b * n for b, n in MICRO]
    print(f"Micro-batches per step: {MICRO} -> real tokens {tokens}; the short one gets "
          f"{100 / len(MICRO):.0f}% of the vote in the bug but holds {100 * tokens[-1] / sum(tokens):.1f}% of the tokens")
    curves = {m: train(m, data, device, dtype) for m in ["full", "correct", "bug"]}
    last = lambda c: sum(c[-50:]) / 50
    summary = {m: {"train_loss_last50": last(c[0]), "val_loss_final": c[1][-1][1]} for m, c in curves.items()}
    for m, s in summary.items():
        print(f"{m:8s}: train loss (token-weighted, mean of last 50 steps) {s['train_loss_last50']:.4f}, "
              f"final val loss {s['val_loss_final']:.4f}")
    gap = summary["bug"]["val_loss_final"] - summary["correct"]["val_loss_final"]
    print(f"Final val gap bug - correct = {gap:+.4f}")
    out = {"env": env_info(device), "dtype": str(dtype), "micro": MICRO, "steps": STEPS, "lr": LR,
           "worked_example": ex, "step0_grads": g, "step0_grads_equal_lengths": ge, "summary": summary, "val_gap_bug_minus_correct": gap,
           "curves": {m: {"train": c[0], "val": c[1]} for m, c in curves.items()}}
    save_json("p3_accum_bug.json", out)
    if plot:
        plot_curves(curves, g)
    return out


def plot_curves(curves, g):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ema = lambda c, a=0.9: [v for v in _ema(c, a)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    style = {"full": ("k", ":"), "correct": ("tab:blue", "-"), "bug": ("tab:red", "-")}
    for m, (tr, va) in curves.items():
        c, ls = style[m]
        ax[0].plot(ema(tr), color=c, ls=ls, label=f"{m}")
        ax[1].plot([s for s, _ in va], [v for _, v in va], color=c, ls=ls, marker="o", ms=3, label=m)
    ax[0].set(title="Train loss (token-weighted, EMA 0.9)", xlabel="step", ylabel="loss")
    ax[1].set(title="Validation loss (full 128-token windows)", xlabel="step", ylabel="loss")
    for a in ax:
        a.legend(); a.grid(alpha=.3)
    fig.suptitle(f"Gradient accumulation: correct vs average of averages\n"
                 f"micro-batch tokens {[b*n for b, n in MICRO]}; step-0 gradient of the bug differs from "
                 f"the full batch by {100*g['rel_diff_bug_vs_full']:.1f}%")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "p3_accum_bug.png", dpi=120)
    plt.close(fig)


def _ema(xs, a):
    m = None
    for v in xs:
        m = v if m is None else a * m + (1 - a) * v
        yield m


if __name__ == "__main__":
    run()
