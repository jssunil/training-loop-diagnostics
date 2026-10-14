"""Part 1: print every tensor shape in one training step, with one line on what each dimension means."""
import torch

from src import model as nanogpt
from src.data import CONFIG_S, CharData, make_model
from src.device import env_info, pick_device, save_json

B, T = 4, 16  # small numbers, all different from C, nh and hs, so each dimension is easy to see

MEANING = {
    "idx": "B sequences x T token positions; each entry is a character id in [0, V)",
    "tok_emb": "B x T x C: one C-number embedding vector for each token",
    "pos_emb": "T x C: one C-number vector for each position; it is added to every sequence (broadcast over B)",
    "x0 = tok_emb + pos_emb": "B x T x C: the residual stream that enters block 0",
    "attn.q": "B x nh x T x hs: the query of each head, for each position (C = nh x hs)",
    "attn.k": "B x nh x T x hs: the key of each head, for each position",
    "attn.v": "B x nh x T x hs: the value of each head, for each position",
    "attn.scores (inside SDPA, not stored)": "B x nh x T x T: how much position t (row) looks at position s (column); causal, so s <= t",
    "attn.y_heads": "B x nh x T x hs: the attention output of each head",
    "attn.y_merged": "B x T x C: the nh head outputs put side by side again",
    "ln_f out": "B x T x C: the residual stream after the last LayerNorm",
    "logits": "B x T x V: one score per vocabulary entry, for the next token at each position",
    "loss": "a scalar (shape ()): the mean cross-entropy over all B x T predictions",
    "y (targets)": "B x T: the correct next character id for each position (x moved one place right)",
}


def meaning(name):
    if name.startswith("block["):
        return "B x T x C: the residual stream after this block (attention + MLP, each with a residual add)"
    return MEANING.get(name, "")


def run():
    device = pick_device()
    data = CharData()
    model = make_model(CONFIG_S, data.vocab_size, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = data.get_batch(B, T, gen=torch.Generator().manual_seed(0), device=device)
    cfg = model.config
    print(f"Device: {env_info(device)['device']}")
    print(f"Symbols: B={B} sequences, T={T} positions, C={cfg.n_embd} channels, nh={cfg.n_head} heads, "
          f"hs={cfg.n_embd // cfg.n_head} head size, V={data.vocab_size} characters, L={cfg.n_layer} blocks\n")

    rows, seen_blocks = [], set()

    def trace(name, t):
        shape = tuple(t.shape) if isinstance(t, torch.Tensor) else tuple(t)
        if name.startswith("attn."):  # the same tensors repeat in every block; print block 0 only
            if "attn" in seen_blocks and name == "attn.q":
                trace.skip = True
            if name == "attn.q" and "attn" not in seen_blocks:
                seen_blocks.add("attn")
                trace.skip = False
            if trace.skip:
                return
        rows.append((name, shape, meaning(name)))
    trace.skip = False

    # Forward
    nanogpt.set_trace(trace)
    logits, loss = model(x, y)
    nanogpt.set_trace(None)
    rows.insert(1, ("y (targets)", tuple(y.shape), meaning("y (targets)")))

    print("== Forward pass (attention tensors shown for block 0; blocks 1..L-1 are the same) ==")
    print(f"| tensor | shape | what each dimension means |\n|---|---|---|")
    for n, s, m in rows:
        print(f"| {n} | {s} | {m} |")

    # Backward + step
    loss.backward()
    print("\n== Backward pass: every parameter and its gradient (a gradient always has its weight's shape) ==")
    print("| parameter | weight shape | grad shape | what the dimensions mean |\n|---|---|---|---|")
    params = []
    for n, p in model.named_parameters():
        dims = describe_param(n, tuple(p.shape), cfg, data.vocab_size)
        params.append((n, tuple(p.shape), tuple(p.grad.shape), dims))
        print(f"| {n} | {tuple(p.shape)} | {tuple(p.grad.shape)} | {dims} |")
    opt.step()
    state = opt.state[next(model.parameters())]
    print(f"\n== Optimizer step: AdamW keeps exp_avg {tuple(state['exp_avg'].shape)} and exp_avg_sq "
          f"{tuple(state['exp_avg_sq'].shape)} for each weight (same shape as the weight) ==")
    opt.zero_grad(set_to_none=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,} (lm_head.weight is tied to transformer.wte.weight, so it is listed once)")
    save_json("p1_shapes.json", {"env": env_info(device), "B": B, "T": T,
                                 "forward": [list(r[:2]) + [r[2]] for r in rows],
                                 "params": params, "n_params": n_params})
    return rows, params


def describe_param(name, shape, cfg, V):
    C = cfg.n_embd
    if name.endswith("wte.weight"):
        return f"V x C: one {C}-number vector per character; also the output head (tied)"
    if name.endswith("wpe.weight"):
        return f"block_size x C: one vector per position 0..{cfg.block_size - 1}"
    if "c_attn.weight" in name:
        return "3C x C: q, k and v projections stacked (out x in)"
    if "c_attn.bias" in name:
        return "3C: bias of q, k and v"
    if "attn.c_proj.weight" in name:
        return "C x C: merges the heads back into the residual stream (out x in)"
    if "c_fc.weight" in name:
        return "4C x C: MLP expansion (out x in)"
    if "mlp.c_proj.weight" in name:
        return "C x 4C: MLP contraction (out x in)"
    if "ln" in name:
        return "C: one scale or shift per channel"
    if name.endswith("bias"):
        return "out: one bias per output feature"
    return ""


if __name__ == "__main__":
    run()
