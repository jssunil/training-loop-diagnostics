"""Part 2: verify one gradient by hand. Nudge a weight, measure the loss change, compare with backward()."""
import torch

from src.data import CONFIG_S, CharData, make_model
from src.device import env_info, save_json, logged

PARAM = "transformer.h.0.attn.c_attn.weight"  # a query weight in block 0
INDEX = (3, 7)                                # row 3 (output feature), column 7 (input channel)
EPSILONS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]


def toy_chain():
    """The worked chain: h = w1*x, y = w2*h, loss = (y - t)^2 with w1=3, w2=4, x=2, t=20."""
    w1 = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    x, t = 2.0, 20.0
    loss = (w2 * (w1 * x) - t) ** 2
    loss.backward()
    f = lambda a: float((4.0 * (a * x) - t) ** 2)
    nudge_forward = (f(3.001) - f(3.0)) / 0.001          # the one-sided nudge: 64.064 (the page rounds the change to 0.064)
    nudge_central = (f(3.0 + 1e-6) - f(3.0 - 1e-6)) / 2e-6  # central difference: 64.000000
    return {"autograd_dL_dw1": w1.grad.item(), "autograd_dL_dw2": w2.grad.item(),
            "nudge_forward_eps_1e-3": nudge_forward, "nudge_central_eps_1e-6": nudge_central}


def check(dtype, data, x, y, index=INDEX):
    """Central difference (L(w+eps) - L(w-eps)) / 2eps for one weight, against autograd, in one dtype."""
    model = make_model(CONFIG_S, data.vocab_size, "cpu").to(dtype)
    model.eval()  # dropout is 0 already; eval() makes sure the forward pass repeats exactly
    p = dict(model.named_parameters())[PARAM]
    _, loss = model(x, y)
    loss.backward()
    if index == "max":  # the weight in PARAM with the largest |gradient|
        flat = p.grad.abs().argmax().item()
        index = (flat // p.shape[1], flat % p.shape[1])
    g_autograd = p.grad[index].item()
    rows = []
    with torch.no_grad():
        w0 = p[index].item()
        for eps in EPSILONS:
            p[index] = w0 + eps
            lp = model(x, y)[1].item()
            p[index] = w0 - eps
            lm = model(x, y)[1].item()
            p[index] = w0
            g_nudge = (lp - lm) / (2 * eps)
            rows.append({"eps": eps, "nudge": g_nudge, "autograd": g_autograd,
                         "abs_err": abs(g_nudge - g_autograd),
                         "rel_err": abs(g_nudge - g_autograd) / max(abs(g_autograd), 1e-30)})
    return {"index": list(index), "weight_value": w0, "autograd": g_autograd, "loss": loss.item(), "rows": rows}


@logged("p2_gradcheck")
def run():
    torch.manual_seed(0)
    data = CharData()
    x, y = data.get_batch(8, 64, gen=torch.Generator().manual_seed(1))
    out = {"env": env_info(torch.device("cpu")), "param": PARAM, "index": INDEX, "toy_chain": toy_chain()}
    tc = out["toy_chain"]
    print(f"Toy chain (worked example): autograd dL/dw1 = {tc['autograd_dL_dw1']:.6f}, "
          f"one-sided nudge 3.000->3.001 = {tc['nudge_forward_eps_1e-3']:.6f}, "
          f"central nudge = {tc['nudge_central_eps_1e-6']:.6f}\n")
    runs = [("float64", torch.float64, INDEX), ("float32", torch.float32, INDEX),
            ("float32_largest_grad", torch.float32, "max")]
    for name, dtype, index in runs:
        r = check(dtype, data, x, y, index)
        out[name] = r
        print(f"== {name}: {PARAM}{r['index']} = {r['weight_value']:.6f}, loss = {r['loss']:.6f}, "
              f"autograd grad = {r['autograd']:.10e} ==")
        print("| eps | nudge (L(w+e)-L(w-e))/2e | autograd | abs error | rel error |\n|---|---|---|---|---|")
        for row in r["rows"]:
            print(f"| {row['eps']:.0e} | {row['nudge']:.10e} | {row['autograd']:.10e} | "
                  f"{row['abs_err']:.2e} | {row['rel_err']:.2e} |")
        best = min(r["rows"], key=lambda q: q["rel_err"])
        print(f"Best agreement at eps={best['eps']:.0e}: relative error {best['rel_err']:.2e}\n")
        r["best"] = best
    save_json("p2_gradcheck.json", out)
    return out


if __name__ == "__main__":
    run()
