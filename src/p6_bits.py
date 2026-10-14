"""Part 6: 0.1 in fp32, bf16 and fp8 E4M3, bit by bit.

encode() does by hand what the hardware does: write the number as 1.f x 2^e, add the bias to e,
keep the first M bits of f, and round to nearest (ties to even). torch then checks every result.
"""
from fractions import Fraction

import torch

from src.device import save_json

FORMATS = {  # name: (exponent bits, mantissa bits, bias)
    "fp32": (8, 23, 127),
    "bf16": (8, 7, 127),
    "fp8_e4m3": (4, 3, 7),
}


def encode(value, E, M, bias):
    """Normal numbers only (0.1 is normal in all three formats). Exact arithmetic with Fraction."""
    # Fraction("0.1") is exactly 1/10, not the float64 approximation of 0.1
    v = Fraction(str(value)) if isinstance(value, float) else Fraction(value)
    sign = 0 if v >= 0 else 1
    v = abs(v)
    e = 0
    while v >= 2:
        v /= 2; e += 1
    while v < 1:
        v *= 2; e -= 1
    frac = v - 1                       # 0.1 = 1.6 x 2^-4  ->  frac = 0.6
    scaled = frac * 2 ** M             # the mantissa bits we can keep, as a number with a remainder
    keep = int(scaled)
    rem = scaled - keep
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and keep % 2 == 1):
        keep += 1                      # round to nearest, ties to even
    if keep == 2 ** M:                 # the rounding carried into the exponent
        keep = 0; e += 1
    exp_field = e + bias
    stored = (1 + Fraction(keep, 2 ** M)) * Fraction(2) ** e
    return {"sign": sign, "exponent": e, "exp_field": exp_field,
            "exp_bits": format(exp_field, f"0{E}b"), "mant_bits": format(keep, f"0{M}b"),
            "first_dropped_bits": format(int(rem * 2 ** 8), "08b"),
            "bits": f"{sign} {format(exp_field, f'0{E}b')} {format(keep, f'0{M}b')}",
            "hex": f"0x{(sign << (E + M)) | (exp_field << M) | keep:0{(1 + E + M + 3) // 4}X}",
            "stored": float(stored), "abs_error": float(abs(stored - Fraction(1, 10))),
            "rel_error": float(abs(stored - Fraction(1, 10)) / Fraction(1, 10))}


def torch_bits(name):
    if name == "fp32":
        return torch.tensor(0.1, dtype=torch.float32).view(torch.int32).item() & 0xFFFFFFFF, 32
    if name == "bf16":
        return torch.tensor(0.1, dtype=torch.float32).to(torch.bfloat16).view(torch.int16).item() & 0xFFFF, 16
    return torch.tensor(0.1, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).item(), 8


def run():
    print("0.1 = 1.6 x 2^-4. In binary, 1.6 = 1.1001 1001 1001 ... (the pattern 1001 repeats without end),")
    print("so no binary format can hold 0.1 exactly. Each format keeps a different number of those bits.\n")
    out = {}
    print("| format | sign | exponent field (e + bias) | mantissa | hex | stored value | relative error | torch agrees |")
    print("|---|---|---|---|---|---|---|---|")
    for name, (E, M, bias) in FORMATS.items():
        r = encode(0.1, E, M, bias)
        tb, width = torch_bits(name)
        r["torch_hex"] = f"0x{tb:0{width // 4}X}"
        r["torch_agrees"] = int(r["hex"], 16) == tb
        out[name] = r
        print(f"| {name} | {r['sign']} | {r['exp_bits']} ({r['exponent']}+{bias}={r['exp_field']}) | {r['mant_bits']} | "
              f"{r['hex']} | {r['stored']:.15g} | {r['rel_error']:.2e} | {r['torch_agrees']} ({r['torch_hex']}) |")
    save_json("p6_bits.json", out)
    return out


if __name__ == "__main__":
    run()
