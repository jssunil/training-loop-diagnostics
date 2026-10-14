"""Fast checks for the claims in the README (CPU only, a few seconds)."""
import torch

from src.data import CONFIG_S, CharData
from src.p2_gradcheck import check, toy_chain
from src.p3_accum_bug import first_step_grads, worked_example
from src.p6_bits import encode, torch_bits, FORMATS

DATA = CharData()


def test_toy_chain_is_64():
    r = toy_chain()
    assert abs(r["autograd_dL_dw1"] - 64.0) < 1e-12
    assert abs(r["nudge_central_eps_1e-6"] - 64.0) < 1e-6


def test_nudge_matches_autograd_in_float64():
    x, y = DATA.get_batch(8, 64, gen=torch.Generator().manual_seed(1))
    r = check(torch.float64, DATA, x, y)
    assert min(row["rel_err"] for row in r["rows"]) < 1e-7


def test_correct_accumulation_equals_full_batch_and_bug_does_not():
    g = first_step_grads(DATA, torch.device("cpu"), torch.float32)
    assert g["rel_diff_correct_vs_full"] < 1e-5
    assert g["rel_diff_bug_vs_full"] > 0.05


def test_bug_hides_with_equal_lengths():
    g = first_step_grads(DATA, torch.device("cpu"), torch.float32, micro=[(4, 128)] * 3)
    assert g["rel_diff_bug_vs_full"] < 1e-5


def test_worked_example_numbers():
    ex = worked_example()
    assert abs(ex["correct"] - 2.6) < 1e-12 and abs(ex["bug"] - 3.0) < 1e-12
    assert abs(ex["error_pct"] - 15.38) < 0.01


def test_bits_of_point_one():
    expected = {"fp32": "0x3DCCCCCD", "bf16": "0x3DCD", "fp8_e4m3": "0x1D"}
    for name, (E, M, bias) in FORMATS.items():
        r = encode(0.1, E, M, bias)
        tb, _ = torch_bits(name)
        assert r["hex"] == expected[name]
        assert int(r["hex"], 16) == tb
    assert encode(0.1, 4, 3, 7)["stored"] == 0.1015625
    assert encode(0.1, 8, 7, 127)["stored"] == 0.10009765625
