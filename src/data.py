"""Tiny Shakespeare, character level (the nanoGPT shakespeare_char data set)."""
import pathlib
import urllib.request

import torch

from src.model import GPT, GPTConfig

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PAD = -1  # nanoGPT's ignore_index: a target of -1 adds nothing to the loss

# S: small, for parts 1-4 (seconds per run). SHAKESPEARE_CHAR: nanoGPT's own config, for part 5.
CONFIG_S = dict(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.0, bias=True)
CONFIG_SHAKESPEARE_CHAR = dict(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.0, bias=True)


class CharData:
    def __init__(self):
        DATA.mkdir(exist_ok=True)
        path = DATA / "input.txt"
        if not path.exists():
            urllib.request.urlretrieve(URL, path)
        text = path.read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)  # 65
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(ids))
        self.train, self.val = ids[:n], ids[n:]

    def get_batch(self, B, T, split="train", gen=None, device="cpu"):
        """B windows of T+1 characters: x = the first T, y = the same window moved one place right."""
        src = self.train if split == "train" else self.val
        ix = torch.randint(len(src) - T - 1, (B,), generator=gen)
        x = torch.stack([src[i:i + T] for i in ix])
        y = torch.stack([src[i + 1:i + T + 1] for i in ix])
        return x.to(device), y.to(device)

    def get_ragged_batch(self, B, T, length, split="train", gen=None, device="cpu"):
        """B sequences that hold only `length` real tokens each; the rest of the T positions is padding.
        Padding targets are PAD (-1), so the loss ignores them."""
        x, y = self.get_batch(B, T, split, gen)
        x[:, length:] = 0
        y[:, length:] = PAD
        return x.to(device), y.to(device)


def make_model(cfg, vocab_size, device="cpu", seed=1337):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=vocab_size, **cfg)).to(device)
