#!/usr/bin/env python3
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.up = nn.Linear(hidden, ffn, bias=False)
        self.gate = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        return self.down(self.up(x) * torch.nn.functional.silu(self.gate(x)))


class MiniBlock(nn.Module):
    def __init__(self, hidden, heads, ffn):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = RMSNorm(hidden)
        self.ffn = SwiGLU(hidden, ffn)

    def forward(self, x):
        n1 = self.norm1(x)
        x = x + self.attn(n1, n1, n1, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class MiniLM(nn.Module):
    def __init__(self, vocab=32000, layers=8, hidden=768, heads=12, ffn=3072):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([MiniBlock(hidden, heads, ffn) for _ in range(layers)])
        self.norm = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


if __name__ == "__main__":
    model = MiniLM()
    total = sum(p.numel() for p in model.parameters())
    print({"params": total})
