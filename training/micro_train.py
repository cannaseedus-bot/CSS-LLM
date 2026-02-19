#!/usr/bin/env python3
import torch
import torch.nn.functional as F


def micro_train_step(model, optimizer, batch, vocab_size):
    logits = model(batch[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def make_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
