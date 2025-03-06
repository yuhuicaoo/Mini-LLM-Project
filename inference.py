import torch
import torch.nn.functional as F
from model import BigramLanguageModel


def generate(model, block_size, idx, max_new_tokens):
    # idx is (B,T) tensor of indicies in the current context
    for _ in range(max_new_tokens):
        # crop idx/ context  to the last block_size tokens / never pass in more than block_size elements
        idx_cond = idx[:, -block_size:]
        # get predictions
        logits, _ = model(idx_cond)
        # only consider last element in the time dimension, becomes (B,C)
        logits = logits[:, -1, :]
        # apply softmax to get proababilities
        probs = F.softmax(logits, dim=-1)  # (B,C)
        # sample from the distribution, only gets 1 prediction so becomes (B,1)
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx
