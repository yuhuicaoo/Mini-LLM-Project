import torch
from data_preprocessing import get_batch
from model import BigramLanguageModel

@torch.no_grad()
def estimate_loss(data, model, eval_iters, batch_size, block_size, device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data,batch_size,block_size,device,split=split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def decode_data(data, tokeniser):
    return tokeniser.decode(data)