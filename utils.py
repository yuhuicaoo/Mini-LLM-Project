import torch
from data_preprocessing import get_batch

@torch.no_grad()
def estimate_metrics(data, model, eval_iters, batch_size, block_size, device):
    out = {}
    total_loss , num_tokens = 0, 0
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data,batch_size,block_size,device,split=split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

            total_loss += loss.item() * X.shape[1]
            num_tokens += X.shape[1]
        
        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))


        out[split] = {'loss': losses.mean(), "perplexity": perplexity.item()}
    model.train()
    return out

def decode_data(data, tokeniser):
    return tokeniser.decode(data)