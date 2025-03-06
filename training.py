import torch
from model import BigramLanguageModel
from data_preprocessing import get_batch
from utils import estimate_loss


def train(data,model, optimiser, max_iters, eval_interval, eval_iters, batch_size, block_size):
    # training loop
    for iter in range(max_iters):

        # every once in a while evaulate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(data, model, eval_iters, batch_size, block_size)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

    # get a batch sample from data
    xb, yb = get_batch(data,batch_size,block_size,split='train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()
