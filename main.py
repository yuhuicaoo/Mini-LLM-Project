import torch
from model import BigramLanguageModel
from data_preprocessing import load_lyrics, get_batch
from utils import estimate_loss
import tiktoken
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(42)

lyrics = load_lyrics("kendrick_lamar_lyrics.txt")
tokeniser = tiktoken.get_encoding("gpt2")
encode = lambda s: tokeniser.encode(s)
vocab_size = tokeniser.n_vocab

data = torch.tensor(encode(lyrics), dtype=torch.long)

#intialise model and optimiser
model = BigramLanguageModel(vocab_size, num_embd, num_heads,num_layers, block_size, dropout, device)
model = model.to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# training loop
for iter in range(max_iters):

    # every once in a while evaulate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(data, model, eval_iters, batch_size, block_size, device)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # get a batch sample from data
    xb, yb = get_batch(data, batch_size, block_size, device, split='train')

    # evaluate loss
    _, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

torch.save(model.state_dict(), "models/model2.pth")