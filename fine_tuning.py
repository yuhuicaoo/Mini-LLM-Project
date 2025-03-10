from transformers import AutoModelForCausalLM, AutoTokenizer
from data_preprocessing import load_lyrics
import torch
import tiktoken
from config import max_iters, eval_interval, eval_iters, batch_size, block_size, learning_rate
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
            outputs = model(X, labels=Y)
            losses[k] = outputs.loss

            total_loss += outputs.loss * X.shape[1]
            num_tokens += X.shape[1]
        
        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))


        out[split] = {'loss': losses.mean(), "perplexity": perplexity.item()}
    model.train()
    return out

split_chunk = lambda data: data.item().split(1024)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokeniser = AutoTokenizer.from_pretrained(model_name)
tokeniser2 = tiktoken.get_encoding(model_name)


lyrics = load_lyrics("data/kendrick_lamar_lyrics.txt")
data = tokeniser.encode(lyrics, truncation=True, return_tensors="pt").squeeze()
data2 = torch.tensor(tokeniser.encode(lyrics), dtype=torch.long)
print(data.shape)
print(data2.shape)
data3 = split_chunk(data)
print(data3)

# optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# model.train()
# for iter in range(max_iters+1):

#     # every once in a while evaulate the loss on train and val sets
#     if iter % eval_interval == 0:
#         metrics = estimate_metrics(data, model, eval_iters, batch_size, block_size, device)
#         print(
#             f"step {iter}: train loss {metrics['train']['loss']:.4f}, train ppl {metrics['train']['perplexity']:.2f}, val loss {metrics['val']['loss']:.4f}, val ppl {metrics['val']['perplexity']:.2f}"
#         )

#     # get a batch sample from data
#     xb, yb = get_batch(data, batch_size, block_size, device, split='train')

#     # evaluate loss
#     outputs = model(xb, labels=yb)
#     loss = outputs.loss
#     optimiser.zero_grad(set_to_none=True)
#     loss.backward()
#     optimiser.step()

# torch.save(model.state_dict(), "models/fine_tuned_gpt2.pth")