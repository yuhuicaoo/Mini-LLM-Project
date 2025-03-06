import torch
from model import BigramLanguageModel
from training import train
from data_preprocessing import load_lyrics, encode_data, decode_data
from inference import generate
import tiktoken

# Hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 16  # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 250
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
num_embd = 64
num_heads = 4
num_layers = 4
dropout = 0.2
# ---------------------

torch.manual_seed(42)

lyrics = load_lyrics("kendrick_lamar_lyrics.txt")
tokeniser = tiktoken.get_encoding("gpt2")
data = torch.tensor(encode_data(lyrics, tokeniser), dtype=torch.long)
vocab_size = tokeniser.n_vocab

model = BigramLanguageModel(
    vocab_size, 
    num_embd, 
    block_size, 
    num_heads, 
    num_layers, 
    dropout, 
    device
)
m = model.to(device)

optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train(
    data, 
    model, 
    optimiser, 
    max_iters, 
    eval_interval, 
    eval_iters, 
    batch_size, 
    block_size,
)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_output = decode_data(
    generate(model, block_size, context, max_new_tokens=500)[0].tolist(),
    tokeniser,
)

with open("generated_output.txt", "w", encoding="utf-8") as f:
    f.write(generated_output)
