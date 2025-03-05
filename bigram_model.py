import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparamters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
num_embd = 32
# ---------------------

torch.manual_seed(42)

# open and read input file
with open("kendrick_lamar_lyrics.txt", "r", encoding="utf-8") as f:
    lyrics = f.read()

# check all the unique characters that occur in the dataset
chars = sorted(list(set(lyrics)))
vocab_size = len(chars)

# create a mapping from characters to integers (encoder) and vice-versa (decoder)
str_to_int = {char: i for i, char in enumerate(chars)}
int_to_str = {i: char for i, char in enumerate(chars)}

# encoder takes a string which outputs a list of integers ,
# characters in string are converted to int via lookup table
encode = lambda s: [str_to_int[c] for c in s]

# decoder takes a list of integers and outputs a string, integers converted via lookup table
decode = lambda l: "".join([int_to_str[i] for i in l])


# Train and validation splits
data = torch.tensor(encode(lyrics), dtype=torch.long)
# split 90% of our data into train and have remaining 10% as our validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data for inputs x and targets y
    data = train_data if split == "train" else val_data

    # generate a tensor of random integers, that represent the starting position of each sequence of data
    batch_start_indicies = torch.randint(
        high=(len(data) - block_size), size=(batch_size,)
    )

    # creates a tensor: x and y where each element is a sequence of block_size, stack the 1-D tensors as rows
    # creating a batch_size x block_size tensors (e.g 4x8 tensor)
    x = torch.stack(
        [data[index : index + block_size] for index in batch_start_indicies]
    )
    y = torch.stack(
        [data[index + 1 : index + block_size + 1] for index in batch_start_indicies]
    )
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# setup self attention head class
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embd, head_size, bias=False)
        self.query = nn.Linear(num_embd, head_size, bias=False)
        self.value = nn.Linear(num_embd, head_size, bias=False)

        # ensures that tril behaves like a constant tensor that follows the model but isnt trained.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        
        # compute attention scores ("affinities" between tokens) using "scaled-attention"
        weights = q @ torch.transpose(k, dim0=1, dim1=2) * C**-0.5 # (B,T, head_size) @ (B, head_size, T) -> (B,T,T) | Note: head_size == attention_dimension
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T ,T)

        # perform weighted aggregation on the values , v
        v = self.value(x) # (B,T,head_size)
        out = weights @ v # (B,T,T) @ (B,T,head_size) --> (B,T,head_size)
        return out


# Create a simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # maps each token in the sequence to its next-token prediction in the form of logits
        self.token_embedding_table = nn.Embedding(vocab_size, num_embd)
        self.position_embedding_table = nn.Embedding(block_size, num_embd)
        self.sa_head = Head(num_embd)
        self.lm_head = nn.Linear(num_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        B,T = idx.shape

        token_embd = self.token_embedding_table(idx)  # (B,T,C) or (B,T,num_embed) , where C = num_embd = embedding dimension
        position_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) --> becomes (B,T,C) when adding to token_embd
        x = token_embd + position_embd # (B,T,C) , x holds both token identity and positions of where the tokens occur
        x = self.sa_head(x) # apply one head of self-attention (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            # if no target, just get the logits
            loss = None
        else:
            # since we have a multi-dimensional input, PyTorch cross_entropy function expects the logit tensor shape to be (B*T,C) and not (B,T,C)
            # and the target tensor to be 1D. Therefore we have to reshape our logit and target tensors
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # want it to be 2-D Tensor
            targets = targets.view(B * T)  # want it to be 1-D Tensor

            # measures the quality of the logits w.r.t the targets,
            # how well are we predicting the next character based on the logits
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) tensor of indicies in the current context
        for _ in range(max_new_tokens):
            # crop idx/ context  to the last block_size tokens / never pass in more than block_size elements
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, _ = self(idx_cond)
            # only consider last element in the time dimension, becomes (B,C)
            logits = logits[:, -1, :]
            # apply softmax to get proababilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution, only gets 1 prediction so becomes (B,1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# intialise optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    # every once in a while evaulate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # get a batch sample from data
    xb, yb = get_batch("train")

    # evaluate loss
    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
