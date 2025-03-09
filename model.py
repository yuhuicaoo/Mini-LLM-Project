import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, num_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(num_embd, head_size, bias=False)
        self.query = nn.Linear(num_embd, head_size, bias=False)
        self.value = nn.Linear(num_embd, head_size, bias=False)

        # ensures that tril behaves like a constant tensor that follows the model but isnt trained.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout) # dropout layers for regularisation

    def forward(self, x):
        # Batch, Time , CHannel
        _,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        
        # compute attention scores ("affinities" between tokens) using "scaled-attention"
        weights = q @ torch.transpose(k, dim0=1, dim1=2) * k.shape[-1]**-0.5 # (B,T, head_size) @ (B, head_size, T) -> (B,T,T) | Note: head_size == attention_dimension
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T ,T)
        weights = self.dropout(weights)   # randomly prevent some of the nodes / tokens from communicating
        # perform weighted aggregation on the values , v
        v = self.value(x) # (B,T,head_size)
        out = weights @ v # (B,T,T) @ (B,T,head_size) --> (B,T,head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multi-headed self-attention module"""

    def __init__(self, num_heads, num_embd, head_size, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, num_embd, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(num_embd, num_embd)
        self.dropout = nn.Dropout(dropout) # dropout layers for regularisation

    def forward(self, x):
        # concatenate on the channel dimension 
        out_sa = torch.cat([head(x) for head in self.heads], dim=-1) # output of the self-attention
        # apply projection (linear transformation) to the self-attention output, projection back to the residual pathway
        out = self.dropout(self.projection(out_sa)) 
        return out

class FeedForwardNetwork(nn.Module):
    """a simple feed-forward network, with ReLU activation"""

    def __init__(self, num_embd, dropout):
        super().__init__()
        self.network = nn.Sequential(
            # apply linear layer on the per-token level, all tokens do this independently
            nn.Linear(num_embd, 4 * num_embd),
            nn.ReLU(),
            nn.Linear(4 * num_embd, num_embd), # projection (linear transformation) layer, back into the residual pathway
            nn.Dropout(dropout),   # regularisation by adding dropout layers
        )

    def forward(self, x):
        return self.network(x)
    
class Block(nn.Module):
    """Transformer block : communication (attention) followed by computation"""

    def __init__(self, num_embd, num_heads, dropout, block_size):
        # num_embd : embedding dimension, num_heads : the number of heads we'd like
        super().__init__()
        # Split the embedding size evenly across attention heads to ensure parallel processing and efficient computation.
        head_size = num_embd // num_heads
        # communicaton done through multi-headed self-attention
        self.self_attention = MultiHeadAttention(num_heads, num_embd, head_size, dropout, block_size)
        # computation done by feed-forward network on all tokens independently
        self.feedforward = FeedForwardNetwork(num_embd, dropout)
        # add layer-normalisation, for 0-mean and unit(1)-variance across the rows. 
        self.layer_norm1 = nn.LayerNorm(num_embd)
        self.layer_norm2 = nn.LayerNorm(num_embd)


    def forward(self, x):
        # add x to output of communication and computation (residual connections) and
        # apply layer normalisation before the transformations (pre-norm formulation)
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feedforward(self.layer_norm2(x))
        return x
    
class BigramLanguageModel(nn.Module):
    """Main model class"""
    def __init__(self, vocab_size, num_embd, block_size, num_heads, num_layers, dropout, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # maps each token in the sequence to its next-token prediction in the form of logits
        self.token_embedding_table = nn.Embedding(vocab_size, num_embd)
        self.position_embedding_table = nn.Embedding(block_size, num_embd)
        self.blocks = nn.Sequential(*[Block(num_embd, num_heads, dropout, block_size) for _ in range(num_layers)])
        self.layer_norm_final = nn.LayerNorm(num_embd)
        self.lm_head = nn.Linear(num_embd, vocab_size)

        self.device = device        

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        B,T = idx.shape

        token_embd = self.token_embedding_table(idx)  # (B,T,C) or (B,T,num_embed) , where C = num_embd = embedding dimension
        position_embd = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C) --> becomes (B,T,C) when adding to token_embd
        x = token_embd + position_embd # (B,T,C) , x holds both token identity and positions of where the tokens occur
        x = self.blocks(x) # (B,T,C)
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