import torch
import torch.nn.functional as F
import tiktoken
from config import num_embd, num_heads, num_layers, block_size, dropout
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, device):
        
        model.eval()

        # idx is (B,T) tensor of indicies in the current context
        for _ in range(max_new_tokens):
            # crop idx/ context  to the last block_size tokens / never pass in more than block_size elements
            idx_cond = idx[:, -block_size:]
            # get predictions
            outputs = model(idx_cond)
            logits = outputs.logits
            # only consider last element in the time dimension, becomes (B,C)
            logits = logits[:, -1, :]
            # apply softmax to get proababilities between 0 to 1.
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample next token from the distribution, only gets 1 prediction so becomes (B,1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T) --> (B, T+1)
        model.train()
        return idx

device = "cuda" if torch.cuda.is_available() else "cpu"
tokeniser = AutoTokenizer.from_pretrained('gpt2')


model = AutoModelForCausalLM.from_pretrained("gpt2")
model.load_state_dict(torch.load('models/fine_tuned_gpt2.pth', weights_only=True))
model = model.to(device)

model.eval()

# generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
context = torch.tensor(tokeniser.encode("Write me a song \n"), dtype=torch.long, device=device).unsqueeze(0)
generated_output = tokeniser.decode(generate(model, context, max_new_tokens=500, block_size=block_size, device=device)[0].tolist())

with open("generated_outputs/gpt2_output.txt", 'w', encoding='utf-8') as f:
    f.write(generated_output)
