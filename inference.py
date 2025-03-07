import torch
from bigram_model import BigramLanguageModel
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"
tokeniser = tiktoken.get_encoding('gpt2')


model = BigramLanguageModel(tokeniser.n_vocab)
model.load_state_dict(torch.load('model.pth', weights_only=True))
model = model.to(device)

model.eval()

# generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
context = torch.tensor(tokeniser.encode("Write me a song \n"), dtype=torch.long, device=device).unsqueeze(0)
generated_output = tokeniser.decode(model.generate(context, max_new_tokens=500)[0].tolist())

with open("generated_output.txt", 'w', encoding='utf-8') as f:
    f.write(generated_output)
