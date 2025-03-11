from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained("./models")
tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token

model.eval()
model = model.to(device)


inputs = tokeniser("Write me a song \n", return_tensors="pt", padding=True).to(device)
attention_mask = inputs['attention_mask']

with torch.no_grad():
    output = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=attention_mask,
        max_length=500,
        pad_token_id=tokeniser.eos_token_id
    )

generated_text = tokeniser.decode(output[0])

with open("generated_outputs/gpt2_output.txt", 'w', encoding='utf-8') as f:
    f.write(generated_text)
