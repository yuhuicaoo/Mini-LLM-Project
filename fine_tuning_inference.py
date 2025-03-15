from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

tokeniser = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("models/gpt2_final_model").to(device)
input = tokeniser("Write me a Kendrick Lamar song\n", return_tensors="pt").to(device)

sample_output = model.generate(
    input_ids = input['input_ids'],
    attention_mask=input['attention_mask'],
    pad_token_id=tokeniser.eos_token_id,
    do_sample=True,
    max_length=500,
    top_k=50,
    top_p=0.9,
    repetition_penalty= 1.2,
    num_beams=5,
    temperature=0.7,
    no_repeat_ngram_size=2
)

generated_text = tokeniser.decode(sample_output[0], skip_special_tokens=True)

with open("generated_outputs/fine_tuned_output.txt", 'w', encoding='utf-8') as f:
    f.write(generated_text)