from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


dataset = load_dataset("huggingartists/kendrick-lamar")

tokeniser = AutoTokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token

# create a function that tokenises all the data
def tokenise_function(examples):
    encoding = tokeniser(examples["text"], padding=True, truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = tokeniser.pad_token_id

    encoding["labels"] = labels
    return encoding

tokenised_dataset = dataset.map(tokenise_function, batched=True)
train_dataset = tokenised_dataset["train"].shuffle(seed=42)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    weight_decay=1e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
model.save_pretrained("./models")
