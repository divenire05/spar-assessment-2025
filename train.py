from datasets import Dataset
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load dataset
with open("dataset/toy_dataset.json") as f:
    raw_data = json.load(f)

# Convert to Hugging Face dataset
dataset = Dataset.from_list(raw_data)

# Load tokenizer & model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Preprocessing
def tokenize(batch):
    inputs = tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=64)
    labels = tokenizer(batch["response"], truncation=True, padding="max_length", max_length=64)
    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

# Training args
args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=15,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)
trainer.train()

# Save finetuned model
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

# Plot training loss curve
logs = trainer.state.log_history
losses = [l["loss"] for l in logs if "loss" in l]
steps = list(range(len(losses)))

plt.plot(steps, losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Training Step")
plt.ylabel("Cross-Entropy Loss")
plt.savefig("train_loss.png")
print("Training loss curve saved to train_loss.png")