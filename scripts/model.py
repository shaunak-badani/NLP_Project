from datasets import load_dataset
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, \
        TrainingArguments, Trainer
import torch
import os

dataset = load_dataset("mteb/tweet_sentiment_extraction")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
   return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

weights_enabled = set(['transformer.wte.weight'])

params_before = {}
for name, param in model.named_parameters():
    params_before[name] = param.clone()
    param.requires_grad = name in weights_enabled


optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=5e-5
)

training_args = TrainingArguments(
   output_dir="test_trainer",
   #evaluation_strategy="epoch",
   per_device_train_batch_size=1,  # Reduce batch size here
   per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
   gradient_accumulation_steps=4,
   num_train_epochs = 1,
)

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=small_train_dataset,
   eval_dataset=small_eval_dataset,
   optimizers=(optimizer, None)
)

trainer.train()

model_dir = "models/"
os.makedirs(model_dir, exist_ok = True)
torch.save(model.state_dict(), "models/fine_tuned_model.pt")