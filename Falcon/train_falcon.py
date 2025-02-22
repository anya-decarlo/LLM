import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np
import json
import os

# ---- SETUP ---- #
MODEL_NAME = "tiiuae/falcon-7b"  # Smallest Falcon model for training
DATASET_NAME = "wikitext"  # Controlled dataset to measure RMI
LOGS_DIR = "falcon_training_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# ---- LOAD MODEL & TOKENIZER ---- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ---- LOAD DATASET ---- #
dataset = load_dataset(DATASET_NAME, "wikitext-2-raw-v1", split="train")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
dataset = dataset.map(tokenize_function, batched=True)

# ---- ENTROPY FUNCTION ---- #
def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)  # Shannon entropy
    return entropy.mean().item()

# ---- TRAINING LOGGING ---- #
def training_step_with_entropy(model, batch):
    outputs = model(**batch)
    logits = outputs.logits
    entropy = compute_entropy(logits)  # Compute entropy per batch
    return outputs.loss, entropy

# ---- CUSTOM TRAINER ---- #
class FalconTrainer(Trainer):
    def training_step(self, model, batch):
        loss, entropy = training_step_with_entropy(model, batch)
        self.log({"entropy": entropy})  # Log entropy at each step
        return loss

# ---- TRAINING ARGS ---- #
training_args = TrainingArguments(
    output_dir=LOGS_DIR,
    num_train_epochs=1,  # Adjustable
    per_device_train_batch_size=4,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="no",
    report_to="none",
)

# ---- RUN TRAINING ---- #
trainer = FalconTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()