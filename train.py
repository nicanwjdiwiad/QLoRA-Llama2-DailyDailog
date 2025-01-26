import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
from config import *
from data_processing import preprocess_function

# Load dataset
dataset_cat = load_dataset(dataset_name)
train_dataset = dataset_cat["train"]
val_dataset = dataset_cat["validation"]
test_dataset = dataset_cat["test"]

# Tokenization
train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=["dialog", "act", "emotion"])
val_tokenized = val_dataset.map(preprocess_function, batched=True, remove_columns=["dialog", "act", "emotion"])
test_tokenized = test_dataset.map(preprocess_function, batched=True, remove_columns=["dialog", "act", "emotion"])

# Tokenize further
train_tokenized = train_tokenized.map(lambda examples: tokenizer(examples["text"]), batched=True)
val_tokenized = val_tokenized.map(lambda examples: tokenizer(examples["text"]), batched=True)

# Load model
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Start training
trainer.train()

# Save the trained model
trainer.model.save_pretrained(new_model)
#评估模型效果
test_results = trainer.evaluate(eval_dataset=test_tokenized)
print(f"Test results: {test_results}")
