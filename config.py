# Model parameters
model_name = "NousResearch/llama-2-7b-chat-hf"
new_model = "llama-2-7b-QLoRA-emotion"
dataset_name = "li2017dailydialog/daily_dialog"

# QLoRA parameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# BitsAndBytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# TrainingArguments parameters
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 8
per_device_eval_batch_size = 8
gradient_accumulation_steps = 4
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 9e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 25

# SFT parameters
max_seq_length = None
packing = False
device_map = {"": 0}
