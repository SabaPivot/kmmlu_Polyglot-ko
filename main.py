import torch
from datasets import load_dataset
from data import four_to_choice, polyglot_prompt, tokenize_function
from memory import show_memory
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb



wandb.init(
    project="KMMLU",
    config={
    "architecture": "Polyglot",
    "dataset": "KMMLU",
    "epochs": 3,
    }
)

# Load and preprocess the dataset
kmmlu_name = "HAERAE-HUB/KMMLU"
model_name = "EleutherAI/polyglot-ko-1.3b" # EleutherAI/polyglot-ko-5.8b / EleutherAI/polyglot-ko-1.3b
max_seq_length = 2048
# Dataset preparation
datasets = load_dataset(kmmlu_name, "Biology") # Accounting
print(datasets)

datasets = datasets.map(four_to_choice)
train_datasets = datasets["train"].map(lambda x: polyglot_prompt(x, True))
eval_datasets = datasets["dev"].map(lambda x: polyglot_prompt(x, False))
test_datasets = datasets["test"].map(lambda x: polyglot_prompt(x, False))

columns_to_remove = ['question', 'answer', 'A', 'B', 'C', 'D', 'Category', 'Human Accuracy', 'choices']
train_datasets = train_datasets.remove_columns(columns_to_remove)
eval_datasets = eval_datasets.remove_columns(columns_to_remove)
test_datasets = test_datasets.remove_columns(columns_to_remove)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize datasets
train_datasets = train_datasets.map(
    lambda x: tokenize_function(x, tokenizer),
    batched=True,
    remove_columns=train_datasets.column_names
)
eval_datasets = eval_datasets.map(
    lambda x: tokenize_function(x, tokenizer),
    batched=True,
    remove_columns=eval_datasets.column_names
)

test_datasets = test_datasets.map(
    lambda x: tokenize_function(x, tokenizer),
    batched=True,
    remove_columns=test_datasets.column_names
)

# Modified configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Model configuration
model_kwargs = {
    "use_cache": False,
    "device_map": "auto"
}

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    **model_kwargs
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        'query_key_value',
        'dense',
        'dense_h_to_4h',
        'dense_4h_to_h'
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

# Prepare the model with LoRA
model = get_peft_model(model, lora_config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

lr = 2e-4
# model.gradient_checkpointing_enable()

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=10,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    learning_rate=lr,
    gradient_checkpointing=False,
    optim="paged_adamw_8bit",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=0.3,
    remove_unused_columns=False,
    bf16=True,
    report_to="wandb",
    run_name=f"{model_name}_{lr}"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_datasets,
    eval_dataset=eval_datasets,
    data_collator=data_collator
)

show_memory()
trainer.train()

show_memory()
