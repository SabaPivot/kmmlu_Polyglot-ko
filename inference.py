from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from data import polyglot_prompt, four_to_choice, tokenize_function

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
kmmlu_name = "HAERAE-HUB/KMMLU"
datasets = load_dataset(kmmlu_name, "Biology") # Accounting
datasets["test"] = datasets["test"].map(four_to_choice)
test_datasets = datasets["test"].map(lambda x: polyglot_prompt(x, False))
test_datasets = test_datasets.remove_columns(['question', 'answer', 'A', 'B', 'C', 'D', 'Category', 'Human Accuracy', 'choices'])

test_datasets = test_datasets.map(
    lambda x: tokenize_function(x, tokenizer),
    batched=True,
    remove_columns=test_datasets.column_names
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained("/home/careforme.dropout/results/checkpoint-550", quantization_config=bnb_config, device_map="auto")

inputs = torch.tensor(test_datasets[5]["input_ids"]).unsqueeze(0).to('cuda')
attention_mask = torch.tensor(test_datasets[5]["attention_mask"]).unsqueeze(0).to('cuda')

outputs = model.generate(
    inputs,
    attention_mask=attention_mask,
    max_new_tokens=3,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

