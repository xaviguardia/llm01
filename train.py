from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_id = "mistralai/Mistral-7B-v0.1"
dataset_path = "./data/qa_dataset.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

dataset = load_dataset("json", data_files=dataset_path, split="train")

def tokenize(sample):
    return tokenizer(
        sample["prompt"] + "\n" + sample["response"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized = dataset.map(tokenize)

args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    fp16=True, # Descomenta si tu GPU lo soporta
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()
