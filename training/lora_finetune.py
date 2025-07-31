# training/lora_finetune.py
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset

model_name = "tiiuae/falcon-7b-instruct"
dataset = load_dataset("json", data_files="dataset/train_qa.json")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(model, config)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./results", per_device_train_batch_size=2, num_train_epochs=3),
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained("./lora_model")

