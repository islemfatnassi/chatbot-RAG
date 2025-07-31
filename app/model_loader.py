# app/model_loader.py
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_lora_model(base_model_name, lora_path):
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer
