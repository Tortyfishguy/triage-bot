import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

def classify_esi(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item() + 1
