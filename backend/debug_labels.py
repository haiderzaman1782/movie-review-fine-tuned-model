from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel
import torch

# 1. Load Model WITHOUT custom labels first
base_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)
model = PeftModel.from_pretrained(base_model, "./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

# 2. Create Pipeline
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 3. Test
print("--- DEBUG RESULT ---")
text = "This movie was absolutely wonderful! Best movie ever."
result = pipe(text)
print(f"Input: {text}")
print(f"Raw Model Output: {result}")