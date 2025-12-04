import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel

# 1. Define paths
base_model_id = "bert-base-uncased"
local_adapter_path = "./lora_results"  # <--- Points to your unzipped folder

print("Loading Base Model...")
# Load the generic BERT model
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_id, 
    num_labels=2
)

print("Loading Your Fine-Tuned Adapter...")
# Load your LoRA weights (the 11 minutes of training you just did)
model = PeftModel.from_pretrained(base_model, local_adapter_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_adapter_path)

# 2. Create the Pipeline
# We map the labels 0/1 to "Negative/Positive" for easier reading
id2label = {0: "NEGATIVE", 1: "POSITIVE"}

pipe = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer,
    device=-1 # Use -1 for CPU, or 0 for GPU if you have one locally
)

# Interactive loop - keeps asking for input
# 3. Test it out
def classify_review(text):
    result = pipe(text)
    label_id = int(result[0]['label'].split('_')[-1]) # Extracts 0 or 1
    score = result[0]['score'] *100
    sentiment = id2label[label_id]
    
    print(f"Review: '{text}'")
    print(f"Sentiment: {sentiment} (Confidence: {score:.4f})\n")


while True:
    user_input = input("Enter a review (or 'quit' to exit): ")
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    classify_review(user_input)