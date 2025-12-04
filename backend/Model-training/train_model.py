import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from peft import get_peft_model, LoraConfig, TaskType

# 1️⃣ Load CSV
df = pd.read_csv("../Dataset/review_cleaned.csv")

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
train_ds = dataset["train"]
test_ds = dataset["test"]

# Rename target column to 'labels' (required)
train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

# 2️⃣ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 3️⃣ Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # sequence classification
    r=16,                          #rank of low-rank matrices
    lora_alpha=32,               #scaling factor
    lora_dropout=0.05,
    target_modules=["query", "value"],  # LoRA applied only to attention layers
)

model = get_peft_model(model, lora_config)

# 4️⃣ Tokenization
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Remove raw text
train_ds = train_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])

# Convert to torch tensors
train_ds.set_format("torch")
test_ds.set_format("torch")

# 5️⃣ Training Arguments
training_args = TrainingArguments(
    output_dir="./lora_results",
    eval_strategy="epoch",
    learning_rate=2e-4,         # higher LR works well for LoRA
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
)

# 6️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

# 7️⃣ Train
trainer.train()

# 8️⃣ Save LoRA-adapted model
model.save_pretrained("./lora-bert")
tokenizer.save_pretrained("./lora-bert")

# 9️⃣ Test with pipeline
pipe = pipeline("text-classification", model="./lora-bert", tokenizer="./lora-bert")
print(pipe("This movie was absolutely wonderful!"))
