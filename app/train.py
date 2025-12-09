import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model, TaskType

# CONFIG

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "chatbot_dataset.jsonl"  
OUTPUT_DIR = "./lora-phi3-m123ni"

print("Training on GPU:" if torch.cuda.is_available() else "Training on CPU")

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = load_dataset("json", data_files=DATASET_PATH)["train"].train_test_split(test_size=0.2)
train_dataset, eval_dataset = dataset["train"], dataset["test"]

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Format dataset (Instruction + Output clearly separated)
def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Output:\n{example['output']}"
    }

train_dataset = train_dataset.map(format_example)
eval_dataset = eval_dataset.map(format_example)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# -----------------------------
# LOAD MODEL w/ QUANTIZATION
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

# -----------------------------
# LORA CONFIG
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# -----------------------------
# TRAINING ARGS
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False
)


# -----------------------------
# TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -----------------------------
# TRAIN
# -----------------------------
trainer.train()

# -----------------------------
# EVALUATION (Loss + Perplexity)
# -----------------------------
eval_results = trainer.evaluate()
print(" Eval results:", eval_results)
print(" Perplexity:", math.exp(eval_results["eval_loss"]))

# -----------------------------
# Q&A ACCURACY CHECK
# -----------------------------
print("\n Running Q&A Accuracy Check...")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

correct, total = 0, 0
for ex in dataset["test"]:
    inp = ex["instruction"]
    expected = ex["output"].strip()
    pred = pipe(inp, max_new_tokens=100, do_sample=False)[0]["generated_text"]

    if expected in pred:
        correct += 1
    total += 1

print(f" Q&A Accuracy: {correct}/{total} = {correct/total:.2%}")

# -----------------------------
# SAVE ADAPTER
# -----------------------------
model.save_pretrained(OUTPUT_DIR)
print("Training complete. LoRA adapter saved at:", OUTPUT_DIR)
