import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- 1. Configuration ---
MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat" # Base model for QLoRA fine-tuning
OUTPUT_DIR = "../models/qwen_qlora_adapter"
DATASET_PATH = "../data/qwen_tool_data.jsonl" 

# Training Parameters
CONTEXT_LENGTH = 2048 # Max sequence length
LEARNING_RATE = 1e-4
BATCH_SIZE = 1 
GRADIENT_ACCUMULATION_STEPS = 4 
EPOCHS = 3 
# Dynamic BF16/FP16 check for hardware compatibility
BF16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

# LoRA Parameters (Optimized for Qwen architecture)
LORA_R = 32
LORA_ALPHA = 64 
LORA_DROPOUT = 0.05
# Target modules for Qwen 1.5 attention and feed-forward gates
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- 2. Quantization Configuration (QLoRA) [14] ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
)

# --- 3. Model and Tokenizer Loading [16] ---
print(f"Loading Qwen model {MODEL_ID} with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if BF16 else torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Set padding tokens, crucial for batched training with SFTTrainer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right" 
    
# Prepare model for k-bit training (re-casts necessary layers like LayerNorm to full precision) [15]
model = prepare_model_for_kbit_training(model) 

# --- 4. PEFT Configuration (LoRA) [17] ---
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)

# --- 5. Data Loading and Formatting [18] ---
# Load the fine-tuning data
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Define the formatting function to apply Qwen's ChatML template
def formatting_function(example):
    # Apply ChatML structure required by Qwen [18]
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

# --- 6. Training Arguments and Trainer Initialization ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=0.03,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no", 
    bf16=BF16,
    fp16=not BF16, 
    optim="paged_adamw_8bit", # Utilizing Paged Optimizers for memory efficiency [15]
    report_to="wandb",
    max_seq_length=CONTEXT_LENGTH,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=CONTEXT_LENGTH,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=formatting_function,
    packing=False,
)

# --- 7. Execution and Saving ---
if __name__ == "__main__":
    print("\nStarting QLoRA fine-tuning...")
    trainer.train()

    # Save the final adapter weights [19]
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

    print(f"\nQLoRA training complete. Adapters saved to {OUTPUT_DIR}/final_adapter")