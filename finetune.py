import os
import torch
import transformers 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
import peft
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_dataset
import pandas as pd
from tqdm import tqdm

# Set seed for reproducibility
set_seed(42)

# Basic configuration
model_id = "Qwen/Qwen2-0.5B"
output_dir = "./lora-qwen-physio"

# LoRA Configuration
lora_config = peft.LoraConfig(
    r=8,                      # Rank dimension
    lora_alpha=16,            # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,         # Dropout probability for LoRA layers
    bias="none",              # Don't train bias parameters
    task_type=TaskType.CAUSAL_LM  # This is for causal language modeling
)

print("Loading base model...")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # Use float32 for training
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Get the LoRA model
print("Applying LoRA adapters...")
model = peft.get_peft_model(model, lora_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")

# Prepare your dataset
# This is where you'd load your physiotherapy documents
def prepare_dataset():
    # Load your 3 research papers
    sample_data = [
        # Keep your existing content but expand with more context from your papers
        """Rhythmic stabilization is a therapeutic exercise technique used in physiotherapy for lower back pain. 
        It involves alternating isometric contractions of antagonistic muscles around a joint to promote stability.
        For the lower back, this often involves contracting the abdominal muscles and lower back muscles in an alternating pattern.
        This helps develop co-contraction abilities and improves neuromuscular control of the spine.""",
        
        # Add more content from your papers
        """Rhythmic stabilization is particularly important for patients with lower back pain because it addresses core weakness
        and improves proprioception. The technique helps retrain deep stabilizing muscles like the transversus abdominis and multifidus
        which are often inhibited or weakened in patients with chronic low back pain. Regular practice of rhythmic stabilization exercises
        has been shown to reduce pain levels and improve functional outcomes in clinical studies.""",
        
        # Add more content from your papers
        """To perform rhythmic stabilization for lower back pain, a patient typically starts in a neutral spine position.
        The physiotherapist provides resistance in different directions while the patient maintains their position.
        As the patient improves, exercises can progress to more challenging positions such as quadruped, sitting, or standing.
        A typical program might include 3 sets of 10 alternating contractions, held for 5-10 seconds each, performed daily."""
    ]
    
    # Create MORE diverse instruction-response pairs for better generalization
    formatted_data = []
    
    # Create more variations with specific questions
    questions = [
        "Explain what rhythmic stabilization is for lower back pain.",
        "How should rhythmic stabilization exercises be performed for patients with lower back pain?",
        "Why is rhythmic stabilization beneficial for treating lower back pain?",
        "What muscles are involved in rhythmic stabilization for lower back pain?",
        "How frequently should rhythmic stabilization exercises be performed?",
        "What are the clinical outcomes of rhythmic stabilization therapy?",
        "How does rhythmic stabilization compare to other stabilization techniques?",
        "What are contraindications for rhythmic stabilization exercises?",
        "Describe the progression of rhythmic stabilization exercises.",
        "How can rhythmic stabilization be modified for elderly patients?"
    ]
    
    # Generate more examples by combining each question with each data point
    for doc in sample_data:
        for question in questions:
            formatted_data.append({
                "instruction": question,
                "response": doc
            })
    
    return Dataset.from_pandas(pd.DataFrame(formatted_data))

# Create and process dataset
print("Preparing dataset...")
dataset = prepare_dataset()

# Process the dataset
def process_data(examples):
    texts = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        # Format as a chat/instruction example
        texts.append(f"USER: {instruction}\nASSISTANT: {response}")
    
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Create input_ids and labels for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

processed_dataset = dataset.map(
    process_data,
    batched=True,
    remove_columns=dataset.column_names
)

# Split dataset
train_val_split = processed_dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split["train"]
eval_dataset = train_val_split["test"]

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not using masked language modeling
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,       # Increase from 3 to 5 for better learning
    per_device_train_batch_size=2,  # Reduce batch size for CPU
    per_device_eval_batch_size=2,   # Reduce batch size for CPU
    gradient_accumulation_steps=16, # Increased for effective batch size
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    learning_rate=1e-4,       # Slightly reduced learning rate
    fp16=False,               # Keep this off for CPU
    report_to="none",
    push_to_hub=False,
    load_best_model_at_end=True,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the trained model
print("Saving model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Training complete! Model saved to {output_dir}") 