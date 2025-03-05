import os
import torch
import transformers
import peft
import data
import sklearn
from tqdm import tqdm
from dotenv import load_dotenv
import gc
import psutil

# Load environment variables
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration - CPU-optimized settings
model_id = "Qwen/Qwen2-0.5B"
output_dir = "./lora-qwen-physio"
num_epochs = 5
learning_rate = 1e-4
weight_decay = 0.01
batch_size = 1
grad_accum_steps = 4  # With 9 training examples, this will accumulate 2-3 times per epoch
max_grad_norm = 0.3
warmup_steps = 50
max_length = 128  # Further reduced sequence length for CPU
eval_interval = 10
seed = 0
r_value = 16  # Small LoRA rank to reduce memory

# Set seed for reproducibility
torch.manual_seed(seed)
transformers.set_seed(seed)

# Memory usage tracker (CPU version)
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"RAM used: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    gc.collect()

# LoRA Configuration - reduced parameters for CPU
lora_config = peft.LoraConfig(
    r=r_value,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Reduced target modules
    lora_dropout=0.05,
    bias="none",
    task_type=peft.TaskType.CAUSAL_LM
)

# Load model with CPU optimizations
print("Loading base model and tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Loading model (CPU optimized)...")
print_memory_usage()  # Check memory before loading

# Load model with minimal memory usage
base_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # Use regular float32 for CPU
    trust_remote_code=True,
    token=os.getenv("HF_TOKEN_CURSOR"),
    use_cache=False,
    low_cpu_mem_usage=True  # Enable low memory usage
)

print_memory_usage()  # Check memory after loading

# Apply LoRA adapters
print("Applying LoRA adapters...")
model = peft.get_peft_model(base_model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
print_memory_usage()

# Delete the base model reference to save memory
del base_model
gc.collect()

# Set model to training mode
model.train()

# Create dataset with reduced max length
print("Preparing dataset...")
train_q, val_q, train_a, val_a = sklearn.model_selection.train_test_split(
    data.questions, data.answers, test_size=0.1, random_state=seed
)

# Create datasets
train_dataset = data.PhysioDataset(train_q, train_a, tokenizer, max_length)
val_dataset = data.PhysioDataset(val_q, val_a, tokenizer, max_length)

# Create data loaders - no pinning for CPU
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=1
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=1
)

# Optimizer with CPU settings
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay,
    eps=1e-7  # Increased epsilon for stability
)

# LR scheduler
total_steps = max(1, len(train_loader) * num_epochs // grad_accum_steps)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

# Training loop
print("Starting training...")
global_step = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    # Training
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for step, batch in enumerate(train_pbar):
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / grad_accum_steps
        
        # Backward pass
        loss.backward()
        
        # Explicitly delete outputs to free memory
        del outputs
        gc.collect()
        
        # Update stats
        total_loss += loss.item() * grad_accum_steps
        train_pbar.set_postfix({"loss": loss.item() * grad_accum_steps})
        
        # Gradient accumulation and optimizer step
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Print memory usage occasionally
            if global_step % 5 == 0:
                print_memory_usage()
            
            # Evaluation
            if global_step % eval_interval == 0:
                model.eval()
                val_loss = 0
                
                # Validation loop
                eval_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Eval]")
                with torch.no_grad():
                    for eval_batch in eval_pbar:
                        outputs = model(**eval_batch)
                        val_loss += outputs.loss.item()
                        eval_pbar.set_postfix({"val_loss": outputs.loss.item()})
                        del outputs
                
                val_loss /= len(val_loader)
                print(f"Step {global_step} | Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                    model.save_pretrained(os.path.join(output_dir, "best_model"))
                
                gc.collect()
                model.train()
    
    # End of epoch
    epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {epoch_loss:.4f}")
    
    # Save checkpoint
    model.save_pretrained(os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}"))
    gc.collect()

# Save final model
print("Saving final model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Training complete! Model saved to {output_dir}") 