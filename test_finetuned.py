import torch
import transformers
from peft import PeftModel, PeftConfig
import time
import os

# Path to your best fine-tuned LoRA adapter
base_dir = "./lora-qwen-physio"
best_model_path = os.path.join(base_dir, "best_model")  # Use the best model saved during training
model_id = "Qwen/Qwen2-0.5B"

print(f"Loading base model and LoRA adapter from {best_model_path}...")
# Load tokenizer from the base model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load the base model
print("Loading base model...")
base_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map="cpu",
    low_cpu_mem_usage=True
)

# Load LoRA configuration and model
print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, best_model_path)
model.eval()
print("Model loaded successfully!")

# Add environment variable to avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Test prompts
test_prompts = [
    "What is rhythmic stabilization and how does it help with lower back pain?",
    "Can you give me some rhythmic stabilization exercises for lumbar spine?",
    "Why is core stability important for lower back pain management?",
    # Add a new prompt to test generalization
    "What is the difference between rhythmic stabilization and static stabilization?"
]

# Generate responses
for i, prompt in enumerate(test_prompts):
    print(f"\n\nTest {i+1}: {prompt}")
    
    # Format with Qwen2 chat template
    # This uses the proper Qwen2 format with <|im_start|> and <|im_end|> tokens
    input_text = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    
    # Generate
    print("Generating base model response...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Time measurement
    elapsed = time.time() - start_time
    
    # Extract output
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    print(f"Generation took {elapsed:.2f} seconds")
    print("\n===== BASE MODEL RESPONSE =====")
    print(response.strip())
    print("===========================") 

    # Generate
    print("Generating finetuned model response...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Time measurement
    elapsed = time.time() - start_time
    
    # Extract output
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    print(f"Generation took {elapsed:.2f} seconds")
    print("\n===== FINETUNED MODEL RESPONSE =====")
    print(response.strip())
    print("===========================") 