import torch
import transformers
import time
import os
import psutil  # If not installed, run: pip install psutil
import sys

# Print system info
process = psutil.Process(os.getpid())
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Available RAM: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.2f} GB")
print(f"CPU count: {psutil.cpu_count()}")
if torch.cuda.is_available():
    print(f"CUDA available: Yes, version {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA available: No")

# Keep using the smaller model since it works well
model_id = "Qwen/Qwen2-0.5B"

print(f"Loading model {model_id}...")
start_load = time.time()

# Load tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code=True
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print(f"Model loaded in {time.time() - start_load:.2f} seconds")
print(f"Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

# Set to evaluation mode
model.eval()
transformers.set_seed(0)

# Choose a more complex prompt - uncomment one or create your own
complex_prompt = """What is rhythmic stabilisation and why is it important for lower back pain?"""

# Alternative prompts:
# complex_prompt = "Explain the difference between supervised, unsupervised, and reinforcement learning with examples."
# complex_prompt = "Write a short story about a robot that becomes self-aware."

print(f"\nGenerating response for complex prompt...")
print(f"Prompt: '{complex_prompt}'")

# Format with chat template for better results
messages = [
    # {"role": "system", "content": "You are a helpful, concise physiotherapist assistant. Provide accurate responses with examples where appropriate."},
    {"role": "user", "content": complex_prompt}
]

# Apply chat template if the model supports it
try:
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
except:
    # Fallback to direct input if chat template isn't supported
    input_text = complex_prompt

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")
input_token_count = inputs.input_ids.shape[1]
print(f"Input length: {input_token_count} tokens")

# Generate with longer output for complex prompt
print("Starting generation...")
start_time = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # Generate more tokens for complex response
        do_sample=False,     # Deterministic for faster generation
        pad_token_id=tokenizer.eos_token_id
    )

# Measure generation time
elapsed = time.time() - start_time
output_token_count = outputs.shape[1] - input_token_count
tokens_per_second = output_token_count / elapsed

print(f"Generation took {elapsed:.2f} seconds")
print(f"Generated {output_token_count} new tokens ({tokens_per_second:.2f} tokens/sec)")

# Decode only the new tokens
response = tokenizer.decode(
    outputs[0][input_token_count:], 
    skip_special_tokens=True
)

print("\n===== MODEL RESPONSE =====")
print(response)
print("===========================")

print("\nTest completed successfully!")