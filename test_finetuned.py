import torch
import transformers
from peft import PeftModel, PeftConfig
import time

# Path to your fine-tuned LoRA adapter
lora_path = "./lora-qwen-physio"
model_id = "Qwen/Qwen2-0.5B"

print("Loading base model and LoRA adapter...")
# Load tokenizer from the base model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load the base model
base_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map="cpu"
)

# Load LoRA configuration and model
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# Test prompts
test_prompts = [
    "What is rhythmic stabilization and how does it help with lower back pain?",
    "Can you give me some rhythmic stabilization exercises for lumbar spine?",
    "Why is core stability important for lower back pain management?"
]

# Generate responses
for i, prompt in enumerate(test_prompts):
    print(f"\n\nTest {i+1}: {prompt}")
    
    # Format with chat template
    messages = [{"role": "user", "content": prompt}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        input_text = f"USER: {prompt}\nASSISTANT:"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    
    # Generate
    print("Generating response...")
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
    print("\n===== MODEL RESPONSE =====")
    print(response.strip())
    print("===========================") 