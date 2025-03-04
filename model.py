import transformers
import torch

model_id = "meta-llama/Llama-3.2-3B"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

transformers.set_seed(0)

prompt = """How many helicopters can a human eat in one sitting? Reply as a thug."""
model_inputs = tokenizer([prompt], return_tensors="pt")
input_length = model_inputs.input_ids.shape[1]
generated_ids = model.generate(**model_inputs)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
