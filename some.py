import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/workspace/models/Phi-3-mini-4k-instruct-GPTQ"
print(f"Attempting to load model from: {model_path}")

try:
    # Try loading without trusting remote code
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    print("Tokenizer loaded successfully (trust_remote_code=False).")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=False, # Try loading without trusting remote code
        torch_dtype=torch.float16,
    )
    print("Model loaded successfully (trust_remote_code=False).")

    # Optional: Try a simple generation
    print("Attempting simple generation...")
    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=5)
    print("Generation completed.")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()

print("\nScript finished.")
