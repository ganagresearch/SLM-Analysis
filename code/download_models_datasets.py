import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import gc

# --- Configuration ---
model_ids = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/electra-small-discriminator" # Not CausalLM, handle separately
]
cache_dir = "/workspace/hf_cache" # e.g., /runpod-data/hf_cache

os.makedirs(cache_dir, exist_ok=True)
print(f"Using cache directory: {cache_dir}")

# --- Download Models and Tokenizers ---
for model_id in model_ids:
    print(f"\nProcessing {model_id}...")
    try:
        # Special handling for electra (not a CausalLM)
        if "electra" in model_id:
            model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
        # Add trust_remote_code=True for models that require it (like Phi-3)
        elif "Phi-3" in model_id:
             model = AutoModelForCausalLM.from_pretrained(
                 model_id,
                 cache_dir=cache_dir,
                 trust_remote_code=True
             )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir
            )

        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        print(f"Successfully downloaded model and tokenizer for {model_id}")

        # Clean up memory
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Failed to download {model_id}: {e}")

print("\nModel downloads complete.")
