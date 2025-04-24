import json
import os
import logging
from transformers import AutoConfig # To get model_type if needed

# --- CONFIGURATION (MUST MATCH quantize_models.py) ---
QUANTIZED_MODEL_OUTPUT_DIR = "/workspace/models/Mistral-7B-Instruct-v0.3/quantized-gptq-4bit"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3" # Used to fetch model_type if needed
GPTQ_BITS = 4
GPTQ_GROUP_SIZE = 128
GPTQ_DAMP_PERCENT = 0.01
GPTQ_DESC_ACT = False
TRUST_REMOTE_CODE = True # Set according to the model
# --- END CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

quantize_config_path = os.path.join(QUANTIZED_MODEL_OUTPUT_DIR, "quantize_config.json")
model_safetensors_path = os.path.join(QUANTIZED_MODEL_OUTPUT_DIR, "model.safetensors")

def create_config():
    logging.info(f"Attempting to create: {quantize_config_path}")

    if not os.path.exists(QUANTIZED_MODEL_OUTPUT_DIR):
        logging.error(f"Output directory does not exist: {QUANTIZED_MODEL_OUTPUT_DIR}")
        return

    if not os.path.exists(model_safetensors_path):
        logging.error(f"CRITICAL: Quantized weights file not found: {model_safetensors_path}. Cannot proceed.")
        return

    if os.path.exists(quantize_config_path):
        logging.warning(f"File already exists: {quantize_config_path}. Overwriting.")
        # Decide if you want to overwrite or exit. Let's overwrite for this fix.
        # return

    try:
        # Try to get model_type from the config saved in the *output* directory first
        model_type = None
        try:
            output_config = AutoConfig.from_pretrained(QUANTIZED_MODEL_OUTPUT_DIR, trust_remote_code=TRUST_REMOTE_CODE)
            model_type = getattr(output_config, "model_type", None)
            logging.info(f"Found model_type '{model_type}' in {QUANTIZED_MODEL_OUTPUT_DIR}/config.json")
        except Exception as e:
            logging.warning(f"Could not load config from output directory ({e}). Trying original model ID.")
            # Fallback to loading from original ID
            original_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)
            model_type = getattr(original_config, "model_type", None)
            logging.info(f"Found model_type '{model_type}' from original model ID {MODEL_ID}")

        if not model_type:
            raise ValueError("Could not determine model_type for the config.")

        # --- Construct the dictionary manually ---
        manual_config = {
            "bits": GPTQ_BITS,
            "group_size": GPTQ_GROUP_SIZE,
            "damp_percent": GPTQ_DAMP_PERCENT,
            "desc_act": GPTQ_DESC_ACT,
            "sym": True, # Common default for GPTQ symmetric quantization
            "model_type": model_type,
            "quant_method": "gptq" # Standard identifier
            # Note: We don't include tokenizer or dataset info here,
            # as it's not strictly required by transformers to *load* the config.
        }
        logging.info(f"Creating config content: {json.dumps(manual_config, indent=2)}")

        with open(quantize_config_path, 'w') as f:
            json.dump(manual_config, f, indent=2)

        logging.info(f"Successfully created: {quantize_config_path}")

    except Exception as e:
        logging.error(f"Error creating quantize_config.json: {e}", exc_info=True)

if __name__ == "__main__":
    create_config()
