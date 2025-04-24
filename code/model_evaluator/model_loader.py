import time
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from evaluation_utils import get_memory_usage, NVML_FOUND, nvml_handle # Import necessary utils

def load_model_and_tokenizer(model_path, model_type='causal', device='cuda', trust_remote_code=False):
    """
    Loads model and tokenizer, tracking resource usage.
    Accepts trust_remote_code flag to pass to Hugging Face methods.
    Returns model, tokenizer, and load_metrics dict (excluding initial values).
    """
    logging.info(f"Loading {model_type} model from: {model_path}...")
    logging.info(f"Trust Remote Code: {trust_remote_code}") # Log the value being used
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logging.info("Reset peak PyTorch VRAM stats.")

    # --- Load Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            legacy=False
        )
        logging.info("Tokenizer loaded.")
    except Exception as e:
         logging.error(f"Error loading tokenizer: {e}", exc_info=True)
         return None, None, {}

    # --- Load Model ---
    try:
        model_args = {
             "trust_remote_code": trust_remote_code,
             # "torch_dtype": torch.bfloat16, # Consider adding
        }
        if model_type == 'causal':
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                **model_args
            )
            logging.info("CausalLM Model loaded.")
        elif model_type == 'encoder':
             model = AutoModel.from_pretrained(
                 model_path,
                 device_map=device,
                 **model_args
             )
             logging.info("Encoder Model loaded.")
        else:
            logging.error(f"Unsupported model_type: {model_type}")
            return None, None, {}

        model.eval()
        logging.info(f"Model placed on device: {device} and set to eval mode.")

    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        if 'model' in locals(): del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None, None, {}

    load_time = time.time() - start_time

    # Get final memory state AFTER loading
    final_ram, pt_vram_after_load, pt_peak_vram_load, sys_vram_after_load = get_memory_usage()

    logging.info(f"Model loaded in {load_time:.2f} seconds.")
    # Logging of initial values and deltas moved to main CLI script

    load_metrics_partial = {
        "load_time_sec": load_time,
        "ram_after_load_mb": final_ram,
        "pytorch_vram_current_after_load_mb": pt_vram_after_load,
        "pytorch_vram_peak_load_mb": pt_peak_vram_load,
        "system_vram_current_after_load_mb": sys_vram_after_load if NVML_FOUND and nvml_handle else None,
    }
    return model, tokenizer, load_metrics_partial 

