# Suggested path: /workspace/code/quantize_models.py
import os
import sys
import logging
from datetime import datetime
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig, AutoConfig
import torch
import psutil
import importlib.metadata
import numpy as np
import shutil
import json # For manual config saving

# --- Configuration ---
WORKSPACE_ROOT = "/workspace/"
CALIBRATION_DATA_PATH = os.path.join(WORKSPACE_ROOT, "calibration_data/sevenllm_instruct_subset_manual/")
NUM_CALIBRATION_SAMPLES = 128 # GPTQ often needs fewer samples than static ONNX, 128 is common
CALIBRATION_COLUMN = "instruction" # Field from SEVENLLM dataset to use
MAX_SEQ_LENGTH = 512 # Max sequence length for tokenization during calibration prep (if needed)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" # AutoGPTQ requires CUDA

# --- GPTQ Configuration (Apply to all models in this run) ---
GPTQ_BITS = 4 # Number of bits for quantization (4 is common)
GPTQ_GROUP_SIZE = 128 # Group size for quantization params. -1 means per-column. 128 is common.
GPTQ_DAMP_PERCENT = 0.01 # Dampening factor for Hessian calculation stability
GPTQ_DESC_ACT = False # Activation order determination. False is often better for Llama-based models.
# USE_EXLLAMA = False # Set True if exllama kernels are installed for potentially faster quantization

# --- Models to Quantize ---
MODELS_TO_QUANTIZE = [
    # {
    #     "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
    #     "trust_remote_code": True,
    # },
    {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "trust_remote_code": True, # Phi-3 requires trust_remote_code
    },
    {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "trust_remote_code": False,
    },
]

# --- Setup Logging ---
LOG_DIR = os.path.join(WORKSPACE_ROOT, "results", "quantization_logs_gptq")
os.makedirs(LOG_DIR, exist_ok=True)
# Use a generic log file name for the multi-model run
LOG_FILE = os.path.join(LOG_DIR, f"quantization_log_gptq_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- System Info Logging ---
logging.info("--- System Information ---")
logging.info(f"Script: AutoGPTQ Quantization (Multi-Model)")
logging.info(f"Python Version: {sys.version}")
logging.info(f"Torch Version: {torch.__version__}")
try:
    transformers_version = importlib.metadata.version('transformers')
    logging.info(f"Transformers Version: {transformers_version}")
except importlib.metadata.PackageNotFoundError:
    logging.warning("Could not determine Transformers version.")
try: # Optimum is needed for GPTQConfig sometimes, or for auto-gptq integration
    optimum_version = importlib.metadata.version('optimum')
    logging.info(f"Optimum Version: {optimum_version}")
except importlib.metadata.PackageNotFoundError:
    logging.info("Optimum not found (may not be strictly needed depending on backend).")
try:
    accelerate_version = importlib.metadata.version('accelerate')
    logging.info(f"Accelerate Version: {accelerate_version}")
except importlib.metadata.PackageNotFoundError:
    logging.warning("Accelerate not found (recommended for AutoGPTQ).")
try:
    bitsandbytes_version = importlib.metadata.version('bitsandbytes')
    logging.info(f"Bitsandbytes Version: {bitsandbytes_version}")
except importlib.metadata.PackageNotFoundError:
    logging.info("Bitsandbytes not found.")
logging.info(f"Datasets Version: {datasets.__version__}")
logging.info(f"CPU Count: {os.cpu_count()}")
ram_gb = psutil.virtual_memory().total / (1024**3)
logging.info(f"Total RAM: {ram_gb:.2f} GB")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(DEVICE)
    total_vram_gb = torch.cuda.get_device_properties(DEVICE).total_memory / (1024**3)
    logging.info(f"GPU: {gpu_name}")
    logging.info(f"Total VRAM: {total_vram_gb:.2f} GB")
    logging.info(f"CUDA Version: {torch.version.cuda}")
    logging.info(f"Using Device: {DEVICE}")
else:
    logging.error("FATAL: AutoGPTQ requires CUDA, but it is not available or not detected.")
    sys.exit(1)
logging.info(f"GPTQ Config: bits={GPTQ_BITS}, group_size={GPTQ_GROUP_SIZE}, damp={GPTQ_DAMP_PERCENT}, desc_act={GPTQ_DESC_ACT}")
logging.info("--------------------------")


# --- Main Quantization Function (now accepts model config) ---
def run_quantization(model_id, trust_remote_code):
    """Quantizes a single model using AutoGPTQ."""

    model_short_name = model_id.split('/')[-1]
    base_model_path = os.path.join(WORKSPACE_ROOT, "models", model_short_name)
    quantized_model_output_dir = os.path.join(base_model_path, f"quantized-gptq-{GPTQ_BITS}bit")

    logging.info(f"\n===== Starting AutoGPTQ for model: {model_id} =====")
    logging.info(f"Quantizing to {GPTQ_BITS}-bit precision.")
    logging.info(f"Trust remote code: {trust_remote_code}")
    logging.info(f"Output directory: {quantized_model_output_dir}")

    # --- Check if already quantized ---
    quantize_config_file = os.path.join(quantized_model_output_dir, "quantize_config.json")
    model_file = os.path.join(quantized_model_output_dir, "model.safetensors") # Check for safetensors

    if os.path.exists(quantized_model_output_dir) and os.path.exists(quantize_config_file) and os.path.exists(model_file):
         logging.warning(f"SKIPPING {model_id}: Output directory {quantized_model_output_dir} already exists and seems complete.")
         return True # Indicate skipped but successful state

    os.makedirs(quantized_model_output_dir, exist_ok=True)

    quantized_model = None # Ensure variable exists for finally block

    try:
        # --- 1. Load Tokenizer ---
        logging.info(f"Loading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info("Set tokenizer pad_token to eos_token.")
        logging.info("Tokenizer loaded.")

        # --- 2. Load and Prepare Calibration Data ---
        # (Loading logic moved outside the function for efficiency, just use the loaded data)
        logging.info(f"Preparing calibration data list (using pre-loaded data)...")
        if 'calibration_data_list' not in globals():
             raise NameError("Calibration data list not found in global scope.")
        logging.info(f"Using {len(calibration_data_list)} calibration samples.")


        # --- 3. Define GPTQ Configuration ---
        logging.info("Defining GPTQ configuration...")
        gptq_config = GPTQConfig(
            bits=GPTQ_BITS,
            dataset=calibration_data_list, # Pass the prepared list of text samples
            tokenizer=tokenizer,
            group_size=GPTQ_GROUP_SIZE,
            damp_percent=GPTQ_DAMP_PERCENT,
            desc_act=GPTQ_DESC_ACT,
        )
        logging.info(f"GPTQ Config: bits={gptq_config.bits}, group_size={gptq_config.group_size}, damp_percent={gptq_config.damp_percent}, desc_act={gptq_config.desc_act}")

        # --- 4. Load Model and Quantize ---
        logging.info(f"Loading model {model_id} and starting quantization...")
        quantization_start_time = datetime.now()

        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=gptq_config,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        quantization_end_time = datetime.now()
        logging.info(f"Model loading and quantization finished. Duration: {quantization_end_time - quantization_start_time}")

        # --- 5. Save Quantized Model and Tokenizer ---
        logging.info(f"Saving quantized model and tokenizer to {quantized_model_output_dir}...")
        save_start_time = datetime.now()
        quantized_model.save_pretrained(quantized_model_output_dir)
        tokenizer.save_pretrained(quantized_model_output_dir)
        save_end_time = datetime.now()
        logging.info(f"Save calls completed. Duration: {save_end_time - save_start_time}")

        # --- Verification and Manual Fix for quantize_config.json ---
        quantize_config_path = os.path.join(quantized_model_output_dir, "quantize_config.json")
        model_safetensors_path = os.path.join(quantized_model_output_dir, "model.safetensors")

        save_successful = False
        if not os.path.exists(model_safetensors_path):
             logging.error("CRITICAL: model.safetensors not found after saving!")
        elif not os.path.exists(quantize_config_path):
             logging.warning(f"quantize_config.json is missing. Attempting manual creation...")
             try:
                 final_gptq_config = getattr(quantized_model.config, "quantization_config", None)
                 if final_gptq_config and isinstance(final_gptq_config, GPTQConfig):
                      logging.info("Found quantization config in model.config. Saving it.")
                      # Try saving without use_diff first
                      try:
                          final_gptq_config.to_json_file(quantize_config_path)
                      except TypeError:
                          logging.warning("to_json_file failed with TypeError (likely 'use_diff'). Falling back to manual construction.")
                          # Fallback to manual construction if direct save fails
                          final_gptq_config = None # Force fallback
                      else:
                           logging.info(f"Manually saved quantize_config.json from model's config.")
                 else:
                      # Force fallback if not found or if direct save failed
                      final_gptq_config = None

                 if final_gptq_config is None: # Proceed with fallback if needed
                     logging.warning("Creating quantize_config.json from script parameters.")
                     # Get model_type from the loaded & quantized model's config
                     model_type = getattr(quantized_model.config, "model_type", None)
                     if not model_type:
                         # Try loading original config if needed
                          orig_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                          model_type = getattr(orig_config, "model_type", None)

                     if not model_type:
                         raise ValueError(f"Could not determine model_type for {model_id}")

                     manual_config = {
                         "bits": GPTQ_BITS,
                         "group_size": GPTQ_GROUP_SIZE,
                         "damp_percent": GPTQ_DAMP_PERCENT,
                         "desc_act": GPTQ_DESC_ACT,
                         "sym": True, # Assuming symmetric quantization
                         "model_type": model_type,
                         "quant_method": "gptq"
                     }
                     with open(quantize_config_path, 'w') as f:
                         json.dump(manual_config, f, indent=2)
                     logging.info(f"Manually created quantize_config.json using script parameters.")

                 # Re-verify after attempting manual creation
                 if os.path.exists(quantize_config_path):
                     logging.info("Successfully created/verified quantize_config.json.")
                     save_successful = True
                 else:
                     logging.error("Failed to manually create quantize_config.json.")
             except Exception as config_err:
                 logging.error(f"Error manually creating quantize_config.json: {config_err}", exc_info=True)
        else:
             # Both files exist
             logging.info("Verified essential files (quantize_config.json, model.safetensors) exist.")
             save_successful = True

        return save_successful # Return True only if essential files are present

    except Exception as e:
        logging.error(f"FATAL Error during quantization for {model_id}: {e}", exc_info=True)
        logging.warning(f"Removing possibly incomplete output directory: {quantized_model_output_dir}")
        shutil.rmtree(quantized_model_output_dir, ignore_errors=True)
        return False # Indicate failure

    finally:
        # --- Cleanup ---
        logging.info(f"Cleaning up model object for {model_id} from memory...")
        del quantized_model # Delete the model object
        if 'tokenizer' in locals(): del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Cleared CUDA cache.")
        logging.info(f"===== Finished processing model: {model_id} =====")


# --- Script Execution ---
if __name__ == "__main__":
    # --- Load Calibration Data Once ---
    logging.info(f"Loading calibration dataset from: {CALIBRATION_DATA_PATH}")
    calibration_data_list = [] # Make it global for the function to access
    try:
        if not os.path.isdir(CALIBRATION_DATA_PATH):
             raise FileNotFoundError(f"Calibration data path is not a directory: {CALIBRATION_DATA_PATH}.")
        calibration_dataset_full = datasets.load_from_disk(CALIBRATION_DATA_PATH)
        actual_num_samples = min(NUM_CALIBRATION_SAMPLES, len(calibration_dataset_full))
        if actual_num_samples < NUM_CALIBRATION_SAMPLES:
            logging.warning(f"Requested {NUM_CALIBRATION_SAMPLES} calibration samples, but dataset only has {len(calibration_dataset_full)}. Using {actual_num_samples} samples.")
        elif actual_num_samples == 0:
            raise ValueError("Calibration dataset is empty.")

        calibration_dataset_shuffled = calibration_dataset_full.shuffle(seed=42)
        calibration_dataset_selected = calibration_dataset_shuffled.select(range(actual_num_samples))
        logging.info(f"Loaded and selected {len(calibration_dataset_selected)} calibration samples.")

        if CALIBRATION_COLUMN not in calibration_dataset_selected.column_names:
             raise ValueError(f"Calibration column '{CALIBRATION_COLUMN}' not found.")

        calibration_data_list = calibration_dataset_selected[CALIBRATION_COLUMN]
        logging.info(f"Prepared global calibration data list with {len(calibration_data_list)} strings.")

    except Exception as e:
        logging.error(f"FATAL: Error loading or processing calibration dataset: {e}", exc_info=True)
        sys.exit(1)

    # --- Loop Through Models ---
    successful_quantizations = []
    failed_quantizations = []
    skipped_quantizations = []

    for config in MODELS_TO_QUANTIZE:
        model_id = config["model_id"]
        trust_code = config["trust_remote_code"]

        # Check if already done before calling the function
        model_short_name = model_id.split('/')[-1]
        base_model_path = os.path.join(WORKSPACE_ROOT, "models", model_short_name)
        quantized_model_output_dir = os.path.join(base_model_path, f"quantized-gptq-{GPTQ_BITS}bit")
        quantize_config_file = os.path.join(quantized_model_output_dir, "quantize_config.json")
        model_file = os.path.join(quantized_model_output_dir, "model.safetensors")

        if os.path.exists(quantized_model_output_dir) and os.path.exists(quantize_config_file) and os.path.exists(model_file):
            logging.warning(f"SKIPPING {model_id}: Output directory {quantized_model_output_dir} already exists and seems complete.")
            skipped_quantizations.append(model_id)
            continue

        # Run quantization for the current model
        success = run_quantization(model_id=model_id, trust_remote_code=trust_code)

        if success:
            successful_quantizations.append(model_id)
        else:
            failed_quantizations.append(model_id)

    # --- Final Summary ---
    logging.info("\n--- AutoGPTQ Quantization Script Finished ---")
    logging.info("Summary:")
    logging.info(f"  Successfully Quantized: {len(successful_quantizations)}")
    for m in successful_quantizations: logging.info(f"    - {m}")
    logging.info(f"  Skipped (Already Done): {len(skipped_quantizations)}")
    for m in skipped_quantizations: logging.info(f"    - {m}")
    logging.info(f"  Failed: {len(failed_quantizations)}")
    for m in failed_quantizations: logging.info(f"    - {m}")

    if failed_quantizations:
        logging.warning("Some models failed quantization. Please check the logs above for details.")

