# Suggested path: /workspace/code/quantize_models.py
import os
import sys
import logging
from datetime import datetime
import datasets
from optimum.onnxruntime import ORTQuantizer, AutoQuantizationConfig, AutoCalibrationConfig
from optimum.exporters.onnx import main_export
import shutil
from transformers import AutoTokenizer
import torch
import psutil
import optimum


# --- Configuration ---
WORKSPACE_ROOT = "/workspace/"
CALIBRATION_DATA_PATH = os.path.join(WORKSPACE_ROOT, "calibration_data/sevenllm_instruct_subset_manual/")
NUM_CALIBRATION_SAMPLES = 200
CALIBRATION_COLUMN = "instruction"
MAX_SEQ_LENGTH = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODELS_TO_QUANTIZE = [
    # # --- Run 1: Mistral-7B --- (Comment out others to test individually)
    # {
    #     "name": "Mistral-7B-Instruct-v0.3",
    #     "input_path": os.path.join(WORKSPACE_ROOT, "models/Mistral-7B-Instruct-v0.3/"),
    #     "trust_remote_code": True,
    #     "task": "text-generation",
    # },
    # --- Run 2: TinyLlama --- (Uncomment to test)
    #{
    #    "name": "TinyLlama-1.1B-Chat-v1.0",
    #    "input_path": os.path.join(WORKSPACE_ROOT, "models/TinyLlama-1.1B-Chat-v1.0/"),
    #    "trust_remote_code": False,
    #    "task": "text-generation",
    #},
    # --- Run 3: Phi-3-mini --- (Comment out others to test individually)
     {
         "name": "Phi-3-mini-4k-instruct",
         "input_path": os.path.join(WORKSPACE_ROOT, "models/Phi-3-mini-4k-instruct/"),
         "trust_remote_code": True,
         "task": "text-generation",
     },
]

# --- Setup Logging ---
LOG_DIR = os.path.join(WORKSPACE_ROOT, "results", "quantization_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"quantization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- System Info Logging ---
logging.info("--- System Information ---")
logging.info(f"Python Version: {sys.version}")
logging.info(f"Torch Version: {torch.__version__}")
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
else:
    logging.info("GPU: Not Available / Not Used")
logging.info(f"Using Device: {DEVICE}")
logging.info("--------------------------")


# --- Load Calibration Data ---
logging.info(f"Loading calibration dataset from: {CALIBRATION_DATA_PATH}")
try:
    # Load the full dataset first to check length
    calibration_dataset_full = datasets.load_from_disk(CALIBRATION_DATA_PATH)
    actual_num_samples = min(NUM_CALIBRATION_SAMPLES, len(calibration_dataset_full))
    if actual_num_samples < NUM_CALIBRATION_SAMPLES:
        logging.warning(f"Requested {NUM_CALIBRATION_SAMPLES} calibration samples, but dataset only has {len(calibration_dataset_full)}. Using {actual_num_samples} samples.")

    # Select the subset for calibration
    calibration_dataset = calibration_dataset_full.select(range(actual_num_samples))
    logging.info(f"Loaded and selected {len(calibration_dataset)} calibration samples.")

    if CALIBRATION_COLUMN not in calibration_dataset.column_names:
         raise ValueError(f"Calibration column '{CALIBRATION_COLUMN}' not found in dataset columns: {calibration_dataset.column_names}")
    logging.info(f"Using column '{CALIBRATION_COLUMN}' for calibration text.")

except Exception as e:
    logging.error(f"FATAL: Error loading calibration dataset from {CALIBRATION_DATA_PATH}: {e}", exc_info=True)
    sys.exit(1)


# --- Define Preprocessing Function Early ---
# Define this before the quantization loop if tokenizer needs to be loaded first,
# OR define inside the loop after tokenizer is loaded if it's specific to the model.
# Defining inside the loop is safer if tokenizers differ significantly.
# Let's define it *inside* the loop for now.

# --- Quantization Loop ---
logging.info("--- Starting Model Quantization ---")

for model_config in MODELS_TO_QUANTIZE:
    model_name = model_config["name"]
    input_path = model_config["input_path"]
    trust_remote_code = model_config["trust_remote_code"]
    model_task = model_config["task"]
    # Final INT8 model path - Saving directly into the original model dir now
    quantized_output_path = input_path
    # Path for the intermediate FP32 ONNX export (temporary location)
    onnx_export_path_temp = os.path.join(quantized_output_path, "temp-fp32-onnx-export") # Use a temp subdir

    logging.info(f"\nProcessing model: {model_name}")
    logging.info(f"Input path (HF): {input_path}")
    logging.info(f"ONNX Export Path (Intermediate): {onnx_export_path_temp}")
    logging.info(f"Output path (Quantized ONNX files): {quantized_output_path}")
    logging.info(f"Trust remote code: {trust_remote_code}")
    logging.info(f"Model Task: {model_task}")

    if not os.path.isdir(input_path):
        logging.error(f"Skipping {model_name}: Input HF directory not found at {input_path}")
        continue

    # Check if final *quantized* file already exists
    potential_quantized_onnx_file = os.path.join(quantized_output_path, "model_quantized.onnx")
    if os.path.exists(potential_quantized_onnx_file):
        logging.warning(f"Skipping {model_name}: Final quantized ONNX file already exists at {potential_quantized_onnx_file}")
        continue

    # Create the temporary export directory if it doesn't exist (it shouldn't if cleanup works)
    os.makedirs(onnx_export_path_temp, exist_ok=True)

    onnx_model_filename = "model.onnx" # Default name
    fp32_export_successful = False
    fit_successful = False
    quantization_export_successful = False
    calibration_ranges = None

    try:
        # --- 1. Export Hugging Face model to ONNX (FP32) ---
        onnx_fp32_model_path = os.path.join(onnx_export_path_temp, onnx_model_filename) # Default path
        # Check if intermediate FP32 export already exists from a previous failed run
        if not os.path.exists(onnx_fp32_model_path):
            logging.info(f"Exporting {model_name} from Hugging Face format to ONNX...")
            export_start_time = datetime.now()

            main_export(
                model_name_or_path=input_path,
                output=onnx_export_path_temp, # Export to temp dir
                task=model_task,
                trust_remote_code=trust_remote_code,
                opset=14 # Explicitly set opset version for compatibility
            )

            # Verify and get the actual ONNX filename created by the exporter
            potential_files = [f for f in os.listdir(onnx_export_path_temp) if f.endswith('.onnx')]
            if not potential_files:
                raise FileNotFoundError(f"No ONNX model file found after export in: {onnx_export_path_temp}")
            onnx_model_filename = potential_files[0] # Use the actual name
            onnx_fp32_model_path = os.path.join(onnx_export_path_temp, onnx_model_filename) # Update full path
            logging.info(f"Identified exported ONNX model file: {onnx_model_filename}")

            export_end_time = datetime.now()
            logging.info(f"ONNX export successful. Duration: {export_end_time - export_start_time}")
        else:
            logging.warning(f"Found existing intermediate FP32 ONNX model at {onnx_fp32_model_path}. Skipping export.")

        fp32_export_successful = True
        logging.info(f"Using FP32 ONNX model: {onnx_fp32_model_path}")

        # --- 2. Create Quantizer & Tokenizer ---
        logging.info("Loading quantizer for the exported ONNX model...")
        quantizer = ORTQuantizer.from_pretrained(onnx_export_path_temp, file_name=onnx_model_filename)
        logging.info("Loading tokenizer from original HF path...")
        tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=trust_remote_code)
        logging.info("Quantizer and tokenizer loaded.")

        # --- Define Preprocessing Function (using the loaded tokenizer) ---
        def preprocess_calibration(examples):
            if tokenizer.padding_side != 'right':
                 tokenizer.padding_side = 'right'
            # Tokenize the input text
            model_inputs = tokenizer(
                examples[CALIBRATION_COLUMN],
                padding='max_length',
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )

            # --- Manually create position_ids ---
            # model_inputs['input_ids'] is a list of lists (batch of sequences)
            input_ids_batch = model_inputs['input_ids']
            position_ids_batch = []
            for input_ids_sequence in input_ids_batch:
                # Create a range from 0 to length-1 for each sequence
                position_ids_batch.append(list(range(len(input_ids_sequence))))

            # Add the position_ids to the dictionary
            model_inputs['position_ids'] = position_ids_batch
            # ------------------------------------

            return model_inputs

        # --- Preprocess the Calibration Dataset ---
        logging.info("Preprocessing (tokenizing) the calibration dataset...")
        # Apply the tokenizer - use batched=True for speed.
        # Remove original text column to only keep model inputs.
        processed_calibration_dataset = calibration_dataset.map(
            preprocess_calibration,
            batched=True,
            remove_columns=calibration_dataset.column_names # Remove all original columns
        )
        logging.info(f"Preprocessing complete. New columns: {processed_calibration_dataset.column_names}")
        # Update check to include position_ids
        expected_cols = ['input_ids', 'attention_mask', 'position_ids']
        if not all(col in processed_calibration_dataset.column_names for col in expected_cols):
             logging.warning(f"Expected {expected_cols} in preprocessed dataset, but found {processed_calibration_dataset.column_names}. Model might require different inputs.")


        # --- 3. Define Quantization & Calibration Config ---
        logging.info("Defining quantization and calibration configurations...")
        operators_to_quantize = ['MatMul', 'Add']
        qconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=True,
            per_channel=False,
            operators_to_quantize=operators_to_quantize
        )
        logging.info(f"Quantization Config: {qconfig}")

        # --- Create Calibration Config using the *processed* dataset ---
        logging.info(f"Creating Calibration Config using preprocessed dataset.")
        calibration_config = AutoCalibrationConfig.minmax(
            processed_calibration_dataset # Pass the processed dataset here
        )
        logging.info(f"Calibration Method: {calibration_config.method}")
        logging.info(f"Operators to quantize: {qconfig.operators_to_quantize}")


        # --- 5. Perform Calibration using fit() with the *processed* dataset ---
        logging.info("Starting calibration step (quantizer.fit)...")
        fit_start_time = datetime.now()
        augmented_model_path = os.path.join(onnx_export_path_temp, "augmented_model.onnx")

        calibration_tensors_range = quantizer.fit(
            dataset=processed_calibration_dataset, # Pass processed dataset here
            calibration_config=calibration_config,
            operators_to_quantize=qconfig.operators_to_quantize,
            batch_size=4, # Keep batch_size
            onnx_augmented_model_name=augmented_model_path,
            use_external_data_format=True,
	    use_gpu=torch.cuda.is_available() # Explicitly request GPU for calibration
        )
        fit_end_time = datetime.now()
        fit_successful = True
        logging.info(f"Calibration step (fit) successful. Duration: {fit_end_time - fit_start_time}")

        if not calibration_tensors_range:
             raise ValueError("Calibration failed: quantizer.fit() did not return calibration ranges.")
        logging.info("Computed calibration ranges (TensorsData object returned).")

        # --- 6. Export Quantized Model using quantize() ---
        logging.info("Applying quantization and exporting INT8 model (quantizer.quantize)...")
        export_quant_start_time = datetime.now()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
            torch.cuda.empty_cache()

        # Define the final output path for the quantized ONNX model file
        final_quantized_model_path = os.path.join(quantized_output_path, "model_quantized.onnx") # Use explicit quantized name

        # *** Using the quantizer.quantize method ***
        quantizer.quantize(
            save_dir=quantized_output_path, # Save final quantized model here
            quantization_config=qconfig,
            calibration_tensors_range=calibration_tensors_range, # Pass computed ranges
            file_suffix="quantized", # Suffix for the quantized model filename (e.g., model_quantized.onnx)
            use_external_data_format=True # Crucial for models > 2GB
        )

        export_quant_end_time = datetime.now()
        quantization_export_successful = True # Mark success only after export finishes

        peak_vram_mb = -1
        if torch.cuda.is_available():
            peak_vram_bytes = torch.cuda.max_memory_allocated(DEVICE)
            peak_vram_mb = peak_vram_bytes / (1024**2)
            logging.info(f"Peak PyTorch VRAM allocated during quantization export: {peak_vram_mb:.2f} MB")

        duration = export_quant_end_time - export_quant_start_time
        logging.info(f"Quantized model export successful for {model_name}!")
        logging.info(f"Quantized model saved to: {final_quantized_model_path}") # Point to specific file
        logging.info(f"Quantization export duration: {duration}")

    except Exception as e:
        logging.error(f"Error during processing for {model_name}: {e}", exc_info=True)

    finally:
        # --- Cleanup ---
        if quantization_export_successful and fp32_export_successful and fit_successful:
            # Remove intermediate FP32 export only on full success
            try:
                logging.info(f"Removing intermediate ONNX export directory: {onnx_export_path_temp}")
                shutil.rmtree(onnx_export_path_temp)
            except Exception as cleanup_e:
                logging.warning(f"Could not remove intermediate ONNX export directory {onnx_export_path_temp}: {cleanup_e}")
        elif fp32_export_successful:
             # If only FP32 export succeeded, keep it but log a warning
             logging.warning(f"Quantization failed for {model_name}. Keeping intermediate FP32 ONNX export at: {onnx_export_path_temp}")
        else:
             # If even FP32 export failed, remove the entire base output directory created
             # (which only contains the temp export dir anyway)
             if os.path.exists(onnx_export_path_temp): # Check temp dir
                 try:
                     logging.info(f"Attempting to clean up failed export directory: {onnx_export_path_temp}")
                     # Also remove the parent if it's empty? Or just the temp. Let's just remove temp.
                     shutil.rmtree(onnx_export_path_temp)
                     # Attempt to remove parent only if empty - this might fail if other files exist, which is fine.
                     try: os.rmdir(quantized_output_path)
                     except OSError: pass
                     logging.info(f"Cleaned up failed export directory: {onnx_export_path_temp}")
                 except Exception as cleanup_e:
                     logging.error(f"Failed to clean up directory {onnx_export_path_temp} after error: {cleanup_e}", exc_info=True)


logging.info("--- Model Quantization Finished ---")
