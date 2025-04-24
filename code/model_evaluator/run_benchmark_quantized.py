import subprocess
import os
import logging
import sys
# import argparse # No longer needed
from datetime import datetime

# --- Configuration ---
# Ensure this points to your evaluation CLI script within the model_evaluator directory
PYTHON_EXE = "/workspace/testbedvenv/bin/python" # Path to python executable in the venv
EVAL_SCRIPT_REL_PATH = "code/model_evaluator/evaluate_cli.py" # Relative to workspace root
WORKSPACE_ROOT = "/workspace/"
# *** Update results directory ***
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results/mod/48") # Quantized results
# *** Update log file name pattern if desired (matching run_benchmark.py) ***
ERROR_LOG_FILE = os.path.join(RESULTS_DIR, "automation_quantized_errors.log")
# *** Define number of samples and max tokens (matching run_benchmark.py) ***
NUM_SAMPLES = 100
MAX_NEW_TOKENS = 512 # Used only for causal models

# *** Define Benchmark Paths (Structure similar to run_benchmark.py) ***
# Use the directory containing the saved HF dataset for sevenllm
# Use the specific JSON file for cyberseceval
# Use a general cache/base path for ctibench_*
DEFAULT_HF_DATASETS_PATH = os.path.join(WORKSPACE_ROOT, "datasets") # Default for Hub sets
BENCHMARKS = [
    {
        "name": "cyberseceval3_mitre",
        "path": os.path.join(WORKSPACE_ROOT, "code/PurpleLlama/CybersecurityBenchmarks/datasets/mitre/mitre_benchmark_100_per_category_with_augmentation.json"),
        "subsets": [None], # No subset needed
    },
    {
        "name": "sevenllm_bench",
        # "path": os.path.join(WORKSPACE_ROOT, "datasets/SEVENLLM_instruct_HF"), # Original path for quantized
        "path": os.path.join(WORKSPACE_ROOT, "calibration_data/sevenllm_instruct_subset_manual/"), # Path from run_benchmark.py - CHOOSE THE CORRECT ONE
        "subsets": [None], # No subset needed
    },
    {
        "name": "ctibench",
        "path": DEFAULT_HF_DATASETS_PATH, # Path to cache/dataset dir for HF
        # Adjust subset names if evaluate_cli.py expects them differently for quantized runs
        # These are based on run_benchmark_quantized.py's splitting logic
        "subsets": ["attack", "exploit_target", "threat_actor", "ttps"],
    },
]


# Models to evaluate (Quantized versions - excluding Electra)
# *** Update model paths and names ***
MODELS_TO_EVALUATE = [
    {
        "name": "TinyLlama-1.1B-Chat-v1.0-GPTQ",
        "path": os.path.join(WORKSPACE_ROOT, 'models/TinyLlama-1.1B-Chat-v1.0/quantized-gptq-4bit/'),
        "type": "causal",
        "trust_remote_code_loader": False # Keep this if needed by loader
    },
    {
        "name": "Phi-3-mini-4k-instruct-GPTQ",
        "path": os.path.join(WORKSPACE_ROOT, 'models/Phi-3-mini-4k-instruct/quantized-gptq-4bit/'),
        "type": "causal",
        "trust_remote_code_loader": True # Keep this if needed by loader
    },
    {
        "name": "Mistral-7B-Instruct-v0.3-GPTQ",
        "path": os.path.join(WORKSPACE_ROOT, 'models/Mistral-7B-Instruct-v0.3/quantized-gptq-4bit/'),
        "type": "causal",
        "trust_remote_code_loader": True # Keep this if needed by loader
    },
    # Electra is not suitable for these generation benchmarks
]

# Benchmarks to run (derived from BENCHMARKS structure)
# BENCHMARKS_TO_RUN = list(BENCHMARK_PATHS.keys()) # Old way

# --- Setup Logging (Similar to run_benchmark.py) ---
os.makedirs(RESULTS_DIR, exist_ok=True)
# Define a timestamped run log file (optional, but keeps track of individual runs)
RUN_LOG_FILE = os.path.join(RESULTS_DIR, f"benchmark_run_quantized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ERROR_LOG_FILE, mode='a'), # Append errors to this file
        logging.FileHandler(RUN_LOG_FILE, mode='a'),   # Log everything to the run file
        logging.StreamHandler(sys.stdout) # Also print info/errors messages to console
    ]
)
# No separate error logger needed if handled by basicConfig

# --- Argument Parser --- # Removed

# --- Filter Models/Benchmarks based on args --- # Removed

# --- Main Loop (Mirrors run_benchmark.py) ---
logging.info("--- Starting Security Benchmark Automation (QUANTIZED MODELS) ---")
logging.info(f"Using Python: {PYTHON_EXE}")
eval_script_abs_path = os.path.join(WORKSPACE_ROOT, EVAL_SCRIPT_REL_PATH) # Calculate absolute path
logging.info(f"Evaluation Script: {eval_script_abs_path}")
logging.info(f"Results Directory: {RESULTS_DIR}")
logging.info(f"Error Log File: {ERROR_LOG_FILE}")
logging.info(f"Run Log File: {RUN_LOG_FILE}")
logging.info(f"Models to run: {[m['name'] for m in MODELS_TO_EVALUATE]}")
logging.info(f"Benchmarks to run: {[(b['name'], b['subsets']) for b in BENCHMARKS]}")
logging.info(f"Running {NUM_SAMPLES} samples per benchmark.")
logging.info(f"Max New Tokens for Causal Models: {MAX_NEW_TOKENS}")


total_runs = sum(len(b["subsets"]) for b in BENCHMARKS) * len(MODELS_TO_EVALUATE)
current_run = 0

for model_config in MODELS_TO_EVALUATE:
    model_name = model_config["name"]
    model_path = model_config["path"]
    model_type = model_config["type"]
    # trust_remote_loader = model_config["trust_remote_code_loader"] # This isn't passed directly

    for bench_config in BENCHMARKS:
        bench_name = bench_config["name"]
        bench_path = bench_config["path"]

        for subset in bench_config["subsets"]:
            current_run += 1
            run_desc = f"Run {current_run}/{total_runs}: Model='{model_name}', Benchmark='{bench_name}'"
            # Adjust benchmark name if subset exists (for CTI mainly)
            actual_benchmark_name = bench_name
            if subset and bench_name == "ctibench":
                 run_desc += f", Subset='{subset}'"
                 # If evaluate_cli expects "ctibench_attack", construct it here
                 # actual_benchmark_name = f"{bench_name}_{subset}" # Or pass subset separately
            elif subset: # Handle potential future benchmarks with subsets
                 run_desc += f", Subset='{subset}'"


            logging.info(f"\n{run_desc}")

            # Check if specific benchmark file/dir exists if not using default path
            # (This check was in the original quantized script, retaining for safety)
            if not os.path.exists(bench_path) and bench_path != DEFAULT_HF_DATASETS_PATH:
                logging.error(f"Benchmark path does not exist: {bench_path}. Skipping this run.")
                # Log to error file manually just in case logging fails
                with open(ERROR_LOG_FILE, 'a') as f_err:
                    f_err.write(f"[{datetime.now()}] ERROR during: {run_desc} - Path missing: {bench_path}\n")
                continue


            # *** Construct the command for evaluate_cli.py (Similar to run_benchmark.py) ***
            cmd = [
                PYTHON_EXE,
                eval_script_abs_path,
                "--model-path", model_path,
                "--model-type", model_type,
                "--benchmark-name", actual_benchmark_name, # Use potentially adjusted name or original
                "--benchmark-path", bench_path,
                "--results-dir", RESULTS_DIR,
                "--num-samples", str(NUM_SAMPLES), # Pass num_samples always
                # Potentially add trust_remote_code here if evaluate_cli needs it
                # "--trust-remote-code", # Example, adjust arg name if needed
            ]

            # Add CTI subset if applicable
            if subset and bench_name == "ctibench":
                 cmd.extend(["--cti-subset", subset]) # Pass subset separately

            # Add max-new-tokens only for causal models
            if model_type == "causal":
                cmd.extend(["--max-new-tokens", str(MAX_NEW_TOKENS)])


            logging.info(f"Executing command: {' '.join(cmd)}")

            try:
                # Execute the command (add cwd for consistency)
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False, # Don't raise exception on non-zero exit
                    env=os.environ.copy(),
                    cwd=WORKSPACE_ROOT # Run from workspace root like run_benchmark.py
                )

                # Log stdout/stderr
                if process.stdout: # Log stdout only if it exists
                    logging.info(f"Stdout:\n{process.stdout}")
                if process.stderr:
                    logging.warning(f"Stderr:\n{process.stderr}") # Log stderr as warning

                # Check the execution result
                if process.returncode != 0:
                    error_message = (
                        f"ERROR during: {run_desc}\n"
                        f"  Command: {' '.join(cmd)}\n"
                        f"  Return Code: {process.returncode}\n"
                        f"  Stderr:\n{process.stderr}\n"
                        # f"  Stdout:\n{process.stdout}\n" # Stderr usually more informative for errors
                        f"----------------------------------------\n"
                    )
                    logging.error(error_message) # Log to console and both files
                    # Manual write just in case logging config failed
                    with open(ERROR_LOG_FILE, 'a') as f_err:
                        f_err.write(f"[{datetime.now()}] {error_message}")
                else:
                    logging.info(f"SUCCESS: {run_desc}. Check output files in {RESULTS_DIR}/")

            except Exception as e:
                 # Catch exceptions during the subprocess call itself
                 error_message = (
                     f"EXCEPTION during subprocess execution: {run_desc}\n"
                     f"  Command: {' '.join(cmd)}\n"
                     f"  Exception Type: {type(e).__name__}\n"
                     f"  Exception: {e}\n"
                     f"----------------------------------------\n"
                 )
                 logging.error(error_message)
                 # Manual write just in case logging config failed
                 with open(ERROR_LOG_FILE, 'a') as f_err:
                    f_err.write(f"[{datetime.now()}] {error_message}")


logging.info("--- Security Benchmark Automation Finished (QUANTIZED MODELS) ---") 
