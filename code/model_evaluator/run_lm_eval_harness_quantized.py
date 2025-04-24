import subprocess
import os
import sys
import logging
from datetime import datetime
import re # For sanitizing names
import argparse # Import argparse earlier

# --- Configuration ---
PYTHON_EXE = "/workspace/testbedvenv/bin/python" # Path to python executable in the venv
WORKSPACE_ROOT = "/workspace/"
LM_EVAL_HARNESS_DIR = os.path.join(WORKSPACE_ROOT, "code/lm-evaluation-harness")
# LM_EVAL_COMMAND = "lm_eval" # REMOVED - We will call it as a module
RESULTS_BASE_DIR = os.path.join(WORKSPACE_ROOT, "results/nlp/mod/48") # Quantized results
ERROR_LOG_FILE = os.path.join(RESULTS_BASE_DIR, "lm_eval_harness_quantized_errors.log") # Specific log
NUM_FEWSHOT = 0
BATCH_SIZE = "auto"

# Models to evaluate (Quantized versions)
MODELS = [
    {
        "name": "TinyLlama-1.1B-Chat-v1.0-GPTQ", # Renamed for clarity
        # ADD device_map="auto"
        "model_args": f"pretrained={os.path.join(WORKSPACE_ROOT, 'models/TinyLlama-1.1B-Chat-v1.0/quantized-gptq-4bit/')},disable_exllama=True,device_map='auto'",
    },
    {
        "name": "Phi-3-mini-4k-instruct-GPTQ", # Renamed for clarity
        # ADD device_map="auto"
        "model_args": f"pretrained={os.path.join(WORKSPACE_ROOT, 'models/Phi-3-mini-4k-instruct/quantized-gptq-4bit/')},trust_remote_code=True,disable_exllama=True,device_map='auto'",
    },
    {
        "name": "Mistral-7B-Instruct-v0.3-GPTQ", # Renamed for clarity
        # ADD device_map="auto"
        "model_args": f"pretrained={os.path.join(WORKSPACE_ROOT, 'models/Mistral-7B-Instruct-v0.3/quantized-gptq-4bit/')},trust_remote_code=True,disable_exllama=True,device_map='auto'",
    },
]

# List of lm-evaluation-harness tasks to run (same as original successful run)
TASKS = [
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "mmlu_elementary_mathematics", # Use the specific successful one
    "truthfulqa_mc2",             # Use the specific successful one
    "gsm8k",
]

# --- Utility Function ---
def sanitize_filename(name):
    """Removes or replaces characters problematic for filenames/paths."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.replace(' ', '_')
    return name

# --- Setup Logging ---
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
log_mode = 'w'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ERROR_LOG_FILE, mode=log_mode),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Argument Parsing (for --limit) ---
parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=str, default=None)
args, unknown = parser.parse_known_args()

# --- Main Loop ---
total_runs = len(MODELS) * len(TASKS)
current_run = 0

logging.info(f"--- Starting LM Evaluation Harness Automation (QUANTIZED MODELS) ---")
# logging.info(f"Using command: {LM_EVAL_COMMAND}") # Updated log message
logging.info(f"Using Python executable: {PYTHON_EXE}")
logging.info(f"Running lm_eval as module: python -m lm_eval")
logging.info(f"Harness Directory (for cwd): {LM_EVAL_HARNESS_DIR}")
logging.info(f"Results Base Directory: {RESULTS_BASE_DIR}")
logging.info(f"Error Log File: {ERROR_LOG_FILE}")
logging.info(f"Models to run: {[m['name'] for m in MODELS]}")
logging.info(f"Tasks to run: {TASKS}")
logging.info(f"Num Fewshot: {NUM_FEWSHOT}")
logging.info(f"Batch Size: {BATCH_SIZE}")
logging.info(f"Applying limit: {args.limit}" if args.limit else "No limit applied.")
logging.info(f"Total runs planned: {total_runs}")

# --- Pre-run Check ---
if not os.path.isdir(LM_EVAL_HARNESS_DIR):
    logging.error(f"CRITICAL ERROR: lm-evaluation-harness directory not found at: {LM_EVAL_HARNESS_DIR}")
    sys.exit(1)
if not os.path.isfile(PYTHON_EXE):
     logging.error(f"CRITICAL ERROR: Python executable not found at: {PYTHON_EXE}")
     sys.exit(1)

logging.info("Verifying lm-evaluation-harness installation via python -m lm_eval --help...")
try:
    # Check if lm_eval can be run as a module using the specified Python executable
    help_cmd = [PYTHON_EXE, "-m", "lm_eval", "--help"]
    help_check = subprocess.run(help_cmd, capture_output=True, text=True, check=True, env=os.environ.copy())
    logging.info("lm_eval module found and appears runnable via specified Python.")
except (subprocess.CalledProcessError, FileNotFoundError) as e:
     logging.error(f"CRITICAL ERROR: Failed to run 'python -m lm_eval --help' using {PYTHON_EXE}.")
     logging.error(f"Please ensure lm-evaluation-harness is installed correctly in the environment:")
     logging.error(f"1. Activate the venv: source {os.path.join(os.path.dirname(PYTHON_EXE), 'activate')}")
     logging.error(f"2. Navigate to the harness dir: cd {LM_EVAL_HARNESS_DIR}")
     logging.error(f"3. Install with: pip install -e \".[gptq]\"") # Ensure gptq extra is installed
     logging.error(f"Underlying error: {e}")
     if isinstance(e, subprocess.CalledProcessError):
         logging.error(f"Stderr:\n{e.stderr}")
         logging.error(f"Stdout:\n{e.stdout}")
     sys.exit(1)


for model_config in MODELS:
    model_name = model_config["name"]
    model_args = model_config["model_args"]
    # Sanitize the potentially modified name
    sanitized_model_name = sanitize_filename(model_name)

    for task in TASKS:
        current_run += 1
        run_desc = f"Run {current_run}/{total_runs}: Model='{model_name}', Task='{task}'"
        logging.info(f"\n{run_desc}")

        # Define specific output directory for this run's results
        sanitized_task_name = sanitize_filename(task)
        output_dir_for_run = os.path.join(RESULTS_BASE_DIR, sanitized_model_name, sanitized_task_name)
        output_path_for_run_file = os.path.join(output_dir_for_run, "results.json")
        os.makedirs(output_dir_for_run, exist_ok=True)

        # *** Construct the command using 'python -m lm_eval' ***
        cmd = [
            PYTHON_EXE, # Explicitly use the venv python
            "-m", "lm_eval", # Run lm_eval as a module
            "--model", "hf",
            "--model_args", model_args, # Now includes device_map='auto'
            "--tasks", task,
            "--num_fewshot", str(NUM_FEWSHOT),
            "--batch_size", str(BATCH_SIZE), # Let device_map handle placement
            "--device", "cuda:0", # Keep this, lm-eval might still use it conceptually
            "--output_path", output_path_for_run_file,
            # Optional: Uncomment to save detailed prediction outputs
            # "--log_samples",
        ]

        # Add limit if provided
        if args.limit:
             cmd.extend(["--limit", args.limit])

        logging.info(f"Executing command: {' '.join(cmd)}")

        try:
            # Execute the command. Running from the harness directory is still a good idea.
            process = subprocess.run(
                cmd, # Use the modified command list
                capture_output=True,
                text=True,
                check=False,
                cwd=LM_EVAL_HARNESS_DIR,
                env=os.environ.copy()
            )

            if process.returncode != 0:
                error_message = (
                    f"ERROR during: {run_desc}\n"
                    f"  Command: {' '.join(cmd)}\n"
                    f"  Return Code: {process.returncode}\n"
                    f"  Output File Target: {output_path_for_run_file}\n"
                    f"  Stderr:\n{process.stderr}\n"
                    f"  Stdout:\n{process.stdout}\n"
                    f"----------------------------------------\n"
                )
                logging.error(error_message)
                with open(ERROR_LOG_FILE, 'a') as f_err:
                    f_err.write(f"[{datetime.now()}] {error_message}")
            else:
                logging.info(f"SUCCESS: {run_desc}. Results saved in {output_path_for_run_file}")

        except Exception as e:
             error_message = (
                 f"EXCEPTION during subprocess execution: {run_desc}\n"
                 f"  Command: {' '.join(cmd)}\n"
                 f"  Exception Type: {type(e).__name__}\n"
                 f"  Exception: {e}\n"
                 f"----------------------------------------\n"
             )
             logging.error(error_message)
             with open(ERROR_LOG_FILE, 'a') as f_err:
                f_err.write(f"[{datetime.now()}] {error_message}")


logging.info(f"--- LM Evaluation Harness Automation Finished (QUANTIZED MODELS) ---") 
