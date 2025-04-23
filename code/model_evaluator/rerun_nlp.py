# path: /workspace/code/run_nlp.py
import subprocess
import os
import sys
import logging
from datetime import datetime
import re # For sanitizing names

# --- Configuration ---
PYTHON_EXE = "/workspace/testbedvenv/bin/python" # Path to python executable in the venv
WORKSPACE_ROOT = "/workspace/"
LM_EVAL_HARNESS_DIR = os.path.join(WORKSPACE_ROOT, "code/lm-evaluation-harness") # Path to the cloned harness repo
LM_EVAL_COMMAND = "lm_eval" # Use the installed command-line tool
RESULTS_BASE_DIR = os.path.join(WORKSPACE_ROOT, "results/nlp/bare/48") # Desired output base directory
ERROR_LOG_FILE = os.path.join(RESULTS_BASE_DIR, "lm_eval_harness_rerun_errors.log") # New log file for this run
NUM_FEWSHOT = 0 # Use 0-shot evaluation as requested
BATCH_SIZE = "auto" # Let the harness determine batch size. Adjust to a number (e.g., 4) if OOM occurs.

# Models to evaluate (excluding Electra)
MODELS = [
    {
        "name": "TinyLlama-1.1B-Chat-v1.0",
        "model_args": f"pretrained={os.path.join(WORKSPACE_ROOT, 'models/TinyLlama-1.1B-Chat-v1.0/')}",
    },
    {
        "name": "Phi-3-mini-4k-instruct",
        "model_args": f"pretrained={os.path.join(WORKSPACE_ROOT, 'models/Phi-3-mini-4k-instruct/')},trust_remote_code=True",
    },
    {
        "name": "Mistral-7B-Instruct-v0.3",
        "model_args": f"pretrained={os.path.join(WORKSPACE_ROOT, 'models/Mistral-7B-Instruct-v0.3/')},trust_remote_code=True",
    },
]

# *** UPDATED TASKS LIST ***
# List of lm-evaluation-harness tasks to run (only the failed/corrected ones)
TASKS = [
    "mmlu_elementary_mathematics", # Specific MMLU subset
    "truthfulqa_mc2",             # Corrected TruthfulQA MC task name (using mc2 variant)
    # If truthfulqa_mc2 fails, try "truthfulqa_mc1"
]

# --- Utility Function ---
def sanitize_filename(name):
    """Removes or replaces characters problematic for filenames/paths."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.replace(' ', '_')
    return name

# --- Setup Logging ---
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
log_mode = 'w' # Overwrite log file for this specific re-run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ERROR_LOG_FILE, mode=log_mode),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Main Loop ---
total_runs = len(MODELS) * len(TASKS)
current_run = 0

logging.info(f"--- Starting LM Evaluation Harness RERUN (Failed Tasks) ---")
logging.info(f"Using command: {LM_EVAL_COMMAND}")
logging.info(f"Harness Directory (for cwd): {LM_EVAL_HARNESS_DIR}")
logging.info(f"Results Base Directory: {RESULTS_BASE_DIR}")
logging.info(f"Error Log File: {ERROR_LOG_FILE}")
logging.info(f"Models to run: {[m['name'] for m in MODELS]}")
logging.info(f"Tasks to run: {TASKS}")
logging.info(f"Num Fewshot: {NUM_FEWSHOT}")
logging.info(f"Batch Size: {BATCH_SIZE}")
logging.info(f"Total runs planned: {total_runs}")

# --- Pre-run Check ---
if not os.path.isdir(LM_EVAL_HARNESS_DIR):
    logging.error(f"CRITICAL ERROR: lm-evaluation-harness directory not found at: {LM_EVAL_HARNESS_DIR}")
    logging.error("Please ensure the path is correct and the repository is cloned.")
    sys.exit(1)

logging.info("Verifying lm-evaluation-harness installation...")
try:
    help_check = subprocess.run([LM_EVAL_COMMAND, "--help"], capture_output=True, text=True, check=True, env=os.environ.copy())
    logging.info("lm_eval command found and appears runnable.")
except (subprocess.CalledProcessError, FileNotFoundError) as e:
     logging.error(f"CRITICAL ERROR: '{LM_EVAL_COMMAND}' command not found or failed to run.")
     logging.error(f"Please ensure lm-evaluation-harness is installed correctly (pip install -e . in {LM_EVAL_HARNESS_DIR} with venv active).")
     logging.error(f"Underlying error: {e}")
     sys.exit(1)


for model_config in MODELS:
    model_name = model_config["name"]
    model_args = model_config["model_args"]
    sanitized_model_name = sanitize_filename(model_name)

    for task in TASKS:
        current_run += 1
        run_desc = f"Run {current_run}/{total_runs}: Model='{model_name}', Task='{task}'"
        logging.info(f"\n{run_desc}")

        # Define specific output directory for this run's results
        # Use sanitize_filename for the task name as well to handle potential slashes in MMLU names if used later
        sanitized_task_name = sanitize_filename(task)
        output_dir_for_run = os.path.join(RESULTS_BASE_DIR, sanitized_model_name, sanitized_task_name)
        output_path_for_run_file = os.path.join(output_dir_for_run, "results.json")
        os.makedirs(output_dir_for_run, exist_ok=True)

        # Construct the command using the lm_eval tool
        cmd = [
            LM_EVAL_COMMAND,
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", task,
            "--num_fewshot", str(NUM_FEWSHOT),
            "--batch_size", str(BATCH_SIZE),
            "--device", "cuda:0",
            "--output_path", output_path_for_run_file,
            # Optional: Uncomment to limit samples per task for quick testing
            # "--limit", "10",
            # Optional: Uncomment to save detailed prediction outputs
            # "--log_samples",
        ]

        logging.info(f"Executing command: {' '.join(cmd)}")

        try:
            process = subprocess.run(
                cmd,
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


logging.info(f"--- LM Evaluation Harness RERUN Finished ---")
