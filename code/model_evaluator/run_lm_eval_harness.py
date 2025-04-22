# path: /workspace/code/run_lm_eval_harness.py
import subprocess
import os
import sys
import logging
from datetime import datetime
import re # For sanitizing names

# --- Configuration ---
PYTHON_EXE = "/workspace/testbedvenv/bin/python" # Path to python executable in the venv
WORKSPACE_ROOT = "/workspace/"
LM_EVAL_HARNESS_DIR = os.path.join(WORKSPACE_ROOT, "code/lm-evaluation-harness") # CORRECTED: Path to the cloned harness repo
# LM_EVAL_MAIN_SCRIPT = os.path.join(LM_EVAL_HARNESS_DIR, "main.py") # REMOVED: Not needed when using lm_eval command
LM_EVAL_COMMAND = "lm_eval" # Use the installed command-line tool
RESULTS_BASE_DIR = os.path.join(WORKSPACE_ROOT, "results/nlp/bare/48") # Desired output base directory
ERROR_LOG_FILE = os.path.join(RESULTS_BASE_DIR, "lm_eval_harness_automation_errors.log")
NUM_FEWSHOT = 0 # Use 0-shot evaluation as requested
BATCH_SIZE = "auto" # Let the harness determine batch size. Adjust to a number (e.g., 4) if OOM occurs.

# Models to evaluate (excluding Electra)
MODELS = [
    {
        "name": "TinyLlama-1.1B-Chat-v1.0",
        "model_args": f"pretrained={os.path.join(WORKSPACE_ROOT, 'models/TinyLlama-1.1B-Chat-v1.0/')}",
        # trust_remote_code=False (default, usually fine for Llama)
    },
    {
        "name": "Phi-3-mini-4k-instruct",
        "model_args": f"pretrained={os.path.join(WORKSPACE_ROOT, 'models/Phi-3-mini-4k-instruct/')},trust_remote_code=True",
        # trust_remote_code=True often needed for Phi-3
    },
    {
        "name": "Mistral-7B-Instruct-v0.3",
        "model_args": f"pretrained={os.path.join(WORKSPACE_ROOT, 'models/Mistral-7B-Instruct-v0.3/')},trust_remote_code=True",
        # trust_remote_code=True added just in case, usually not needed but safer
    },
]

# List of lm-evaluation-harness tasks to run
TASKS = [
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "mmlu",
    "truthfulqa_mc",
    "gsm8k",
]

# --- Utility Function ---
def sanitize_filename(name):
    """Removes or replaces characters problematic for filenames/paths."""
    # Replace common problematic characters with underscore
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace spaces with underscore
    name = name.replace(' ', '_')
    # Optional: Ensure it doesn't end with problematic chars like '.' if needed
    # name = name.rstrip('._')
    return name

# --- Setup Logging ---
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
# Clear previous log file content if desired, or keep append mode ('a')
log_mode = 'w' # 'w' to overwrite on new run, 'a' to append
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

logging.info(f"--- Starting LM Evaluation Harness Automation ---")
# logging.info(f"Python Executable: {PYTHON_EXE}") # Less relevant now
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
    sys.exit(1) # Exit if the directory doesn't exist

# Ensure lm-evaluation-harness is installed in the environment
logging.info("Verifying lm-evaluation-harness installation...")
try:
    # Check if lm_eval command exists and is runnable from the venv
    # This runs `lm_eval --help` and checks the return code.
    help_check = subprocess.run([LM_EVAL_COMMAND, "--help"], capture_output=True, text=True, check=True, env=os.environ.copy())
    logging.info("lm_eval command found and appears runnable.")
except (subprocess.CalledProcessError, FileNotFoundError) as e:
     logging.error(f"CRITICAL ERROR: '{LM_EVAL_COMMAND}' command not found or failed to run.")
     logging.error(f"Please ensure lm-evaluation-harness is installed correctly in the environment:")
     logging.error(f"1. Activate the venv: source {os.path.join(os.path.dirname(PYTHON_EXE), 'activate')}")
     logging.error(f"2. Navigate to the harness dir: cd {LM_EVAL_HARNESS_DIR}")
     logging.error(f"3. Install with: pip install -e .")
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
        output_dir_for_run = os.path.join(RESULTS_BASE_DIR, sanitized_model_name, task)
        # Define the specific path for the results JSON file
        output_path_for_run_file = os.path.join(output_dir_for_run, "results.json") # lm_eval saves to a file
        os.makedirs(output_dir_for_run, exist_ok=True)

        # Construct the command using the lm_eval tool
        cmd = [
            LM_EVAL_COMMAND,
            "--model", "hf",              # Specify Hugging Face model type
            "--model_args", model_args,    # Pass model path and options
            "--tasks", task,               # Specify the benchmark task
            "--num_fewshot", str(NUM_FEWSHOT), # Set number of few-shot examples
            "--batch_size", str(BATCH_SIZE), # Set batch size (or 'auto')
            "--device", "cuda:0",          # Specify GPU device
            "--output_path", output_path_for_run_file, # File path to save results JSON
            # Optional: Uncomment to limit samples per task for quick testing
            # "--limit", "50",
            # Optional: Uncomment to save detailed prediction outputs (saves samples.jsonl in output_path dir)
            # "--log_samples",
        ]

        logging.info(f"Executing command: {' '.join(cmd)}")

        try:
            # Execute the command. Running from the harness directory might be safer
            # for relative path resolution within the harness code, though often not strictly required
            # if paths in arguments are absolute. Let's keep cwd for safety.
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False, # Do not raise exception on non-zero exit code
                cwd=LM_EVAL_HARNESS_DIR, # Run from the harness directory
                env=os.environ.copy() # Pass the current environment variables (includes PATH)
            )

            # Check the execution result
            if process.returncode != 0:
                error_message = (
                    f"ERROR during: {run_desc}\n"
                    f"  Command: {' '.join(cmd)}\n"
                    f"  Return Code: {process.returncode}\n"
                    f"  Output File Target: {output_path_for_run_file}\n"
                    f"  Stderr:\n{process.stderr}\n"
                    f"  Stdout:\n{process.stdout}\n" # Include stdout in errors
                    f"----------------------------------------\n"
                )
                logging.error(error_message)
                # Ensure error is written to the log file
                with open(ERROR_LOG_FILE, 'a') as f_err: # Ensure append mode here after initial clear
                    f_err.write(f"[{datetime.now()}] {error_message}")
            else:
                logging.info(f"SUCCESS: {run_desc}. Results saved in {output_path_for_run_file}")
                # Optionally log stdout for successful runs if needed (can be very verbose)
                # logging.debug(f"Stdout:\n{process.stdout}")

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
             with open(ERROR_LOG_FILE, 'a') as f_err: # Append mode
                f_err.write(f"[{datetime.now()}] {error_message}")


logging.info(f"--- LM Evaluation Harness Automation Finished ---")

