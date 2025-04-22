import subprocess
import os
import sys
import logging
from datetime import datetime

# --- Configuration ---

PYTHON_EXE = "/workspace/testbedvenv/bin/python" # Path to python executable in the venv
EVAL_SCRIPT_REL_PATH = "code/model_evaluator/evaluate_cli.py" # Relative to workspace root
WORKSPACE_ROOT = "/workspace/" # Assuming script is run from this root or we adjust paths
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results/bare/48")
ERROR_LOG_FILE = os.path.join(RESULTS_DIR, "automation_errors.log")
NUM_SAMPLES = 100
MAX_NEW_TOKENS = 512 # Used only for causal models

MODELS = [
    {
        "name": "TinyLlama-1.1B-Chat-v1.0",
        "path": os.path.join(WORKSPACE_ROOT, "models/TinyLlama-1.1B-Chat-v1.0/"),
        "type": "causal",
    },
    {
        "name": "Phi-3-mini-4k-instruct",
        "path": os.path.join(WORKSPACE_ROOT, "models/Phi-3-mini-4k-instruct/"),
        "type": "causal",
    },
    {
        "name": "Mistral-7B-Instruct-v0.3", # Assuming v0.3 based on benchmark-security.md example command path
        "path": os.path.join(WORKSPACE_ROOT, "models/Mistral-7B-Instruct-v0.3/"),
        "type": "causal",
    },
    {
        "name": "electra-small-discriminator",
        "path": os.path.join(WORKSPACE_ROOT, "models/electra-small-discriminator/"),
        "type": "encoder",
    },
]

BENCHMARKS = [
    {
        "name": "cyberseceval3_mitre",
        "path": os.path.join(WORKSPACE_ROOT, "code/PurpleLlama/CybersecurityBenchmarks/datasets/mitre/mitre_benchmark_100_per_category_with_augmentation.json"),
        "subsets": [None], # No subset needed
    },
    {
        "name": "sevenllm_bench",
        "path": os.path.join(WORKSPACE_ROOT, "calibration_data/sevenllm_instruct_subset_manual/"),
        "subsets": [None], # No subset needed
    },
    {
        "name": "ctibench",
        "path": os.path.join(WORKSPACE_ROOT, "datasets"), # Path to cache/dataset dir for HF
        "subsets": ["cti-mcq", "cti-vsp", "cti-rcm"], # Subsets to run
    },
]

# --- Setup Logging ---
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ERROR_LOG_FILE, mode='a'), # Append errors to this file
        logging.StreamHandler(sys.stdout) # Also print info messages to console
    ]
)

# --- Main Loop ---
eval_script_abs_path = os.path.join(WORKSPACE_ROOT, EVAL_SCRIPT_REL_PATH)
total_runs = sum(len(b["subsets"]) for b in BENCHMARKS) * len(MODELS)
current_run = 0

logging.info(f"--- Starting Benchmark Automation ---")
logging.info(f"Python Executable: {PYTHON_EXE}")
logging.info(f"Evaluation Script: {eval_script_abs_path}")
logging.info(f"Results Directory: {RESULTS_DIR}")
logging.info(f"Error Log File: {ERROR_LOG_FILE}")
logging.info(f"Models to run: {[m['name'] for m in MODELS]}")
logging.info(f"Benchmarks to run: {[(b['name'], b['subsets']) for b in BENCHMARKS]}")
logging.info(f"Total runs planned: {total_runs}")

for model_config in MODELS:
    model_name = model_config["name"]
    model_path = model_config["path"]
    model_type = model_config["type"]

    for bench_config in BENCHMARKS:
        bench_name = bench_config["name"]
        bench_path = bench_config["path"]

        for subset in bench_config["subsets"]:
            current_run += 1
            run_desc = f"Run {current_run}/{total_runs}: Model='{model_name}', Benchmark='{bench_name}'"
            if subset:
                run_desc += f", Subset='{subset}'"
            logging.info(f"\n{run_desc}")

            # Construct command
            cmd = [
                PYTHON_EXE,
                eval_script_abs_path,
                "--model-path", model_path,
                "--model-type", model_type,
                "--benchmark-name", bench_name,
                "--benchmark-path", bench_path,
                "--results-dir", RESULTS_DIR,
                "--num-samples", str(NUM_SAMPLES),
            ]

            if subset:
                cmd.extend(["--cti-subset", subset])

            # Add max-new-tokens only for causal models
            if model_type == "causal":
                cmd.extend(["--max-new-tokens", str(MAX_NEW_TOKENS)])

            logging.info(f"Executing command: {' '.join(cmd)}")

            try:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False, # Don't raise exception on non-zero exit
                    cwd=WORKSPACE_ROOT # Run from workspace root
                )

                if process.returncode != 0:
                    error_message = (
                        f"ERROR during: {run_desc}\n"
                        f"  Command: {' '.join(cmd)}\n"
                        f"  Return Code: {process.returncode}\n"
                        f"  Stderr:\n{process.stderr}\n"
                        f"----------------------------------------\n"
                    )
                    logging.error(error_message) # Will log to both console and file
                    # Also write directly just in case basicConfig failed silently
                    with open(ERROR_LOG_FILE, 'a') as f_err:
                        f_err.write(f"[{datetime.now()}] {error_message}")
                else:
                    logging.info(f"SUCCESS: {run_desc}")
                    # Optionally log stdout for successful runs if needed
                    # logging.debug(f"Stdout:\n{process.stdout}")

            except Exception as e:
                 # Catch errors in subprocess execution itself
                 error_message = (
                     f"EXCEPTION during subprocess execution: {run_desc}\n"
                     f"  Command: {' '.join(cmd)}\n"
                     f"  Exception: {e}\n"
                     f"----------------------------------------\n"
                 )
                 logging.error(error_message)
                 with open(ERROR_LOG_FILE, 'a') as f_err:
                    f_err.write(f"[{datetime.now()}] {error_message}")


logging.info(f"--- Benchmark Automation Finished ---")
