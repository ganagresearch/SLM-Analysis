import pandas as pd
import json
import glob
import os
import re
from datetime import datetime
import logging # Added for better error reporting

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Mapping from model directory base names to parameter counts (adjust as needed)
MODEL_PARAMS = {
    "Mistral-7B-Instruct-v0.3": "7B",
    "phi-3-mini-4k-instruct": "3.8B", # Adjusted key based on path
    "Phi-3-mini-4k-instruct": "3.8B", # Adding variation seen in path
    "TinyLlama-1.1B-Chat-v1.0": "1.1B",
    "electra-small-discriminator": "14M",
    # Add quantized model names here later if needed, or handle dynamically
}

# Define columns for the final DataFrame
# Added score_stderr, split task_subset into task_name and fewshot
# Removed inference_speed_samples_per_sec as less relevant for NLP eval harness
DATAFRAME_COLUMNS = [
    "model_name",
    "parameters",
    "quantization_status",
    "benchmark_suite",
    "benchmark_name", # e.g., arc_challenge, hellaswag
    "task_subset", # Primarily for ctibench, null for others
    "score_metric", # e.g., acc_norm, mc2
    "score_value",
    "score_stderr", # Standard error if available
    "model_size_disk_gb", # Added calculation
    "peak_vram_gpu_mb",
    "peak_ram_system_mb",
    "load_time_seconds",
    "inference_speed_tok_per_sec",
    # "inference_speed_samples_per_sec", # Likely not available/relevant from lm-eval
    "avg_inference_time_per_sample_ms", # Likely not available from lm-eval
    "num_fewshot", # Added from lm-eval config
    "hardware",
    "evaluation_script",
    "timestamp",
    "model_path", # Added to store path for size calculation
    "raw_json_path"
]


# --- Helper Functions ---

def parse_model_name(path):
    """Extracts the base model name from a path."""
    path = os.path.normpath(path)
    base_name = os.path.basename(path)
    # Handle variations like Phi-3 vs phi-3
    if base_name == "Phi-3-mini-4k-instruct": base_name = "phi-3-mini-4k-instruct"
    # If it looks like a quantization suffix, go one level up
    if base_name.endswith("-int8-ptq-onnx"):
         parent_dir = os.path.dirname(path)
         base_name = os.path.basename(parent_dir)
         if base_name == "Phi-3-mini-4k-instruct": base_name = "phi-3-mini-4k-instruct"
    return base_name

def parse_timestamp_from_filename(filename):
    """Extracts YYYYMMDD-HHMMSS timestamp from standard filenames."""
    # Try specific lm-eval timestamp format first
    match_lm_eval = re.search(r'results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.\d+\.json', filename)
    if match_lm_eval:
        try:
            # Replace T with space for strptime
            ts_str = match_lm_eval.group(1).replace('T', ' ')
            return datetime.strptime(ts_str, '%Y-%m-%d %H-%M-%S')
        except ValueError:
            pass # Fall through to other methods

    # Try security benchmark format
    match_sec = re.search(r'(\d{8}-\d{6})', filename)
    if match_sec:
        try:
            return datetime.strptime(match_sec.group(1), '%Y%m%d-%H%M%S')
        except ValueError:
            pass # Fall through

    # Fallback: Try to get timestamp from file modification time
    try:
        return datetime.fromtimestamp(os.path.getmtime(os.path.dirname(filename))) # Use parent dir mod time for lm-eval
    except Exception:
        return None

def get_model_name_from_nlp_path(file_path):
    """Estimates model name from the new lm-eval output path structure."""
    # Example: results/nlp/bare/48/Phi-3-mini-4k-instruct/arc_challenge/results.json/__workspace__models__Phi-3-mini-4k-instruct__/results_....json
    parts = os.path.normpath(file_path).split(os.sep)
    # Expected structure: ... / <quant_status> / <gpu_mem> / <model_name> / <task_name> / results.json / <sanitized_model> / results_....json
    if len(parts) >= 5:
        model_name = parts[-5] # Assumes model name is 5 levels up from file
        # Handle variations like Phi-3 vs phi-3
        if model_name == "Phi-3-mini-4k-instruct": model_name = "phi-3-mini-4k-instruct"
        return model_name
    logging.warning(f"Could not determine model name from path structure: {file_path}")
    return "Unknown"

def get_model_path_from_nlp_config(config):
    """Extracts model path from lm-eval config if present."""
    model_args = config.get("model_args", "")
    match = re.search(r'pretrained=([^,]+)', model_args)
    if match:
        path = match.group(1)
        # Simple check if path exists (might be inside container)
        # if os.path.exists(path): # This check might fail if path is relative to script execution
        return path.strip()
    return None

def get_dir_size(start_path):
    """Calculates the total size of a directory in Gigabytes (GB)."""
    if not start_path or not os.path.isdir(start_path):
        logging.warning(f"Cannot calculate size: Path '{start_path}' is not a valid directory.")
        return None
    total_size_bytes = 0
    try:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link (optional)
                if not os.path.islink(fp):
                    total_size_bytes += os.path.getsize(fp)
    except OSError as e:
        logging.error(f"Error walking directory {start_path}: {e}")
        return None
    # Convert bytes to Gigabytes (using 1024^3)
    return round(total_size_bytes / (1024**3), 3)


# --- Parsing Functions ---

def parse_security_metrics_json(file_path):
    """Parses a single *_metrics.json file from evaluate_cli.py runs."""
    data = {}
    try:
        with open(file_path, 'r') as f:
            metrics = json.load(f)

        run_args = metrics.get("run_args", {})
        load_metrics = metrics.get("load_metrics", {})
        inference_metrics = metrics.get("inference_metrics", {})

        # Store model path
        model_path = run_args.get("model_path")
        data["model_path"] = model_path

        model_base_name = parse_model_name(model_path if model_path else "")
        if not model_base_name or model_base_name == "Unknown":
             logging.warning(f"Could not parse model name from path '{model_path}' in {file_path}")
             model_base_name = "Unknown"


        # --- Extract Fields ---
        data["model_name"] = model_base_name
        data["parameters"] = MODEL_PARAMS.get(model_base_name, "Unknown")
        data["quantization_status"] = "baseline" # Assuming bare dir is baseline
        data["benchmark_suite"] = "Security"
        data["benchmark_name"] = run_args.get("benchmark_name", "Unknown")
        data["task_subset"] = run_args.get("cti_subset") # Will be None if not present
        if data["task_subset"] is None and data["benchmark_name"] != "ctibench":
             data["task_subset"] = "full" # Assign 'full' if not ctibench subset

        # Resource Usage & Speed (Specific to evaluate_cli.py output)
        data["peak_vram_gpu_mb"] = inference_metrics.get("pytorch_vram_peak_inference_mb")
        data["peak_ram_system_mb"] = inference_metrics.get("ram_after_inference_mb")
        data["load_time_seconds"] = load_metrics.get("load_time_sec")
        data["inference_speed_tok_per_sec"] = inference_metrics.get("tokens_per_second")
        avg_time_sec = inference_metrics.get("avg_generate_time_per_sample_sec")
        data["avg_inference_time_per_sample_ms"] = avg_time_sec * 1000 if avg_time_sec is not None else None

        # Context
        data["hardware"] = "NVIDIA A40"
        data["evaluation_script"] = "evaluate_cli.py"
        # Use filename for timestamp parsing here
        data["timestamp"] = parse_timestamp_from_filename(os.path.basename(file_path))
        data["raw_json_path"] = file_path

        # Placeholders for fields not present in security metrics
        data["score_metric"] = None # Scores need separate processing
        data["score_value"] = None
        data["score_stderr"] = None
        data["num_fewshot"] = 0 # Assuming 0-shot for security script
        data["model_size_disk_gb"] = None # Calculated later


    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error parsing security file {file_path}: {e}")
        return None

    # Return only keys defined in DATAFRAME_COLUMNS, ensure all exist
    filtered_data = {col: data.get(col) for col in DATAFRAME_COLUMNS}
    return filtered_data


def parse_nlp_metrics_json(file_path):
    """Parses a single results_*.json file from lm-evaluation-harness runs."""
    parsed_records = []
    try:
        with open(file_path, 'r') as f:
            metrics = json.load(f)

        config = metrics.get("config", {})
        results = metrics.get("results", {})
        n_shot_info = metrics.get("n-shot", {}) # Get n-shot info

        if not results:
            logging.warning(f"No 'results' key found in {file_path}. Skipping.")
            return []

        # Get model path from config
        model_path = get_model_path_from_nlp_config(config)

        # Infer model name from path (primary) or config (fallback)
        model_base_name = get_model_name_from_nlp_path(file_path)
        if model_base_name == "Unknown" and model_path:
            model_base_name = parse_model_name(model_path) # Fallback to parsed path

        # Iterate through each task reported in the results dict
        for task_name, task_results in results.items():
            if not isinstance(task_results, dict):
                logging.warning(f"Task result for '{task_name}' in {file_path} is not a dict. Skipping.")
                continue

            # Determine num_fewshot for this task
            num_fewshot = config.get("num_fewshot", n_shot_info.get(task_name, 0)) # Use config first, fallback to n-shot dict

            # Extract common info
            base_data = {
                "model_name": model_base_name,
                "parameters": MODEL_PARAMS.get(model_base_name, "Unknown"),
                "quantization_status": "baseline", # Assuming bare dir is baseline
                "benchmark_suite": "NLP",
                "benchmark_name": task_name,
                "task_subset": None, # Typically None for lm-eval tasks
                "num_fewshot": num_fewshot,
                "hardware": "NVIDIA A40",
                "evaluation_script": "lm-evaluation-harness",
                # Use filename for timestamp parsing for lm-eval results file
                "timestamp": parse_timestamp_from_filename(os.path.basename(file_path)),
                "model_path": model_path, # Store model path
                "raw_json_path": file_path,
                # Metrics not typically available from lm-eval harness JSON
                "peak_vram_gpu_mb": None,
                "peak_ram_system_mb": None,
                "load_time_seconds": None,
                "inference_speed_tok_per_sec": None,
                "avg_inference_time_per_sample_ms": None,
                "model_size_disk_gb": None # Calculated later
            }

            # Create a separate record for each metric reported for the task
            for metric_key, score in task_results.items():
                # Clean the metric name (remove ,none suffix)
                metric_name = metric_key.split(',')[0]

                 # Look for a corresponding stderr field (also clean its key)
                stderr_key_clean = f"{metric_name}_stderr"
                stderr_val = None
                for k in task_results.keys(): # Search for the cleaned stderr key
                    if k.startswith(stderr_key_clean):
                        stderr_val = task_results.get(k)
                        break

                # Filter out alias fields and stderr entries themselves from being main metrics
                if metric_name == "alias" or metric_name.endswith("_stderr"):
                    continue

                record = base_data.copy()
                record["score_metric"] = metric_name
                record["score_value"] = score
                record["score_stderr"] = stderr_val

                # Add only keys defined in DATAFRAME_COLUMNS
                filtered_record = {col: record.get(col) for col in DATAFRAME_COLUMNS}
                parsed_records.append(filtered_record)

    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error parsing NLP file {file_path}: {e}")
        return []

    return parsed_records


# --- Aggregation Functions ---

def aggregate_results(results_base_dir, file_pattern, parser_func):
    """Generic function to find and parse result files."""
    # Adjusted glob pattern to search deeper for the actual JSON file
    search_pattern = os.path.join(results_base_dir, "**", file_pattern)
    all_metrics_files = glob.glob(search_pattern, recursive=True)
    logging.info(f"Found {len(all_metrics_files)} files matching '{search_pattern}' in {results_base_dir}")

    all_parsed_data = []
    processed_files = 0
    for f in all_metrics_files:
        # Ensure we are only parsing files, not directories named like files
        if not os.path.isfile(f):
             continue
        parsed = parser_func(f)
        if parsed:
            if isinstance(parsed, list):
                all_parsed_data.extend(parsed)
            else:
                all_parsed_data.append(parsed)
            processed_files += 1

    logging.info(f"Successfully parsed {processed_files} files.")
    if not all_parsed_data:
        logging.warning(f"No valid data parsed for pattern '{file_pattern}' in {results_base_dir}.")
        return pd.DataFrame(columns=DATAFRAME_COLUMNS)

    df = pd.DataFrame(all_parsed_data)

    # Add any missing columns just in case (shouldn't be needed if parsers are correct)
    for col in DATAFRAME_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Ensure correct order
    df = df.reindex(columns=DATAFRAME_COLUMNS, fill_value=None)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert numeric columns
    numeric_cols = [
        "peak_vram_gpu_mb", "peak_ram_system_mb", "load_time_seconds",
        "inference_speed_tok_per_sec", "avg_inference_time_per_sample_ms",
        "score_value", "score_stderr", "model_size_disk_gb", "num_fewshot"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def calculate_and_add_model_sizes(df):
    """Calculates disk size for unique model paths and merges back into df."""
    if 'model_path' not in df.columns:
        logging.warning("Column 'model_path' not found. Cannot calculate model sizes.")
        return df

    # Get unique, non-null model paths
    unique_paths = df['model_path'].dropna().unique()
    logging.info(f"Found {len(unique_paths)} unique model paths to calculate size for.")

    model_sizes = {}
    for path in unique_paths:
        size_gb = get_dir_size(path)
        if size_gb is not None:
            model_sizes[path] = size_gb
        else:
            logging.warning(f"Could not calculate size for path: {path}")


    # Map sizes back to the DataFrame
    df['model_size_disk_gb'] = df['model_path'].map(model_sizes)

    # Convert the new column to numeric, coercing errors
    df['model_size_disk_gb'] = pd.to_numeric(df['model_size_disk_gb'], errors='coerce')

    logging.info("Finished calculating and mapping model sizes.")
    return df


# --- Execution Block ---

if __name__ == "__main__":
    # Define base directories
    baseline_results_dir = "results/bare/48"
    nlp_results_dir = "results/nlp/bare/48" # Base dir for NLP results
    # quantized_results_dir = "results/mod/48" # Uncomment when ready
    output_dir = "results/aggregated"
    output_csv_file = os.path.join(output_dir, "baseline_all_metrics.csv")
    output_json_file = os.path.join(output_dir, "baseline_all_metrics.json")

    os.makedirs(output_dir, exist_ok=True)

    # --- Aggregate Baseline Results ---
    logging.info("Aggregating baseline security results...")
    baseline_sec_df = aggregate_results(baseline_results_dir, "*_metrics.json", parse_security_metrics_json)

    logging.info("Aggregating baseline NLP results...")
    # Update file pattern for lm-eval harness results based on new structure
    baseline_nlp_df = aggregate_results(nlp_results_dir, "results_*.json", parse_nlp_metrics_json)

    # --- Combine Baseline DataFrames ---
    if not baseline_sec_df.empty and not baseline_nlp_df.empty:
        logging.info("Combining security and NLP baseline results...")
        baseline_all_df = pd.concat([baseline_sec_df, baseline_nlp_df], ignore_index=True)
    elif not baseline_sec_df.empty:
        baseline_all_df = baseline_sec_df
    elif not baseline_nlp_df.empty:
        baseline_all_df = baseline_nlp_df
    else:
        baseline_all_df = pd.DataFrame() # Empty

    # --- Calculate Model Sizes ---
    if not baseline_all_df.empty:
        logging.info("Calculating model disk sizes...")
        baseline_all_df = calculate_and_add_model_sizes(baseline_all_df)
    else:
        logging.info("Skipping model size calculation as DataFrame is empty.")


    # --- Aggregate Quantized Results (Future) ---
    # logging.info("Aggregating quantized security results...")
    # quantized_sec_df = aggregate_results(quantized_results_dir, "*_metrics.json", parse_security_metrics_json) # Adapt parser if format differs
    # logging.info("Aggregating quantized NLP results...")
    # quantized_nlp_df = aggregate_results(quantized_results_dir, "results_*.json", parse_nlp_metrics_json) # Adapt parser if format differs
    # combined_quantized_df = pd.concat([quantized_sec_df, quantized_nlp_df], ignore_index=True)
    # combined_all_df = pd.concat([baseline_all_df, combined_quantized_df], ignore_index=True) # Final combined df

    # For now, work only with baseline
    final_df = baseline_all_df


    # --- Save Final Combined DataFrame ---
    if not final_df.empty:
        print("\n--- Aggregated Baseline DataFrame (Security + NLP + Size) ---")
        # Display relevant columns for NLP check
        print(final_df[final_df['benchmark_suite'] == 'NLP'][['model_name', 'benchmark_name', 'num_fewshot','score_metric', 'score_value', 'score_stderr']].head())
        print(f"\nDataFrame shape: {final_df.shape}")
        # Check size calculation
        print("\nModel Sizes Calculated (GB):")
        print(final_df[['model_name', 'model_size_disk_gb']].drop_duplicates().to_string(index=False))

        print("\nSaving aggregated data...")

        try:
            final_df.to_csv(output_csv_file, index=False)
            print(f"Saved CSV to {output_csv_file}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

        try:
            final_df_json = final_df.where(pd.notnull(final_df), None)
            final_df_json.to_json(output_json_file, orient='records', indent=2, date_format='iso')
            print(f"Saved JSON to {output_json_file}")
        except Exception as e:
            print(f"Error saving JSON: {e}")

    else:
        print("No baseline data aggregated.")


    # TODO: Add aggregation for quantized results (mod directories) - Requires parsing logic for ONNX eval output
    # TODO: Add calculation for model disk size (requires iterating model dirs and using os.path.getsize)
    # TODO: Add merging/parsing of scores from security _outputs.json or manual eval files (major task)

