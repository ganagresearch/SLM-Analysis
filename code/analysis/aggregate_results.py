import pandas as pd
import json
import glob
import os
import re
from datetime import datetime
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Mapping from model directory base names to parameter counts
MODEL_PARAMS = {
    "Mistral-7B-Instruct-v0.3": "7B",
    "phi-3-mini-4k-instruct": "3.8B",
    "Phi-3-mini-4k-instruct": "3.8B", # Handle variations
    "TinyLlama-1.1B-Chat-v1.0": "1.1B",
    "electra-small-discriminator": "14M",
    # Handle quantized path variations if they differ significantly
    # Adding entries based on file names in results/mod
    "Mistral-7B-Instruct-v0.3-GPTQ": "7B",
    "Phi-3-mini-4k-instruct-GPTQ": "3.8B",
    "TinyLlama-1.1B-Chat-v1.0-GPTQ": "1.1B",
}

# Mapping for quantization hardware (using base names)
QUANT_HARDWARE = {
    "Mistral-7B-Instruct-v0.3": "H200",
    "phi-3-mini-4k-instruct": "H200",
    "Phi-3-mini-4k-instruct": "H200",
    "TinyLlama-1.1B-Chat-v1.0": "A40",
}

# Define columns for the final DataFrame
DATAFRAME_COLUMNS = [
    "model_name",
    "parameters",
    "quantization_status", # 'baseline', 'AutoGPTQ-4bit'
    "benchmark_suite", # 'Security', 'NLP'
    "benchmark_name", # e.g., 'arc_challenge', 'sevenllm_bench', 'ctibench'
    "task_subset", # Primarily for ctibench, 'full' for others if applicable
    "score_metric", # e.g., 'acc_norm', 'ROUGE-L-F1', 'Accuracy', 'MAE', 'Reject Rate', 'Benign Rate'
    "score_value",
    "score_stderr", # Standard error if available (NLP)
    "model_size_disk_gb", # Calculated
    "peak_vram_gpu_mb",
    "peak_ram_system_mb",
    "load_time_seconds",
    "inference_speed_tok_per_sec",
    "avg_inference_time_per_sample_ms",
    "num_fewshot", # Added from lm-eval config (0 for others)
    "eval_hardware", # A40
    "quant_hardware", # A40 or H200 or None
    "evaluation_script", # 'evaluate_cli.py', 'lm-evaluation-harness', 'calculate_scores.py' (or specific eval script)
    "timestamp", # Timestamp of the original metrics file run
    "model_path", # Path used for the run
    "metrics_json_path", # Path to the source *_metrics.json
    "outputs_json_path", # Path to the corresponding *_outputs.json (used for linking scores)
    "score_source_file", # Path to the file where the score was calculated (e.g., from results/evaluation/)
]


# --- Helper Functions ---

def parse_model_name(path_or_filename):
    """Extracts the base model name from a path or filename, handling quantization suffixes."""
    if not path_or_filename: return "Unknown"

    # If it's a path, get the basename first
    if os.path.sep in path_or_filename:
        path_or_filename = os.path.normpath(path_or_filename)
        base_name = os.path.basename(path_or_filename)

        # Check if it's a known quantization subdir
        quant_suffixes = ["quantized-gptq-4bit"] # Add more if needed
        if base_name in quant_suffixes:
            parent_dir = os.path.dirname(path_or_filename)
            model_base_name = os.path.basename(parent_dir)
            if model_base_name == "Phi-3-mini-4k-instruct": model_base_name = "phi-3-mini-4k-instruct"
            return model_base_name

        # If not quant subdir, the basename is likely the model dir name
        if base_name == "Phi-3-mini-4k-instruct": base_name = "phi-3-mini-4k-instruct"
        return base_name
    else:
        # It's likely already a filename component
        filename_part = path_or_filename

        # Try matching known patterns from filenames (e.g., in results/mod/)
        quant_patterns = {
             "Mistral-7B-Instruct-v0.3-GPTQ": "Mistral-7B-Instruct-v0.3",
             "Phi-3-mini-4k-instruct-GPTQ": "phi-3-mini-4k-instruct",
             "TinyLlama-1.1B-Chat-v1.0-GPTQ": "TinyLlama-1.1B-Chat-v1.0",
        }
        for pattern, base_model in quant_patterns.items():
            if pattern in filename_part:
                return base_model # Return the mapped base model name

        # Handle non-quantized variations
        if "Phi-3-mini-4k-instruct" in filename_part: return "phi-3-mini-4k-instruct"

        # Fallback: check against known model list directly if it's part of filename
        known_models = list(MODEL_PARAMS.keys())
        for known in known_models:
             # Avoid partial matches like 'eval' in 'cyberseceval3'
             # Check if the known model name is a distinct part of the filename
             if f"_{known}_" in f"_{filename_part}_" or filename_part.startswith(known):
                  if known == "Phi-3-mini-4k-instruct": return "phi-3-mini-4k-instruct"
                  return known

        logging.warning(f"Could not reliably parse model name from: {path_or_filename}")
        return "Unknown"


def parse_timestamp_from_filename(filename):
    """Extracts YYYYMMDD-HHMMSS timestamp from standard filenames."""
    if not filename: return None
    # Try specific lm-eval timestamp format first
    match_lm_eval = re.search(r'results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})\.\d+\.json', filename)
    if match_lm_eval:
        try:
            ts_str = match_lm_eval.group(1).replace('T', ' ')
            return datetime.strptime(ts_str, '%Y-%m-%d %H-%M-%S')
        except ValueError: pass

    # Try security benchmark format (YYYYMMDD-HHMMSS) in filename
    # Make regex more specific to avoid matching parts of model names if needed
    match_sec = re.search(r'(\d{8}-\d{6})', filename)
    if match_sec:
        try:
            return datetime.strptime(match_sec.group(1), '%Y%m%d-%H%M%S')
        except ValueError: pass

    # Fallback: Try to get timestamp from file modification time of the file itself
    try:
        return datetime.fromtimestamp(os.path.getmtime(filename))
    except Exception:
        logging.warning(f"Could not determine timestamp for file: {filename}")
        return None

def get_model_name_from_nlp_path(file_path):
    """Estimates model name from the lm-eval output path structure."""
    # Path structure: ... / <model_name> / <task_name> / results.json / ... / results_....json
    parts = os.path.normpath(file_path).split(os.sep)
    try:
        # Find the index of 'results.json'
        results_json_index = parts.index('results.json')
        # Model name should be 2 levels above 'results.json'
        if results_json_index >= 2:
            model_name = parts[results_json_index - 2]
            # Handle variations like Phi-3 vs phi-3
            if model_name == "Phi-3-mini-4k-instruct": model_name = "phi-3-mini-4k-instruct"
            return model_name
    except (ValueError, IndexError):
         # Try alternative: model name might be 4 levels up if structure is deeper
         try:
             results_json_index = parts.index('results.json')
             if results_json_index >= 4:
                 model_name = parts[results_json_index - 4] # Check this level
                 if model_name in MODEL_PARAMS or model_name == "Phi-3-mini-4k-instruct":
                     if model_name == "Phi-3-mini-4k-instruct": model_name = "phi-3-mini-4k-instruct"
                     return model_name
         except (ValueError, IndexError):
            pass # Fall through if still not found

    logging.warning(f"Could not determine model name from path structure: {file_path}")
    return "Unknown"


def get_model_path_from_nlp_config(config):
    """Extracts model path from lm-eval config if present."""
    model_args = config.get("model_args", "")
    match = re.search(r'pretrained=([^,]+)', model_args)
    if match:
        path = match.group(1)
        # Clean potential extra quotes or spaces
        return path.strip().strip("'\"")
    # Check model_name field as fallback
    path = config.get("model_name")
    if path: return path.strip().strip("'\"")
    return None


def get_dir_size(start_path):
    """Calculates the total size of a directory in Gigabytes (GB)."""
    if not start_path or not os.path.isdir(start_path):
        # Try going up one level if the path itself doesn't exist (e.g., from NLP config)
        parent_path = os.path.dirname(start_path)
        if parent_path and os.path.isdir(parent_path):
             logging.info(f"Path '{start_path}' not found, trying parent '{parent_path}' for size calculation.")
             start_path = parent_path
        else:
            logging.warning(f"Cannot calculate size: Path '{start_path}' is not a valid directory.")
            return None

    total_size_bytes = 0
    try:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip symbolic links
                if not os.path.islink(fp):
                    try:
                        total_size_bytes += os.path.getsize(fp)
                    except OSError as e:
                        logging.warning(f"Could not get size of file {fp}: {e}")
    except OSError as e:
        logging.error(f"Error walking directory {start_path}: {e}")
        return None
    # Convert bytes to Gigabytes (using 1024^3)
    return round(total_size_bytes / (1024**3), 3)


# --- Parsing Functions ---

def parse_security_metrics_json(file_path, quantization_type):
    """Parses a single *_metrics.json file from evaluate_cli.py runs."""
    data = {}
    try:
        with open(file_path, 'r') as f:
            metrics = json.load(f)

        run_args = metrics.get("run_args", {})
        load_metrics = metrics.get("load_metrics", {})
        inference_metrics = metrics.get("inference_metrics", {})
        outputs_file = metrics.get("outputs_file", None) # Get the outputs file path

        model_path = run_args.get("model_path")
        data["model_path"] = model_path

        # Use filename parsing as fallback if model_path is missing/unhelpful
        filename_model_name = parse_model_name(os.path.basename(file_path))
        model_base_name = parse_model_name(model_path if model_path else "")
        if not model_base_name or model_base_name == "Unknown":
             model_base_name = filename_model_name # Use filename parsed name
             if model_base_name == "Unknown":
                 logging.warning(f"Could not parse model name from path '{model_path}' or filename in {file_path}")

        # --- Extract Fields ---
        data["model_name"] = model_base_name
        # Try looking up params using potentially quantized name first, then base name
        data["parameters"] = MODEL_PARAMS.get(os.path.basename(model_path) if model_path else filename_model_name,
                                              MODEL_PARAMS.get(model_base_name, "Unknown"))
        data["quantization_status"] = quantization_type
        data["benchmark_suite"] = "Security"
        data["benchmark_name"] = run_args.get("benchmark_name", "Unknown")
        data["task_subset"] = run_args.get("cti_subset") # Will be None for non-ctibench
        if data["task_subset"] is None and data["benchmark_name"] != "ctibench":
             data["task_subset"] = "full" # Assign 'full' to cybersec & sevenllm

        # Resource Usage & Speed
        data["peak_vram_gpu_mb"] = inference_metrics.get("pytorch_vram_peak_inference_mb")
        # Try system VRAM as fallback, check key name correctness
        if data["peak_vram_gpu_mb"] is None:
             data["peak_vram_gpu_mb"] = inference_metrics.get("system_vram_peak_inference_mb")

        data["peak_ram_system_mb"] = inference_metrics.get("ram_peak_inference_mb") # Use peak RAM metric key
        if data["peak_ram_system_mb"] is None: # Fallback to after inference RAM
             data["peak_ram_system_mb"] = inference_metrics.get("ram_after_inference_mb")


        data["load_time_seconds"] = load_metrics.get("load_time_sec")
        data["inference_speed_tok_per_sec"] = inference_metrics.get("tokens_per_second")
        avg_time_sec = inference_metrics.get("avg_generate_time_per_sample_sec")
        data["avg_inference_time_per_sample_ms"] = avg_time_sec * 1000 if avg_time_sec is not None else None

        # Context
        data["eval_hardware"] = "A40"
        data["quant_hardware"] = QUANT_HARDWARE.get(model_base_name) if quantization_type == "AutoGPTQ-4bit" else None
        data["evaluation_script"] = "evaluate_cli.py" # Assuming this script generated these metrics
        data["timestamp"] = parse_timestamp_from_filename(os.path.basename(file_path))
        data["metrics_json_path"] = file_path
        data["outputs_json_path"] = outputs_file # Store path to outputs file for joining scores

        # Placeholders for fields not present in security metrics
        data["score_metric"] = None
        data["score_value"] = None
        data["score_stderr"] = None
        data["num_fewshot"] = 0 # Assuming 0-shot for security script
        data["model_size_disk_gb"] = None # Calculated later
        data["score_source_file"] = None

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
        versions = metrics.get("versions", {}) # Often contains task versions

        if not results:
            logging.warning(f"No 'results' key found in {file_path}. Skipping.")
            return []

        # Get model path from config
        model_path = get_model_path_from_nlp_config(config)

        # Infer model name from path structure
        model_base_name = get_model_name_from_nlp_path(file_path)
        # Fallback using path if structure parsing failed
        if model_base_name == "Unknown" and model_path:
            model_base_name = parse_model_name(model_path) # Use the path parser

        # Iterate through each task reported in the results dict
        for task_name, task_results in results.items():
            if not isinstance(task_results, dict):
                logging.warning(f"Task result for '{task_name}' in {file_path} is not a dict. Skipping.")
                continue

            # Extract few-shot info if available
            num_fewshot = config.get("num_fewshot", 0) # Default to 0
            # Check if n-shot info is available per task (newer lm-eval format)
            # n_shot_info = metrics.get("n-shot", {})
            # num_fewshot = n_shot_info.get(task_name, num_fewshot) # Override if specific value exists

            base_data = {
                "model_name": model_base_name,
                "parameters": MODEL_PARAMS.get(model_base_name, "Unknown"),
                "quantization_status": "baseline", # Only baseline NLP was run
                "benchmark_suite": "NLP",
                "benchmark_name": task_name,
                "task_subset": None, # No subsets for these NLP tasks usually
                "num_fewshot": num_fewshot,
                "eval_hardware": "A40",
                "quant_hardware": None, # Baseline models weren't quantized
                "evaluation_script": "lm-evaluation-harness",
                "timestamp": parse_timestamp_from_filename(os.path.basename(file_path)),
                "model_path": model_path,
                "metrics_json_path": file_path, # Using this as the source 'metrics' file
                "outputs_json_path": None, # No direct equivalent outputs file
                "score_source_file": file_path, # Score is directly in this file
                # Resource metrics not typically available in lm-eval results files
                "peak_vram_gpu_mb": None,
                "peak_ram_system_mb": None,
                "load_time_seconds": None,
                "inference_speed_tok_per_sec": None,
                "avg_inference_time_per_sample_ms": None,
                "model_size_disk_gb": None # Calculated later
            }

            # Create a separate record for each metric reported for the task
            for metric_key, score in task_results.items():
                # Skip metadata-like fields or stderr fields themselves
                if metric_key == "alias" or metric_key.endswith("_stderr") or metric_key == "samples":
                    continue

                metric_name = metric_key.split(',')[0] # Handle potential formatting like 'metric,none'

                # Find corresponding stderr if it exists
                stderr_key_options = [f"{metric_name}_stderr", f"{metric_key}_stderr"]
                stderr_val = None
                for stderr_key in stderr_key_options:
                    if stderr_key in task_results:
                        stderr_val = task_results[stderr_key]
                        break

                record = base_data.copy()
                record["score_metric"] = metric_name
                record["score_value"] = score
                record["score_stderr"] = stderr_val

                # Make sure all columns are present
                filtered_record = {col: record.get(col) for col in DATAFRAME_COLUMNS}
                parsed_records.append(filtered_record)

    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error parsing NLP file {file_path}: {e}")
        return []

    return parsed_records


def parse_calculated_scores_json(file_path):
    """
    Parses score files from results/evaluation/ subdirs.
    Handles *_eval_summary.json (ctibench, sevenllm) and *_scores.json (cyberseceval3).
    """
    parsed_scores = []
    try:
        with open(file_path, 'r') as f:
            # Check if it's jsonl (ctibench/sevenllm details) - skip these, we want summaries
            if file_path.endswith('.jsonl'):
                return []
            scores_data = json.load(f)

        # Check if it's a list (older format?) or dict (newer summaries?)
        if isinstance(scores_data, list):
            # Assume list contains dicts, one per evaluated file (older combined format?)
            entries = scores_data
        elif isinstance(scores_data, dict):
            # Assume dict is the summary itself, wrap in a list for uniform processing
            entries = [scores_data]
        else:
            logging.warning(f"Unexpected data type in {file_path}: {type(scores_data)}. Skipping.")
            return []

        # Process each entry (should usually be just one for summary files)
        for entry in entries:
            if not isinstance(entry, dict):
                logging.warning(f"Skipping non-dict item in {file_path}: {entry}")
                continue

            # --- Key Information for Linking ---
            # Path to the original output file evaluated
            # Check common keys used by different eval scripts
            original_outputs_path = entry.get("outputs_file", entry.get("output_file", entry.get("results_file")))

            if not original_outputs_path:
                logging.warning(f"Skipping entry in {file_path}: Missing 'outputs_file' key or equivalent. Entry: {entry}")
                continue

            # --- Extract Metadata from entry or filename ---
            benchmark_name = entry.get("benchmark_name", "Unknown")
            subset = entry.get("cti_subset") # May be None

            # If metadata missing in file, try parsing from filename
            if benchmark_name == "Unknown":
                 fname = os.path.basename(file_path)
                 if "ctibench" in fname: benchmark_name = "ctibench"
                 elif "sevenllm" in fname: benchmark_name = "sevenllm_bench"
                 elif "cyberseceval3_mitre" in fname: benchmark_name = "cyberseceval3_mitre"

            if subset is None and benchmark_name == "ctibench":
                 fname = os.path.basename(file_path)
                 mcq_match = re.search(r'cti-mcq', fname)
                 rcm_match = re.search(r'cti-rcm', fname)
                 vsp_match = re.search(r'cti-vsp', fname)
                 if mcq_match: subset = "cti-mcq" # Or extract specific like 'threatynews' if available
                 elif rcm_match: subset = "cti-rcm"
                 elif vsp_match: subset = "cti-vsp"

            # --- Extract Scores ---
            # Scores might be in a nested "scores" dict, or top-level
            scores = entry.get("scores", entry) # Default to top level if "scores" key absent

            if not isinstance(scores, dict):
                logging.warning(f"Scores data is not a dict in {file_path} for {original_outputs_path}. Entry: {entry}")
                continue

            extracted_any_score = False
            for metric, value in scores.items():
                # Skip non-score fields that might be at top level
                if metric in ["outputs_file", "output_file", "results_file", "model_name", "benchmark_name", "cti_subset", "details_file", "num_processed", "num_errors"]:
                    continue
                if value is None: continue # Skip null scores

                # Handle nested dicts like ROUGE often returns
                score_value = value
                if isinstance(value, dict):
                    # Try common patterns like .fmeasure or direct value if just one
                    score_value = value.get('fmeasure', list(value.values())[0] if len(value) == 1 and isinstance(list(value.values())[0], (int, float)) else None)

                if score_value is None or not isinstance(score_value, (int, float)):
                    logging.debug(f"Could not extract numeric score for metric '{metric}' in {file_path}. Value: {value}")
                    continue


                score_record = {
                    "outputs_json_path": os.path.normpath(original_outputs_path), # Normalize path for matching
                    "benchmark_name": benchmark_name,
                    "task_subset": subset,
                    "score_metric": metric,
                    "score_value": score_value,
                    "score_source_file": file_path,
                }
                parsed_scores.append(score_record)
                extracted_any_score = True

            if not extracted_any_score:
                 logging.warning(f"Could not extract any valid score metrics from {file_path} for {original_outputs_path}. Scores dict: {scores}")


    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON score file: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error parsing score file {file_path}: {e}")
        return []

    # Only log if scores were actually found
    if parsed_scores:
        logging.info(f"Parsed {len(parsed_scores)} score entries from {file_path}")
    return parsed_scores


# --- Aggregation Functions ---

def aggregate_metrics(results_base_dir, file_pattern, parser_func, quantization_type):
    """Generic function to find and parse metrics files."""
    # Use recursive search as files might be nested unexpectedly or in future runs
    search_pattern = os.path.join(results_base_dir, "**", file_pattern)
    all_metrics_files = glob.glob(search_pattern, recursive=True)
    logging.info(f"Found {len(all_metrics_files)} files matching '{search_pattern}' in {results_base_dir}")

    all_parsed_data = []
    processed_files = 0
    for f in all_metrics_files:
        # Ensure we're only processing files, not directories matching the pattern
        if not os.path.isfile(f):
            logging.debug(f"Skipping directory entry: {f}")
            continue

        # Add specific check to ensure it's a metrics file if pattern is broad
        if file_pattern == "*.json" and not f.endswith("_metrics.json"): # Example if pattern needs refinement
            continue

        parsed = parser_func(f, quantization_type)
        if parsed:
            if isinstance(parsed, list): # Should not happen for metrics parser
                all_parsed_data.extend(parsed)
            else:
                all_parsed_data.append(parsed)
            processed_files += 1

    logging.info(f"Successfully parsed {processed_files} {quantization_type} metrics files.")
    if not all_parsed_data:
        logging.warning(f"No valid {quantization_type} metrics data parsed for pattern '{file_pattern}' in {results_base_dir}.")
        return pd.DataFrame(columns=DATAFRAME_COLUMNS)

    df = pd.DataFrame(all_parsed_data)
    return df


def aggregate_nlp_results(results_base_dir, file_pattern, parser_func):
    """Find and parse lm-eval-harness result files."""
    search_pattern = os.path.join(results_base_dir, "**", file_pattern) # Recursive needed
    all_metrics_files = glob.glob(search_pattern, recursive=True)
    logging.info(f"Found {len(all_metrics_files)} files matching '{search_pattern}' in {results_base_dir}")

    all_parsed_data = []
    processed_files = 0
    for f in all_metrics_files:
        if not os.path.isfile(f): continue
        # Filter out non-result files that might match pattern (e.g. in parent dirs)
        if "results_" not in os.path.basename(f) or not f.endswith(".json"):
             logging.debug(f"Skipping non-results file matching NLP pattern: {f}")
             continue

        parsed = parser_func(f)
        if parsed:
            if isinstance(parsed, list):
                all_parsed_data.extend(parsed)
            else: # Should not happen for NLP parser
                all_parsed_data.append(parsed)
            processed_files += 1

    logging.info(f"Successfully parsed {processed_files} NLP result files.")
    if not all_parsed_data:
        logging.warning(f"No valid NLP data parsed for pattern '{file_pattern}' in {results_base_dir}.")
        return pd.DataFrame(columns=DATAFRAME_COLUMNS)

    df = pd.DataFrame(all_parsed_data)
    return df

def aggregate_scores(scores_dir, parser_func):
    """Find and parse calculated score files recursively."""
    # Define patterns for score files
    score_patterns = ["*_eval_summary.json", "*_scores.json"]
    all_score_files = []
    for pattern in score_patterns:
        search_pattern = os.path.join(scores_dir, "**", pattern)
        found_files = glob.glob(search_pattern, recursive=True)
        logging.info(f"Found {len(found_files)} files matching '{pattern}' in {scores_dir}")
        all_score_files.extend(found_files)

    logging.info(f"Found total {len(all_score_files)} potential score files in {scores_dir}")

    all_parsed_scores = []
    parsed_files_count = 0
    for f in all_score_files:
        if not os.path.isfile(f): continue
        parsed = parser_func(f)
        if parsed: # parser returns a list, could be empty
            all_parsed_scores.extend(parsed)
            if parsed: # Only increment if non-empty list was returned
                parsed_files_count +=1


    if not all_parsed_scores:
         logging.warning(f"No scores parsed from {scores_dir}.")
         return pd.DataFrame() # Return empty df with no columns, merge will handle it

    logging.info(f"Successfully parsed {parsed_files_count} score files yielding {len(all_parsed_scores)} score records.")
    scores_df = pd.DataFrame(all_parsed_scores)

    # Keep only necessary columns for merging + maybe benchmark/subset for diagnostics
    cols_to_keep = ['outputs_json_path', 'score_metric', 'score_value', 'score_source_file', 'benchmark_name', 'task_subset']
    return scores_df[[col for col in cols_to_keep if col in scores_df.columns]]


def finalize_dataframe(df):
    """Cleans up the combined dataframe."""
    # Add any missing columns just in case
    for col in DATAFRAME_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Ensure correct order
    df = df.reindex(columns=DATAFRAME_COLUMNS, fill_value=None)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Normalize outputs_json_path for reliable merging
    if 'outputs_json_path' in df.columns:
         df['outputs_json_path'] = df['outputs_json_path'].apply(lambda x: os.path.normpath(x) if pd.notna(x) else None)

    # Convert numeric columns
    numeric_cols = [
        "peak_vram_gpu_mb", "peak_ram_system_mb", "load_time_seconds",
        "inference_speed_tok_per_sec", "avg_inference_time_per_sample_ms",
        "score_value", "score_stderr", "model_size_disk_gb", "num_fewshot"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort for better readability
    try:
        df = df.sort_values(by=['model_name', 'quantization_status', 'benchmark_suite', 'benchmark_name', 'task_subset', 'timestamp'])
    except Exception as e:
        logging.warning(f"Could not sort DataFrame: {e}. Proceeding without sorting.")


    return df


def calculate_and_add_model_sizes(df):
    """Calculates disk size for unique model paths and merges back into df."""
    if 'model_path' not in df.columns:
        logging.warning("Column 'model_path' not found. Cannot calculate model sizes.")
        return df

    # Ensure model_path is string type before processing
    df['model_path'] = df['model_path'].astype(str)

    unique_paths = df['model_path'].dropna().unique()
    # Filter out placeholder paths like 'None' or empty strings if they sneak in
    unique_paths = [p for p in unique_paths if p and p.lower() != 'none']
    logging.info(f"Found {len(unique_paths)} unique, valid model paths to calculate size for.")

    model_sizes = {}
    missing_paths = []
    for path in unique_paths:
        # Adjust path if it's pointing inside a 'quantized-gptq-4bit' dir for size calc
        effective_path = path
        if os.path.basename(path) == 'quantized-gptq-4bit':
            effective_path = os.path.dirname(path) # Calculate size of parent dir

        size_gb = get_dir_size(effective_path)
        if size_gb is not None:
            model_sizes[path] = size_gb # Store size against original path used in run
        else:
            # Log only paths that were expected to be found
            if os.path.exists(effective_path):
                 logging.warning(f"Could not calculate size for existing path: {path} (tried {effective_path})")
            else:
                 missing_paths.append(path)
                 logging.debug(f"Model path not found for size calculation: {path} (tried {effective_path})")


    if missing_paths:
         logging.warning(f"{len(missing_paths)} model paths were not found on disk for size calculation (e.g., {missing_paths[:3]}{'...' if len(missing_paths) > 3 else ''}). Size will be NaN.")

    df['model_size_disk_gb'] = df['model_path'].map(model_sizes)
    df['model_size_disk_gb'] = pd.to_numeric(df['model_size_disk_gb'], errors='coerce')
    logging.info("Finished calculating and mapping model sizes.")
    return df


# --- Execution Block ---

if __name__ == "__main__":
    # --- Define Paths based on provided tree structure ---
    baseline_sec_metrics_dir = "results/bare" # Removed /48
    quantized_sec_metrics_dir = "results/mod" # Removed /48
    nlp_results_dir = "results/nlp/bare" # Removed /48
    scores_evaluation_dir = "results/evaluation" # Top-level dir for scores
    output_dir = "results/aggregated"
    output_csv_file = os.path.join(output_dir, "all_evaluation_metrics_v2.csv") # New version name
    output_json_file = os.path.join(output_dir, "all_evaluation_metrics_v2.json") # New version name

    os.makedirs(output_dir, exist_ok=True)

    # --- Aggregate Performance Metrics ---
    logging.info("Aggregating baseline security metrics...")
    baseline_sec_metrics_df = aggregate_metrics(
        baseline_sec_metrics_dir, "*_metrics.json", parse_security_metrics_json, "baseline"
    )

    logging.info("Aggregating quantized security metrics...")
    quantized_sec_metrics_df = aggregate_metrics(
        quantized_sec_metrics_dir, "*_metrics.json", parse_security_metrics_json, "AutoGPTQ-4bit"
    )

    # --- Aggregate NLP Results (Metrics + Scores) ---
    logging.info("Aggregating baseline NLP results...")
    baseline_nlp_df = aggregate_nlp_results(
        nlp_results_dir, "results_*.json", parse_nlp_metrics_json
    )

    # --- Combine All Metrics & NLP Results ---
    # NLP df already contains scores, sec_metrics dfs contain resource usage
    all_metrics_df = pd.concat(
        [baseline_sec_metrics_df, quantized_sec_metrics_df, baseline_nlp_df],
        ignore_index=True
    )
    logging.info(f"Combined metrics/NLP DataFrame shape: {all_metrics_df.shape}")

    # --- Aggregate Calculated Security Scores ---
    logging.info("Aggregating calculated security scores...")
    # Search recursively in results/evaluation for score files
    combined_scores_df = aggregate_scores(scores_evaluation_dir, parse_calculated_scores_json)


    if not combined_scores_df.empty:
        logging.info(f"Combined security scores DataFrame shape: {combined_scores_df.shape}")

        # --- Merge Security Scores into Metrics DataFrame ---
        logging.info("Merging security scores into main DataFrame...")

        # Check if the key column exists and is not all null in the metrics DataFrame
        if 'outputs_json_path' not in all_metrics_df.columns:
             logging.error("Cannot merge scores: 'outputs_json_path' missing from metrics DataFrame.")
             final_df = all_metrics_df # Proceed without merge
        elif all_metrics_df['outputs_json_path'].isnull().all():
             logging.warning("Cannot merge scores: 'outputs_json_path' column is all null in metrics DataFrame. Scores will not be added.")
             final_df = all_metrics_df # Proceed without merge
        else:
             # Ensure the key column is normalized in both dataframes before merge
             all_metrics_df['outputs_json_path'] = all_metrics_df['outputs_json_path'].apply(
                 lambda x: os.path.normpath(x) if pd.notna(x) else None
             )
             # Ensure the score df key column is also normalized (already done in parser)
             # combined_scores_df['outputs_json_path'] = combined_scores_df['outputs_json_path'].apply(
             #     lambda x: os.path.normpath(x) if pd.notna(x) else None
             # )

             # Prepare the metrics DataFrame: Drop existing score columns to avoid merge conflicts
             # Keep track of original NLP scores to potentially add back later if merge fails
             original_nlp_scores = all_metrics_df[all_metrics_df['benchmark_suite'] == 'NLP'][['metrics_json_path', 'score_metric', 'score_value', 'score_stderr']].copy()

             cols_to_drop = ['score_metric', 'score_value', 'score_stderr', 'score_source_file']
             metrics_cols_only = all_metrics_df.drop(columns=[col for col in cols_to_drop if col in all_metrics_df.columns])

             # Perform the merge
             # Use left merge to keep all metric rows (from baseline/quantized security runs and NLP runs)
             # Security runs with matching output paths will get score info added
             # NLP runs will initially have NaN for score columns after this merge
             merged_df = pd.merge(
                 metrics_cols_only,
                 combined_scores_df.drop(columns=['benchmark_name', 'task_subset'], errors='ignore'), # Drop potentially duplicate columns from score df
                 on='outputs_json_path',
                 how='left' # Keep all rows from metrics_cols_only
             )
             logging.info(f"Shape after merging security scores: {merged_df.shape}")

             # Add back the original NLP scores
             # Identify NLP rows in the merged dataframe (they won't have outputs_json_path from sec scores)
             # Use metrics_json_path as a key for NLP scores
             if not original_nlp_scores.empty and 'metrics_json_path' in merged_df.columns:
                 # Create a unique key for NLP rows if needed, or iterate
                 nlp_rows_idx = merged_df[merged_df['benchmark_suite'] == 'NLP'].index

                 # Iterate through original NLP scores and update the merged_df
                 # This assumes one metrics_json_path corresponds to potentially multiple score rows
                 for idx, row in original_nlp_scores.iterrows():
                      # Find matching rows in merged_df based on the source metrics file
                      target_rows = merged_df[
                           (merged_df['metrics_json_path'] == row['metrics_json_path']) &
                           (merged_df['score_metric'].isnull()) # Only update if score wasn't added by sec merge
                          ]
                      # Update the first matching row found for this metric (should be unique by task)
                      if not target_rows.empty:
                            target_idx = target_rows.index[0]
                            merged_df.loc[target_idx, 'score_metric'] = row['score_metric']
                            merged_df.loc[target_idx, 'score_value'] = row['score_value']
                            merged_df.loc[target_idx, 'score_stderr'] = row['score_stderr']
                            merged_df.loc[target_idx, 'score_source_file'] = row['metrics_json_path'] # Use source file path

                 logging.info(f"Attempted to re-populate NLP scores.")

             final_df = merged_df # Assign the result to final_df

    else:
        logging.warning("No security score files found or parsed. Using metrics/NLP data only.")
        final_df = all_metrics_df # Use the df without merged scores


    # --- Calculate Model Sizes ---
    if not final_df.empty:
        logging.info("Calculating model disk sizes...")
        final_df = calculate_and_add_model_sizes(final_df)
    else:
        logging.info("Skipping model size calculation as DataFrame is empty.")

    # --- Final Cleanup and Save ---
    if not final_df.empty:
        final_df = finalize_dataframe(final_df)

        print("\n--- Aggregated DataFrame Preview (First 5 rows) ---")
        print(final_df.head().to_string())
        print(f"\nFinal DataFrame shape: {final_df.shape}")

        # Verify score merge - check rows where score_metric is not null
        print("\nSample of merged scores (Security):")
        sec_scores_sample = final_df[(final_df['benchmark_suite'] == 'Security') & (final_df['score_metric'].notna())]
        if not sec_scores_sample.empty:
             print(sec_scores_sample[['model_name', 'quantization_status','benchmark_name', 'task_subset', 'score_metric', 'score_value', 'outputs_json_path', 'score_source_file']].head().to_string())
        else:
             print("No security scores were successfully merged.")

        print("\nSample of NLP scores:")
        nlp_scores_sample = final_df[(final_df['benchmark_suite'] == 'NLP') & (final_df['score_metric'].notna())]
        if not nlp_scores_sample.empty:
            print(nlp_scores_sample[['model_name', 'benchmark_name', 'score_metric', 'score_value', 'score_stderr']].head().to_string())
        else:
            print("No NLP scores found in final DataFrame.")


        # Check size calculation
        print("\nModel Sizes Calculated (GB):")
        size_info = final_df[['model_name', 'quantization_status', 'model_path', 'model_size_disk_gb']].drop_duplicates().sort_values(by=['model_name', 'quantization_status'])
        if not size_info.empty:
            print(size_info.to_string(index=False))
        else:
            print("No model size information available.")

        print("\nSaving aggregated data...")
        try:
            final_df.to_csv(output_csv_file, index=False)
            print(f"Saved CSV to {output_csv_file}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

        try:
            # Convert NaN/NaT to None for JSON compatibility
            final_df_json_ready = final_df.copy()
            # Convert Timestamp to ISO format string for JSON
            if 'timestamp' in final_df_json_ready.columns:
                 final_df_json_ready['timestamp'] = final_df_json_ready['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').fillna('NaT')

            final_df_json_ready = final_df_json_ready.astype(object).where(pd.notnull(final_df_json_ready), None)
            with open(output_json_file, 'w') as f:
                 json.dump(final_df_json_ready.to_dict(orient='records'), f, indent=2)
            # final_df_json_ready.to_json(output_json_file, orient='records', indent=2, date_format='iso') # date_format='iso' might handle NaT better
            print(f"Saved JSON to {output_json_file}")
        except Exception as e:
            print(f"Error saving JSON: {e}")

    else:
        print("No data aggregated.")