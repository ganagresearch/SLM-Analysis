import argparse
import json
import logging
import re
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Loading Functions ---

def load_json_outputs(file_path: Path) -> List[Dict[str, Any]]:
    """Loads model output JSON file (must be a list of objects)."""
    try:
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
             raise ValueError("Output file should contain a JSON list.")
        logging.info(f"Loaded {len(data)} model outputs from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e} for file {file_path}")
        raise
    except FileNotFoundError:
        logging.error(f"Model output file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading outputs file {file_path}: {e}", exc_info=True)
        raise

def load_json(file_path: Path) -> Optional[Dict | List]:
    """Loads a standard JSON file (dict or list). Returns None on error."""
    if not file_path.is_file():
        logging.warning(f"JSON file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.debug(f"Loaded JSON data from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading JSON file {file_path}: {e}", exc_info=True)
        return None

def load_ground_truth(file_path: Path) -> Optional[Dict[str, str]]:
    """
    Loads ground truth data (assuming JSONL format: {"item_id": "...", "ground_truth": "..."}).
    Returns a dictionary mapping item_id to ground_truth string or None on error.
    """
    ground_truth_map = {}
    if not file_path.is_file():
        logging.error(f"Ground truth file not found: {file_path}")
        return None
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                        item_id_key = "item_id"
                        gt_key = "ground_truth"
                        item_id = record.get(item_id_key)
                        gt = record.get(gt_key)

                        if item_id is None or gt is None:
                            logging.warning(f"Skipping GT line {line_num} in {file_path}: missing '{item_id_key}' or '{gt_key}'. Line: {line.strip()}")
                            continue
                        ground_truth_map[str(item_id)] = str(gt).strip()
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid JSON line {line_num} in GT file {file_path}: {e} - Line: {line.strip()}")
        logging.info(f"Loaded {len(ground_truth_map)} ground truth records from {file_path}")
        return ground_truth_map
    except Exception as e:
        logging.error(f"Error loading ground truth file {file_path}: {e}", exc_info=True)
        return None


# --- Answer Parsing Functions ---

def parse_mcq_answer(response_text: str) -> Optional[str]:
    """Extracts a single uppercase letter (A, B, C, or D) from the response."""
    response_text = response_text.strip()
    # Prioritize single-letter answers
    if len(response_text) == 1 and response_text.upper() in ['A', 'B', 'C', 'D']:
        return response_text.upper()
    # Check if it starts or ends with the letter maybe followed by punctuation
    match_start = re.match(r'^([A-D])[^A-Za-z0-9]*', response_text, re.IGNORECASE)
    if match_start:
        return match_start.group(1).upper()
    match_end = re.search(r'([A-D])\s*$', response_text, re.IGNORECASE)
    if match_end:
        return match_end.group(1).upper()
    # Fallback: Check if the entire short string is just the letter
    if len(response_text) < 5 and re.match(r'^[A-D]$', response_text, re.IGNORECASE):
         return response_text.upper()

    # Added check for "Answer: X" patterns
    match_answer = re.search(r'(?:Answer|Choice|Option):\s*([A-D])', response_text, re.IGNORECASE)
    if match_answer:
        return match_answer.group(1).upper()

    logging.debug(f"Could not parse MCQ answer: {response_text[:50]}")
    return None

def parse_rcm_answer(response_text: str) -> Optional[str]:
    """Extracts a CWE ID (e.g., CWE-123 or CWE ID: CWE-123) from the response, often on the last line."""
    # Look for "CWE ID: CWE-XXX" or just "CWE-XXX" at the end of the string or line
    match = re.search(r'(?:CWE ID:\s*)?(CWE-\d+)\s*$', response_text.strip(), re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Fallback: Find the last occurrence of CWE-XXX anywhere
    matches = re.findall(r'(CWE-\d+)', response_text, re.IGNORECASE)
    if matches:
        return matches[-1].upper() # Return the last one found

    logging.debug(f"Could not parse RCM (CWE) answer: {response_text[:50]}")
    return None

def parse_vsp_answer(response_text: str) -> Optional[str]:
    """Extracts a CVSS v3.1 vector string from the response."""
    # Regex to find CVSS string, potentially ignoring surrounding text or markdown
    # Looks for CVSS:3.1/ followed by metric pairs like AV:N/AC:L/...
    match = re.search(r'(CVSS:3\.1/((?:[A-Z]{1,3}:[NALPHUC]{1}/?)+))', response_text.strip(), re.IGNORECASE)
    if match:
        # Return the full matched vector string e.g., CVSS:3.1/AV:N/AC:L/...
        # Normalize casing just in case: CVSS:3.1/AV:N/AC:L...
        vector_part = match.group(2).upper()
        # Ensure no trailing slash
        if vector_part.endswith('/'):
             vector_part = vector_part[:-1]
        return f"CVSS:3.1/{vector_part}"

    logging.debug(f"Could not parse VSP (CVSS) answer: {response_text[:50]}")
    return None

# Skipping ATE as per previous logic
# def parse_ate_answer(response_text: str) -> Optional[str]:
#     """
#     Extracts a comma-separated list of MITRE ATT&CK technique IDs (TXXXX or TXXXX.XXX).
#     Returns a canonical, sorted string representation for comparison.
#     """
#     # Find all Txxxx or Txxxx.xxx patterns
#     ids = re.findall(r'(T\d{4}(?:\.\d{3})?)', response_text, re.IGNORECASE)
#     if not ids:
#         logging.debug(f"Could not parse ATE (MITRE ID) answer: {response_text[:50]}")
#         return None
#     # Normalize (uppercase) and sort for consistent comparison
#     normalized_sorted_ids = sorted(list(set([id.upper() for id in ids]))) # Use set to remove duplicates before sorting
#     return ",".join(normalized_sorted_ids)

# --- Helper Functions ---
def parse_model_name(model_path: Optional[str]) -> str:
    """Helper function to extract a clean model name from a path."""
    if not model_path: return "Unknown"
    # Normalize path separators and get the last part
    norm_path = os.path.normpath(model_path)
    base_name = os.path.basename(norm_path)
    # Remove common suffixes like '-hf', '_local', '-GPTQ' etc.
    # Add more suffixes as needed (e.g., '-AWQ', '-GGUF')
    base_name = re.sub(r'(-hf|_local|-gptq|-awq|-gguf)$', '', base_name, flags=re.IGNORECASE)
    return base_name

# --- Evaluation Function ---

def evaluate_cti_task(
    model_outputs: List[Dict[str, Any]],
    ground_truth_map: Dict[str, str],
    task_type: str
) -> Dict[str, Any]:
    """Performs evaluation based on the specified task type. Returns summary dict."""
    parsing_function: Optional[Callable[[str], Optional[str]]] = None
    comparison_logic: Callable[[str, str], bool] = lambda pred, gt: pred == gt # Default: exact string match

    # --- Select Parsing and Comparison Logic based on Task ---
    if task_type == 'mcq':
        parsing_function = parse_mcq_answer
        # Ground truth should be single uppercase letter
        comparison_logic = lambda pred, gt: pred.upper() == gt.upper() if pred and gt else False
    elif task_type == 'rcm':
        parsing_function = parse_rcm_answer
        # Ground truth should be uppercase CWE-ID
        comparison_logic = lambda pred, gt: pred.upper() == gt.upper() if pred and gt else False
    elif task_type == 'vsp':
        parsing_function = parse_vsp_answer
        # Case sensitivity might matter for CVSS vectors, use exact match after normalization in parser
        comparison_logic = lambda pred, gt: pred == gt if pred and gt else False
    # Skipping ATE as per previous logic
    # elif task_type == 'ate':
    #     parsing_function = parse_ate_answer
    #     # Comparison assumes GT is also sorted, comma-separated, uppercase
    #     comparison_logic = lambda pred, gt: pred == gt if pred and gt else False
    else:
        # This case should ideally be caught before calling the function
        logging.error(f"Invalid task type passed to evaluate_cti_task: {task_type}")
        raise ValueError(f"Unsupported task type: {task_type}")

    # --- Process Each Item ---
    results_per_item = [] # Store detailed results for saving later
    total_items = len(model_outputs)
    correct_count = 0
    refusal_count = 0 # Placeholder if refusal check added later
    parsing_error_count = 0
    missing_gt_count = 0
    processed_count = 0 # Items with GT and successfully parsed

    for i, item in enumerate(model_outputs):
        item_id = item.get("item_id") or item.get("id") # Handle potential variations in key name
        if item_id is None:
            logging.warning(f"Skipping output item at index {i}: missing 'item_id' or 'id'. Data: {str(item)[:100]}")
            continue
        item_id = str(item_id) # Ensure string comparison

        model_response_raw = item.get("response", "")

        # Store detailed info for jsonl output
        result_record = {
            "item_id": item_id,
            "model_response_raw": model_response_raw,
            "predicted_answer": None,
            "ground_truth": None,
            "is_correct": None,
            "status": "Unknown",
        }

        ground_truth_answer = ground_truth_map.get(item_id)

        if ground_truth_answer is None:
            logging.warning(f"No ground truth found for item_id: {item_id}")
            result_record["status"] = "Error: Ground truth missing"
            missing_gt_count += 1
            results_per_item.append(result_record)
            continue

        result_record["ground_truth"] = ground_truth_answer

        # --- Parse Model Response ---
        predicted_answer = parsing_function(model_response_raw)
        result_record["predicted_answer"] = predicted_answer

        if predicted_answer is None:
            # Only log warning here, debug log inside parsing functions
            logging.warning(f"Could not parse answer for item_id: {item_id}. Raw response: {model_response_raw[:100]}")
            result_record["status"] = "Error: Parsing failed"
            parsing_error_count += 1
            results_per_item.append(result_record)
            continue

        # If we reach here, the item has GT and was parsed
        processed_count += 1
        result_record["status"] = "Evaluated"

        # --- Compare Prediction to Ground Truth ---
        is_correct = comparison_logic(predicted_answer, ground_truth_answer)
        result_record["is_correct"] = is_correct
        if is_correct:
            correct_count += 1
        else:
            logging.debug(f"Incorrect match for item {item_id}: Pred='{predicted_answer}', GT='{ground_truth_answer}'")


        results_per_item.append(result_record) # Add record regardless of correctness

    # --- Calculate Summary Statistics ---
    accuracy = round(correct_count / processed_count, 4) if processed_count > 0 else 0.0

    summary = {
        # Metadata will be added in the main loop
        "task_type": task_type,
        "total_items_in_output": total_items,
        "items_with_ground_truth": total_items - missing_gt_count,
        "items_missing_ground_truth": missing_gt_count,
        "items_refused": refusal_count, # Placeholder
        "items_parsing_error": parsing_error_count,
        "items_processed_for_accuracy": processed_count,
        "correct_matches": correct_count,
        "accuracy": accuracy,
    }
    # Return both summary base and details for saving
    return {"summary": summary, "details": results_per_item}


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate CTIBench benchmark results for QUANTIZED models.")
    # Changed arguments to directories and updated defaults
    parser.add_argument("--input_dir", required=False, default="results/mod/48", type=Path, help="Directory containing model _outputs.json and _metrics.json files.")
    parser.add_argument("--ground_truth_dir", required=False, default="code/analysis/ctibench_ground_truth", type=Path, help="Directory containing ground truth JSONL files (e.g., cti-mcq_ground_truth.jsonl).")
    parser.add_argument("--output_dir", default="results/evaluation/ctibench", type=Path, help="Base directory to save evaluation results.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    parser.add_argument("--benchmark_filter", default="ctibench", help="Optional: Process only files containing this string (case-insensitive). Default 'ctibench'.")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip evaluation if the summary file already exists.")


    args = parser.parse_args()

    # --- Configure Logging ---
    logging.getLogger().setLevel(args.log_level.upper())

    # --- Find Output Files ---
    input_dir_path = Path(args.input_dir)
    glob_pattern = str(input_dir_path / f"*{args.benchmark_filter}*_outputs.json")
    logging.info(f"Searching for output files in: {input_dir_path} using pattern: {glob_pattern}")
    output_files = glob.glob(glob_pattern)

    if not output_files:
        logging.warning(f"No '*_outputs.json' files matching the filter '{args.benchmark_filter}' found in {input_dir_path}. Exiting.")
        exit(0)

    logging.info(f"Found {len(output_files)} potential output files to process.")

    # --- Ensure output directory exists ---
    # Create a subdirectory reflecting the input dir (e.g., results/evaluation/ctibench/mod/48)
    eval_output_subdir = args.output_dir / input_dir_path.parent.name / input_dir_path.name # e.g. results/evaluation/ctibench/mod/48
    eval_output_subdir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Evaluation results will be saved in: {eval_output_subdir}")

    all_run_summaries = []
    processed_count = 0
    skipped_count = 0

    # --- Process Each Found File ---
    for output_file_path_str in output_files:
        output_file_path = Path(output_file_path_str)
        logging.info(f"--- Processing: {output_file_path.name} ---")

        # --- Infer Task Type and Base Name ---
        # Match ctibench_cti-task_model_... or cti-task_model_... patterns
        match = re.match(r'(?:ctibench_)?cti-(mcq|rcm|vsp|ate)', output_file_path.name, re.IGNORECASE)
        if not match:
            logging.warning(f"Skipping file: Could not determine task type from filename: {output_file_path.name}")
            skipped_count += 1
            continue
        task_type = match.group(1).lower()
        if task_type == 'ate': # Skip tasks not yet handled
             logging.warning(f"Skipping file: Task type 'ate' is not fully supported yet. {output_file_path.name}")
             skipped_count += 1
             continue
        # Add other task types to skip if needed (e.g., taa)

        base_name = output_file_path.stem.replace('_outputs', '')

        # --- Construct Other File Paths ---
        metrics_file_path = input_dir_path / f"{base_name}_metrics.json"
        gt_file_name = f"cti-{task_type}_ground_truth.jsonl" # Assumes GT filenames match task type
        ground_truth_file_path = Path(args.ground_truth_dir) / gt_file_name

        # --- Define Output Paths for this Run ---
        summary_filename = eval_output_subdir / f"{base_name}_eval_summary.json"
        details_filename = eval_output_subdir / f"{base_name}_eval_details.jsonl"

        # --- Skip if Already Processed ---
        if args.skip_if_exists and summary_filename.exists():
            logging.info(f"Skipping file (summary already exists): {summary_filename.name}")
            skipped_count +=1
            # Optionally load existing summary to add to all_run_summaries
            try:
                existing_summary = load_json(summary_filename)
                if existing_summary and isinstance(existing_summary, dict):
                    all_run_summaries.append(existing_summary)
            except Exception as e:
                 logging.warning(f"Could not load existing summary {summary_filename}: {e}")
            continue

        # --- Load Metrics and Extract Metadata ---
        metrics_data = load_json(metrics_file_path)
        model_name = "UnknownModel"
        model_name_display = "UnknownModel"
        timestamp_str = "UnknownTimestamp"
        run_args = {}

        if metrics_data and isinstance(metrics_data, dict):
            run_args = metrics_data.get("run_args", {})
            model_path_in_metrics = run_args.get("model_path")
            if model_path_in_metrics:
                model_name = parse_model_name(model_path_in_metrics) # Clean name
                model_name_display = os.path.basename(os.path.normpath(model_path_in_metrics)) # Keep original name for display/ID
            # Extract timestamp from filename as fallback or primary
            ts_match = re.search(r'(\d{8}-\d{6})', output_file_path.name)
            timestamp_str = ts_match.group(1) if ts_match else timestamp_str
        else:
            logging.warning(f"Could not load or parse metrics file: {metrics_file_path}. Using defaults for metadata.")
            # Try extracting from filename anyway
            ts_match = re.search(r'(\d{8}-\d{6})', output_file_path.name)
            timestamp_str = ts_match.group(1) if ts_match else timestamp_str
            # Try to infer model name from base_name structure
            name_parts = base_name.split('_')
            if len(name_parts) > 2: # e.g., cti-mcq_ModelName_timestamp
                potential_name = name_parts[1]
                # Basic check if it looks like a model name
                if not potential_name.isdigit() and len(potential_name) > 3:
                    model_name_display = potential_name # Use inferred name as display name


        logging.info(f"Model: {model_name_display}, Timestamp: {timestamp_str}, Task: {task_type}")

        # --- Load Outputs and Ground Truth ---
        try:
            model_outputs = load_json_outputs(output_file_path)
        except Exception: # Error logged in function
             skipped_count += 1
             continue # Skip this file if outputs can't be loaded

        ground_truth_map = load_ground_truth(ground_truth_file_path)
        if ground_truth_map is None: # Error logged in function
            logging.error(f"Could not load ground truth for {task_type} from {ground_truth_file_path}. Skipping evaluation for {output_file_path.name}")
            skipped_count += 1
            continue

        # --- Perform Evaluation ---
        logging.info(f"Starting evaluation calculation for {output_file_path.name}")
        try:
            evaluation_results = evaluate_cti_task(
                model_outputs=model_outputs,
                ground_truth_map=ground_truth_map,
                task_type=task_type
            )
            # Augment summary with metadata
            final_summary = {
                "model_name": model_name_display, # Use potentially more specific name from path/metrics
                "timestamp": timestamp_str,
                "input_output_file": str(output_file_path),
                "input_metrics_file": str(metrics_file_path),
                "input_ground_truth_file": str(ground_truth_file_path),
                 "run_args": run_args, # Include run arguments dictionary if available
                 **evaluation_results["summary"] # Add the calculated stats
            }
        except ValueError as e: # Catch specific task type errors
            logging.error(f"Evaluation failed for {output_file_path.name}: {e}")
            skipped_count += 1
            continue
        except Exception as e:
            logging.error(f"Unexpected error during evaluation calculation for {output_file_path.name}: {e}", exc_info=True)
            skipped_count += 1
            continue

        # --- Save Results ---
        logging.info(f"Attempting to save summary to: {summary_filename}")
        try:
            with summary_filename.open('w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=4)
            logging.info(f"Saved evaluation summary successfully.")
            all_run_summaries.append(final_summary) # Add to list for combined summary
            processed_count += 1
        except Exception as e:
            logging.error(f"Failed to save summary results to {summary_filename}: {e}", exc_info=True)
            skipped_count += 1 # Count as skipped if saving fails
            continue # Don't save details if summary failed

        logging.info(f"Attempting to save details to: {details_filename}")
        try:
            with details_filename.open('w', encoding='utf-8') as f:
                for record in evaluation_results["details"]:
                    f.write(json.dumps(record) + '\n')
            logging.info(f"Saved detailed evaluation results successfully.")
        except Exception as e:
            logging.error(f"Failed to save detailed results to {details_filename}: {e}", exc_info=True)
            # Don't increment skipped_count again, just log error


    # --- Save Combined Summary ---
    if all_run_summaries:
        # Modified combined summary filename
        combined_summary_path = eval_output_subdir / f"_COMBINED_summary_ctibench_{input_dir_path.name}_quantized.json"
        logging.info(f"Attempting to save combined summary of {len(all_run_summaries)} runs to {combined_summary_path}")
        try:
            # Sort summaries for consistency, e.g., by timestamp then model name
            all_run_summaries.sort(key=lambda x: (x.get('timestamp', ''), x.get('model_name', '')))
            with open(combined_summary_path, 'w', encoding='utf-8') as f:
               json.dump(all_run_summaries, f, indent=4)
            logging.info(f"Saved combined summary successfully.")
        except Exception as e:
            logging.error(f"Failed to save combined summary file {combined_summary_path}: {e}", exc_info=True)
    elif processed_count == 0:
         logging.warning("No runs were successfully processed, no combined summary saved.")
    else:
         logging.warning("Combined summary list is empty despite processing runs? Check logic.")


    logging.info(f"--- Overall Processing Complete ---")
    logging.info(f"Successfully processed and saved results for {processed_count} files.")
    logging.info(f"Skipped {skipped_count} files (filter mismatch, missing files, errors, or already exists).")
    logging.info(f"Evaluation results saved in: {eval_output_subdir}")


if __name__ == "__main__":
    # Example: python code/analysis/evaluate_ctibench_quantized.py --input_dir results/mod/48 --output_dir results/evaluation/ctibench_mod
    main() 