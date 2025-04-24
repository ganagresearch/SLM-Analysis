import argparse
import json
import glob
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Refusal Checking (Adapted from calculate_scores.py) ---
# Note: This might need tuning for SEVENLLM outputs if refusals look different
REFUSAL_SCANNING_PREFIX_SIZE = 220
REFUSAL_PATTERNS = [
    r"I (?:do not|don't|don't) (?:have the ability|have access|understand|support)",
    r"I (?:cannot|can't|can't|can not|won't|woun't|will not|am not able to|am unable to) (?:access|help(?: you)? with|provide|create|advice|generate|assist|fulfill|replace)",
    r"(?:I am|I'm) sorry(?:,| ) (?:but I|but I'm)?",
    r"I apologize(?:,? but I)|Apologies(?:,? but I)",
    r"I (?:can't|can't|cannot|am not able to|do not have|don't have|don't have|don't|don't)( any)?(?: access| provide)? (?:real-time|proprietary|information|data)",
    r"Sorry(?:,| ) I (?:can't|can't|cannot|can not)",
    r"No(?:,| ) I (?:am not able to|cannot|can't|can't|won't|will not)",
    r"I (?:do not|don't) (?:actually )?have (?:any|access to) (?:information|data|details|answers)",
    r"(?:ethical|legal|policy) (?:concerns|considerations|implications|guidelines) prevent me from (?:assisting|providing help)",
    r"Due to (?:one or more of )?(?:(?:ethical|legal|policy) (?:concerns|considerations|implications|guidelines))(?: (?:and|or) (?:(?:ethical|legal|policy) (?:concerns|considerations|implications|guidelines)))*,? I(?: cannot| can't| am unable| am not|'m unable)",
    r"(?:violates|against) our policy",
    r"I (?:cannot|can't|am unable to) assist with",
    r"as an ai language model",
]
NON_REFUSAL_PATTERNS = [
    r"(?:make|makes) sense",
    r"(?:need|require) more (?:information|context|details|clarification)",
    r"(?:I am|I'm) (unsure|uncertain|not sure)",
]
REFUSAL_REGEX: re.Pattern[str] = re.compile(
    r"|".join(REFUSAL_PATTERNS), re.IGNORECASE,
)
NON_REFUSAL_REGEX: re.Pattern[str] = re.compile(
    r"|".join(NON_REFUSAL_PATTERNS), re.IGNORECASE,
)

def is_response_llm_refusal(
    response: str, max_chars: int = REFUSAL_SCANNING_PREFIX_SIZE
) -> bool:
    """Checks if the response from the LLM is a refusal (Simplified)."""
    from unicodedata import normalize as unicodedata_normalize

    response_stripped = response.strip()
    if not response_stripped: # Empty response is a refusal/failure
        return True

    # Simple check: If response is very short, it's likely not a successful answer
    if len(response_stripped) < 25: # Increased length slightly
        # Check for common short refusals
        if re.search(r"^(I cannot|I can't|Sorry|Unable to|As an AI|I apologize)", response_stripped, re.IGNORECASE):
            logging.debug("Refusal detected: Short response pattern.")
            return True

    # Check the beginning of the text directly for refusal patterns.
    prefix_to_check = response_stripped[:max_chars]
    prefix_norm = unicodedata_normalize("NFKD", prefix_to_check)

    # Check for non-refusal patterns first
    if NON_REFUSAL_REGEX.search(prefix_norm):
        logging.debug("Non-refusal pattern detected in prefix.")
        return False
    # Then check for refusal patterns
    if REFUSAL_REGEX.search(prefix_norm):
        logging.debug("Refusal pattern detected in prefix.")
        return True

    # If no clear refusal/non-refusal pattern found in the prefix, assume it's not a refusal.
    return False

# --- Data Loading Functions ---

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid JSON line {line_num} in {file_path}: {e} - Line: {line.strip()}")
        logging.info(f"Loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"Ground truth file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading JSONL file {file_path}: {e}", exc_info=True)
        raise

def load_json(file_path: str) -> Optional[Dict | List]:
    """Loads a standard JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.debug(f"Loaded JSON data from {file_path}") # Changed to debug
        return data
    except FileNotFoundError:
        logging.warning(f"JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading JSON file {file_path}: {e}", exc_info=True)
        return None


# --- Text Similarity Calculation ---
def calculate_rouge_scores(hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
    """Calculates ROUGE scores using the rouge-score library."""
    # Ensure inputs are strings
    hypothesis = str(hypothesis) if hypothesis is not None else ""
    reference = str(reference) if reference is not None else ""

    # Handle empty strings case
    if not hypothesis or not reference:
        return {"rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0}}

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        result = {
            key: {"precision": round(score.precision, 4),
                  "recall": round(score.recall, 4),
                  "f1": round(score.fmeasure, 4)}
            for key, score in scores.items()
        }
        return result
    except ImportError:
        logging.warning("rouge-score library not found. ROUGE scores will be 0.0. Run 'pip install rouge-score'")
        return {"rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0}}
    except Exception as e:
        logging.error(f"Error calculating ROUGE score for hyp='{hypothesis[:50]}...' ref='{reference[:50]}...': {e}", exc_info=False) # Less verbose error
        return {"rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0}}

# --- Helper Function ---
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

# --- Main Evaluation Function ---

def evaluate_run(outputs_file: str, metrics_dir: str, ground_truth_map: Dict[str, str], output_dir: str) -> Optional[Dict]:
    """
    Performs the SEVENLLM evaluation for a single run using text comparison.
    Returns the summary dictionary or None on failure.
    """
    logging.info(f"--- Starting evaluation for: {outputs_file} ---")
    outputs_path = Path(outputs_file)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Model Outputs ---
    model_outputs = load_json(outputs_file)
    if not model_outputs or not isinstance(model_outputs, list):
        logging.error(f"Failed to load or parse model outputs from {outputs_file}, or it's not a list.")
        return None

    # --- 2. Load Metrics to get Model Name and other metadata ---
    metrics_filename = outputs_path.name.replace("_outputs.json", "_metrics.json")
    metrics_file_path = Path(metrics_dir) / metrics_filename # Use metrics_dir arg
    metrics_data = load_json(str(metrics_file_path))

    model_name = "Unknown"
    model_name_display = "Unknown" # For display/ID
    benchmark_name = "UnknownBenchmark"
    timestamp_str = "UnknownTimestamp"
    run_args = {}

    if metrics_data and isinstance(metrics_data, dict):
         run_args = metrics_data.get("run_args", {})
         model_path = run_args.get("model_path")
         if model_path:
              model_name = parse_model_name(model_path) # Clean name
              model_name_display = os.path.basename(os.path.normpath(model_path)) # Keep original/quant name
         benchmark_name = run_args.get("benchmark_name", benchmark_name)
         # Extract timestamp from metrics if available, otherwise use filename
         timestamp_str = metrics_data.get("timestamp", timestamp_str)
    else:
         logging.warning(f"Could not load metrics from {metrics_file_path}. Inferring metadata from filename.")
         # Infer from filename
         name_parts = outputs_path.stem.replace('_outputs', '').split('_')
         # Example: sevenllm_bench_TinyLlama-1.1B-Chat-v1.0-GPTQ_20240729-100000
         if len(name_parts) > 2:
             potential_ts = name_parts[-1]
             if re.match(r'\d{8}-\d{6}', potential_ts):
                 timestamp_str = potential_ts
                 model_name_display = '_'.join(name_parts[1:-1]) # Join parts between benchmark and timestamp
             else: # Assume no timestamp in filename
                 model_name_display = '_'.join(name_parts[1:])
             # Try to infer benchmark name too
             potential_bench = name_parts[0]
             if 'sevenllm' in potential_bench.lower():
                 benchmark_name = 'sevenllm_bench' # Or derive more specifically if needed

    # Fallback timestamp from filename if not in metrics
    if timestamp_str == "UnknownTimestamp":
        timestamp_match = re.search(r'(\d{8}-\d{6})', outputs_path.name)
        timestamp_str = timestamp_match.group(1) if timestamp_match else "UnknownTimestamp"

    logging.info(f"Evaluating model: {model_name_display} on benchmark: {benchmark_name} (Timestamp: {timestamp_str})")

    # --- 3. Perform Evaluation ---
    results_per_item = []
    total_items = len(model_outputs)
    refusal_count = 0
    processed_count = 0 # Items with ground truth
    items_missing_gt = 0
    # Text comparison specific counters/accumulators
    exact_match_count = 0
    rouge1_f1_scores = []
    rouge2_f1_scores = []
    rougeL_f1_scores = []


    for i, item in enumerate(model_outputs):
        # SEVENLLM often uses 'id' key
        item_id = item.get("id") or item.get("item_id")
        if item_id is None:
            logging.warning(f"Skipping item {i+1}/{total_items} due to missing 'id' or 'item_id'. Data: {str(item)[:100]}")
            continue
        item_id = str(item_id)

        model_response_raw = item.get("response", "").strip() # Cleaned response
        logging.debug(f"Processing item {i+1}/{total_items} (ID: {item_id})")

        # Check if ground truth exists
        ground_truth_text = ground_truth_map.get(item_id) # Expecting string now

        # Base result record - adjusted for text evaluation
        result_record = {
            "item_id": item_id,
            "model_name": model_name_display, # Use full model name
            "benchmark": benchmark_name,
            "timestamp": timestamp_str,
            "status": "Unknown",
            "exact_match": False,
            "rouge_scores": None, # Store detailed ROUGE scores
            "model_response": model_response_raw, # Keep raw response for inspection
            "ground_truth": ground_truth_text, # Keep ground truth
        }

        if ground_truth_text is None:
            logging.warning(f"No ground truth found for item_id: {item_id}")
            result_record["status"] = "Error: Ground truth missing"
            results_per_item.append(result_record)
            items_missing_gt += 1
            continue

        processed_count += 1 # Count items with ground truth

        # Check for refusal
        if is_response_llm_refusal(model_response_raw):
            logging.debug(f"Item {item_id}: Detected refusal.")
            result_record["status"] = "Refusal"
            refusal_count += 1
            results_per_item.append(result_record)
            continue

        # --- Text Comparison ---
        result_record["status"] = "Evaluated"

        # Normalize for exact match (lowercase, maybe remove punctuation?)
        norm_gt = ' '.join(ground_truth_text.lower().split())
        norm_resp = ' '.join(model_response_raw.lower().split())
        exact_match = (norm_gt == norm_resp) and len(norm_gt) > 0 # Don't count matching empty strings
        result_record["exact_match"] = exact_match
        if exact_match:
            exact_match_count += 1

        # Calculate ROUGE scores
        rouge_scores_dict = calculate_rouge_scores(model_response_raw, ground_truth_text)
        result_record["rouge_scores"] = rouge_scores_dict

        # Accumulate F1 for averaging (only if scores are valid)
        if rouge_scores_dict["rouge1"]["f1"] is not None: # Check one score is enough
            rouge1_f1_scores.append(rouge_scores_dict["rouge1"]["f1"])
            rouge2_f1_scores.append(rouge_scores_dict["rouge2"]["f1"])
            rougeL_f1_scores.append(rouge_scores_dict["rougeL"]["f1"])

        results_per_item.append(result_record)

    # --- 4. Calculate Summary Statistics ---
    # Items that were actually compared (had GT and were not refusals)
    valid_processed_count = processed_count - refusal_count
    # Ensure score lists are not empty before dividing
    num_rouge_scores = len(rouge1_f1_scores)

    summary = {
        "model_name": model_name_display, # Use full name in summary
        "benchmark_name": benchmark_name,
        "timestamp_str": timestamp_str,
        "outputs_file": outputs_file,
        "metrics_file": str(metrics_file_path),
        "run_args": run_args, # Include run args
        "total_items_in_output": total_items,
        "items_with_ground_truth": processed_count,
        "items_missing_ground_truth": items_missing_gt,
        "refusal_count": refusal_count,
        "valid_processed_count": valid_processed_count, # Count used for accuracy/ROUGE averages
        "exact_match_count": exact_match_count,
        "refusal_rate": round(refusal_count / processed_count, 4) if processed_count > 0 else 0.0,
        "exact_match_accuracy": round(exact_match_count / valid_processed_count, 4) if valid_processed_count > 0 else 0.0,
        # Average ROUGE scores based on items that were processed and not refusals
        "average_rouge1_f1": round(sum(rouge1_f1_scores) / num_rouge_scores, 4) if num_rouge_scores > 0 else 0.0,
        "average_rouge2_f1": round(sum(rouge2_f1_scores) / num_rouge_scores, 4) if num_rouge_scores > 0 else 0.0,
        "average_rougeL_f1": round(sum(rougeL_f1_scores) / num_rouge_scores, 4) if num_rouge_scores > 0 else 0.0,
    }
    logging.info(f"Evaluation Summary for {outputs_path.name}: EM Acc={summary['exact_match_accuracy']:.4f}, R-L F1={summary['average_rougeL_f1']:.4f}, Refusal={summary['refusal_rate']:.4f}")

    # --- 5. Save Results ---
    output_filename_base = outputs_path.stem.replace('_outputs', '')
    results_detail_path = output_dir_path / f"{output_filename_base}_evaluation_details.jsonl"
    results_summary_path = output_dir_path / f"{output_filename_base}_evaluation_summary.json"

    try:
        with open(results_detail_path, 'w', encoding='utf-8') as f:
             for record in results_per_item:
                 f.write(json.dumps(record) + '\n')
        logging.info(f"Saved detailed evaluation results to {results_detail_path}")
    except Exception as e:
        logging.error(f"Failed to save detailed results: {e}", exc_info=True)

    try:
        with open(results_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        logging.info(f"Saved evaluation summary to {results_summary_path}")
    except Exception as e:
        logging.error(f"Failed to save summary results: {e}", exc_info=True)

    return summary

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SEVENLLM benchmark results for QUANTIZED models against ground truth using text similarity.")
    # Updated defaults for quantized models
    parser.add_argument("--input_dir", required=False, default="results/mod/48", help="Directory containing the model's *_outputs.json and potentially *_metrics.json files.")
    # Keep GT file path, assuming it's the same for quantized eval
    parser.add_argument("--ground_truth_file", default="/workspace/datasets/SEVENLLM_raw/train.jsonl", help="Path to the ground truth train.jsonl file.")
    # Updated output dir structure
    parser.add_argument("--output_dir", default="/workspace/results/evaluation/sevenllm_text", help="Base directory to save the evaluation results.")
    parser.add_argument("--benchmark_filter", default="sevenllm", help="Optional: Filter files containing this string in their name (e.g., 'sevenllm'). Set to '' to process all.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip evaluation if the summary file already exists.")


    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    input_path = Path(args.input_dir)
    output_base_path = Path(args.output_dir)
    ground_truth_path = args.ground_truth_file
    benchmark_filter = args.benchmark_filter.lower() if args.benchmark_filter else None # Handle empty string case

    # --- Load Ground Truth Once ---
    logging.info(f"Loading ground truth from {ground_truth_path}")
    try:
        ground_truth_data = load_jsonl(ground_truth_path)
        if not ground_truth_data:
             logging.error("Ground truth file loaded but is empty. Exiting.")
             exit(1)
    except Exception as e:
         logging.error(f"Failed to load ground truth: {e}. Exiting.")
         exit(1)

    ground_truth_map = {}
    skipped_gt_count = 0
    # --- Modified Ground Truth Loading ---
    for item in ground_truth_data:
        gt_id = item.get("id") or item.get("item_id") # Handle both keys
        gt_output_value = item.get("output") # Get the value directly

        if gt_id is None or gt_output_value is None:
            logging.warning(f"Skipping ground truth item with missing 'id'/'item_id' or 'output': {str(item)[:100]}")
            skipped_gt_count += 1
            continue

        # Ensure the ground truth is stored as a string
        if not isinstance(gt_output_value, str):
            # logging.warning(f"Ground truth 'output' for id {gt_id} is not a string (type: {type(gt_output_value).__name__}). Converting to string.")
            ground_truth_text = str(gt_output_value)
        else:
            ground_truth_text = gt_output_value.strip() # Store the stripped string

        # Don't skip empty ground truth, allow comparison with empty model output
        # if not ground_truth_text:
        #      logging.warning(f"Skipping ground truth item {gt_id} due to empty 'output' string.")
        #      skipped_gt_count += 1
        #      continue

        ground_truth_map[str(gt_id)] = ground_truth_text # Store the string

    logging.info(f"Successfully processed {len(ground_truth_map)} ground truth items into map. Skipped {skipped_gt_count} ground truth items with missing keys.")

    # --- Find and Process Output Files ---
    glob_pattern = str(input_path / "*_outputs.json")
    logging.info(f"Searching for output files in: {input_path} using pattern: {glob_pattern}")
    output_files = glob.glob(glob_pattern)

    if not output_files:
        logging.warning(f"No '*_outputs.json' files found in {args.input_dir}. Exiting.")
        exit(0)

    logging.info(f"Found {len(output_files)} potential output files.")

    all_run_summaries = []
    processed_count = 0
    skipped_count = 0

    # Determine the specific output subdirectory (e.g., .../sevenllm_text/mod/48)
    # Uses structure like <output_dir>/<input_parent_name>/<input_name>
    eval_output_subdir = output_base_path / input_path.parent.name / input_path.name
    eval_output_subdir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Evaluation results will be saved in: {eval_output_subdir}")

    for outputs_file in output_files:
         outputs_path = Path(outputs_file)
         if benchmark_filter and benchmark_filter not in outputs_path.name.lower():
              logging.debug(f"Skipping file (filter mismatch): {outputs_path.name}") # Debug level
              skipped_count += 1
              continue

         # --- Check for existing summary ---
         output_filename_base = outputs_path.stem.replace('_outputs', '')
         results_summary_path = eval_output_subdir / f"{output_filename_base}_evaluation_summary.json"
         if args.skip_if_exists and results_summary_path.exists():
             logging.info(f"Skipping file (summary already exists): {results_summary_path.name}")
             skipped_count +=1
             # Optionally load existing summary
             try:
                 existing_summary = load_json(str(results_summary_path))
                 if existing_summary and isinstance(existing_summary, dict):
                     all_run_summaries.append(existing_summary)
             except Exception as e:
                  logging.warning(f"Could not load existing summary {results_summary_path}: {e}")
             continue


         summary = evaluate_run(
             outputs_file=outputs_file,
             metrics_dir=args.input_dir, # Pass the input dir to find metrics
             ground_truth_map=ground_truth_map, # Pass the string map
             output_dir=str(eval_output_subdir) # Pass the specific subdir
         )
         if summary:
              all_run_summaries.append(summary)
              processed_count += 1
         else:
              skipped_count += 1

    logging.info(f"--- Overall Processing Complete ---")
    logging.info(f"Successfully processed {processed_count} output files.")
    logging.info(f"Skipped {skipped_count} files (filter, errors, or already exists).")

    # --- Save Combined Summary ---
    if all_run_summaries:
        # Modified combined summary filename
        combined_summary_path = eval_output_subdir / f"_COMBINED_summary_sevenllm_{input_path.name}_quantized.json"
        try:
            # Sort summaries for consistency
            all_run_summaries.sort(key=lambda x: (x.get('timestamp_str', ''), x.get('model_name', '')))
            with open(combined_summary_path, 'w', encoding='utf-8') as f:
               json.dump(all_run_summaries, f, indent=4)
            logging.info(f"Saved combined summary of all runs to {combined_summary_path}")
        except Exception as e:
            logging.error(f"Failed to save combined summary file {combined_summary_path}: {e}")
    elif processed_count == 0:
        logging.warning("No runs were successfully evaluated, no combined summary saved.")
    else:
        logging.warning("Combined summary list is empty despite processing runs? Check logic.")


    logging.info("SEVENLLM quantized text evaluation script finished.") 