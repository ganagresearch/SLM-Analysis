import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

# Ensure 'datasets' library is installed: pip install datasets
try:
    from datasets import load_dataset, get_dataset_split_names
except ImportError:
    print("Error: The 'datasets' library is required. Please install it using:")
    print("pip install datasets")
    exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATASET_NAME = "AI4Sec/cti-bench"
# Subsets we need ground truth for (excluding 'ate' for now, requires special handling)
# Also excluding 'rcm-2021' as we treat it like 'rcm' in the eval script
# Excluding 'taa' as it's not handled yet.
SUBSETS_TO_PROCESS = ['cti-mcq', 'cti-rcm', 'cti-vsp']

# --- Helper Function for ATE formatting (if needed later) ---
def format_ate_ground_truth(gt_string: str) -> str:
    """Sorts and joins comma-separated MITRE IDs for canonical comparison."""
    if not gt_string:
        return ""
    ids = [item.strip().upper() for item in gt_string.split(',')]
    # Use set to ensure uniqueness, then sort
    unique_sorted_ids = sorted(list(set(ids)))
    return ",".join(unique_sorted_ids)

# --- Main Processing Function ---

def download_and_format_subset(
    subset_name: str,
    output_dir: Path,
    cache_dir: Optional[str] = None
):
    """Downloads a specific subset and saves its ground truth in JSONL format."""
    output_filename = output_dir / f"{subset_name}_ground_truth.jsonl"
    logging.info(f"Processing subset: {subset_name}")
    logging.info(f"Target output file: {output_filename}")

    try:
        # Check available splits - typically just 'test' for this dataset
        available_splits = get_dataset_split_names(DATASET_NAME, subset_name)
        if 'test' not in available_splits:
            logging.error(f"Could not find 'test' split for subset '{subset_name}'. Available: {available_splits}")
            return False

        # Load the specific subset and split
        logging.info(f"Loading dataset '{DATASET_NAME}', subset '{subset_name}', split 'test'...")
        dataset = load_dataset(DATASET_NAME, subset_name, split='test', cache_dir=cache_dir)
        logging.info(f"Dataset loaded successfully. Number of rows: {len(dataset)}")

        # Check if 'GT' column exists
        if "GT" not in dataset.column_names:
            logging.error(f"Column 'GT' not found in subset '{subset_name}'. Available columns: {dataset.column_names}")
            return False

        # Process and write to JSONL
        processed_count = 0
        with output_filename.open('w', encoding='utf-8') as f:
            for i, row in enumerate(dataset):
                item_id = f"index_{i}" # Assuming item_ids correspond to row index
                ground_truth_raw = row.get("GT")

                if ground_truth_raw is None:
                    logging.warning(f"Row {i} in subset '{subset_name}' has missing 'GT' value. Skipping.")
                    continue

                ground_truth_processed = str(ground_truth_raw).strip()

                # Special handling for ATE if we were processing it
                # if subset_name == 'cti-ate':
                #    ground_truth_processed = format_ate_ground_truth(ground_truth_processed)

                output_record = {
                    "item_id": item_id,
                    "ground_truth": ground_truth_processed
                }
                f.write(json.dumps(output_record) + '\n')
                processed_count += 1

        logging.info(f"Successfully processed {processed_count} records for '{subset_name}' and saved to {output_filename}")
        return True

    except Exception as e:
        logging.error(f"Error processing subset '{subset_name}': {e}", exc_info=True)
        return False

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description=f"Download and format ground truth data for specific CTIBench subsets from {DATASET_NAME}.")
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where the ground truth JSONL files will be saved."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory to specify for Hugging Face datasets cache."
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level."
    )

    args = parser.parse_args()

    # Configure Logging
    logging.getLogger().setLevel(args.log_level.upper())

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {args.output_dir.resolve()}")

    success_count = 0
    fail_count = 0
    for subset in SUBSETS_TO_PROCESS:
        if download_and_format_subset(subset, args.output_dir, args.cache_dir):
            success_count += 1
        else:
            fail_count += 1

    logging.info(f"--- Processing Complete ---")
    logging.info(f"Successfully processed: {success_count} subsets.")
    logging.info(f"Failed to process: {fail_count} subsets.")

    if fail_count > 0:
        logging.warning("Some subsets failed to process. Check logs for details.")
        exit(1)
    else:
        logging.info("All specified subsets processed successfully.")

if __name__ == "__main__":
    main() 
