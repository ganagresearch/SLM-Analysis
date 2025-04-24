import glob
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def check_and_delete_metric_file(file_path):
    """Checks if a metric file has a valid model_path and deletes it if not."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        run_args = data.get("run_args", {})
        model_path = run_args.get("model_path")

        if not model_path: # Check if model_path is missing, None, or empty string
            logging.warning(f"Missing or empty 'run_args.model_path' in {file_path}. Deleting file.")
            try:
                os.remove(file_path)
                logging.info(f"Successfully deleted {file_path}")
                return True # Indicates deletion
            except OSError as e:
                logging.error(f"Error deleting file {file_path}: {e}")
        else:
            # Optional: Add more checks here if needed (e.g., check if path looks plausible)
            pass

    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON in {file_path}. Keeping file (manual check recommended).")
    except Exception as e:
        logging.error(f"Unexpected error processing {file_path}: {e}. Keeping file.")

    return False # Indicates file was kept

if __name__ == "__main__":
    target_dirs = ["results/bare", "results/mod"]
    file_pattern = "*_metrics.json"
    deleted_count = 0
    processed_count = 0

    logging.info(f"Starting cleanup of corrupted metric files in {target_dirs}...")

    for target_dir in target_dirs:
        search_path = os.path.join(target_dir, "**", file_pattern)
        # Use recursive=True as the structure might be nested unexpectedly
        metric_files = glob.glob(search_path, recursive=True)
        logging.info(f"Found {len(metric_files)} potential metric files in {target_dir} subtree.")

        for file_path in metric_files:
             if os.path.isfile(file_path): # Ensure it's a file
                 processed_count += 1
                 if check_and_delete_metric_file(file_path):
                     deleted_count += 1

    logging.info(f"Cleanup finished. Processed {processed_count} files, deleted {deleted_count} corrupted metric files.")
