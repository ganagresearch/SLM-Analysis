# /workspace/code/load_sevenllm.py

import json
import os
import sys
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Value

def load_jsonl(file_path, max_records=None):
    """
    Loads data from a JSONL file line by line, serializing dicts in 'output'.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return []

    data = []
    print(f"Loading records from {os.path.basename(file_path)} (JSONL)...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=f"Processing {os.path.basename(file_path)}")):
                if max_records is not None and i >= max_records:
                    if i == max_records:
                        print(f"\nReached max_records limit ({max_records}) for {os.path.basename(file_path)}.")
                    break
                try:
                    # Load the record
                    record = json.loads(line.strip())
                    # Ensure output is treated as a string (serialize if it's a dict)
                    if 'output' in record and isinstance(record['output'], dict):
                         record['output'] = json.dumps(record['output']) # Serialize dict to JSON string
                    data.append(record)
                except json.JSONDecodeError as e:
                    print(f"\nWarning: Skipping invalid JSON line {i+1} in {os.path.basename(file_path)}: {line.strip()} - Error: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"\nWarning: Error processing line {i+1} in {os.path.basename(file_path)}: {line.strip()} - Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error opening or reading file {file_path}: {e}", file=sys.stderr)
        return []
    print(f"Successfully loaded {len(data)} records from {os.path.basename(file_path)}.")
    return data

def load_json(file_path):
    """Loads data from a standard JSON file (expected to be a list of records)."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return []

    data = []
    print(f"Loading records from {os.path.basename(file_path)} (Standard JSON)...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                # Process each record in the list
                for record in tqdm(content, desc=f"Processing {os.path.basename(file_path)}"):
                     # Ensure output is treated as a string (serialize if it's a dict)
                    if 'output' in record and isinstance(record['output'], dict):
                         record['output'] = json.dumps(record['output']) # Serialize dict to JSON string
                    data.append(record)
            else:
                print(f"Error: Expected a JSON list in {file_path}, but got {type(content)}.", file=sys.stderr)
                return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error opening or reading file {file_path}: {e}", file=sys.stderr)
        return []
    print(f"Successfully loaded {len(data)} records from {os.path.basename(file_path)}.")
    return data


# --- Configuration ---
RAW_DATA_DIR = "/workspace/datasets/SEVENLLM_raw"
CALIBRATION_SAVE_DIR = "/workspace/calibration_data/sevenllm_instruct_subset_manual"
# *** Define where to save the full HF DatasetDict ***
FULL_DATASET_SAVE_DIR = "/workspace/datasets/SEVENLLM_instruct_HF"
NUM_CALIBRATION_SAMPLES = 200

# --- Define File Paths ---
train_file = os.path.join(RAW_DATA_DIR, "train.jsonl")
test_file = os.path.join(RAW_DATA_DIR, "test.jsonl") # Corrected extension

# --- Define Expected Dataset Structure (Features) based on README and grep output ---
# Adding 'id' and keeping 'output' as string. Removed source/language.
sevenllm_features = Features({
    'id': Value('int64'),         # Added based on grep output (assuming integer ID)
    'category': Value('string'),
    'instruction': Value('string'),
    'input': Value('string'),
    'thought': Value('string'),
    'output': Value('string'),    # Treat output as string; dicts will be serialized
})

# --- Load Raw Data Lists ---
train_data_list = load_jsonl(train_file)
test_data_list = load_jsonl(test_file) # Use load_jsonl for test.jsonl
calibration_data_list = load_jsonl(train_file, max_records=NUM_CALIBRATION_SAMPLES)

# --- Convert to Hugging Face Datasets using defined Features ---
train_dataset = None
test_dataset = None
calibration_dataset = None

# Helper function to filter dictionaries to match features
def filter_dict_to_features(data_list, features):
    filtered_list = []
    expected_keys = set(features.keys())
    if not data_list:
        return filtered_list
    print(f"Filtering {len(data_list)} records to match features: {expected_keys}")
    for record in tqdm(data_list, desc="Filtering records"):
        # Ensure record is a dictionary
        if not isinstance(record, dict):
            print(f"\nWarning: Skipping non-dict record during filtering: {record}", file=sys.stderr)
            continue

        filtered_record = {}
        for key in expected_keys:
            filtered_record[key] = record.get(key) # Use .get() for safety, defaults to None if key missing

        # Optional: Add type casting/checking here if needed, though defining Features should handle it mostly
        # Example: Ensure 'id' is int if present
        # if 'id' in filtered_record and filtered_record['id'] is not None:
        #     try:
        #         filtered_record['id'] = int(filtered_record['id'])
        #     except (ValueError, TypeError):
        #          print(f"\nWarning: Could not cast id '{record.get('id')}' to int. Setting to None.", file=sys.stderr)
        #          filtered_record['id'] = None # Or handle error as appropriate

        filtered_list.append(filtered_record)
    return filtered_list

if train_data_list:
    try:
        print("\nAttempting conversion of training data with defined Features...")
        filtered_train_list = filter_dict_to_features(train_data_list, sevenllm_features)
        if filtered_train_list:
            train_dataset = Dataset.from_list(filtered_train_list, features=sevenllm_features)
            print("Successfully converted training data to Hugging Face Dataset:")
            print(train_dataset)
        else:
             print("\nWarning: Training data list became empty after filtering.", file=sys.stderr)
    except Exception as e:
        print(f"\nError converting training data list to Dataset: {e}", file=sys.stderr)
        print("Try inspecting records/features definition.", file=sys.stderr)
else:
    print("\nNo training data loaded, skipping Dataset conversion.")


if test_data_list:
     try:
        print("\nAttempting conversion of test data with defined Features...")
        filtered_test_list = filter_dict_to_features(test_data_list, sevenllm_features)
        if filtered_test_list:
            test_dataset = Dataset.from_list(filtered_test_list, features=sevenllm_features)
            print("Successfully converted test data to Hugging Face Dataset:")
            print(test_dataset)
        else:
            print("\nWarning: Test data list became empty after filtering.", file=sys.stderr)
     except Exception as e:
        print(f"\nError converting test data list to Dataset: {e}", file=sys.stderr)
        print("Try inspecting records/features definition.", file=sys.stderr)
else:
    print("\nNo test data loaded, skipping Dataset conversion.")

if calibration_data_list:
    try:
        print("\nAttempting conversion of calibration data with defined Features...")
        filtered_calibration_list = filter_dict_to_features(calibration_data_list, sevenllm_features)
        if filtered_calibration_list:
            calibration_dataset = Dataset.from_list(filtered_calibration_list, features=sevenllm_features)
            print("Successfully converted calibration data to Hugging Face Dataset:")
            print(calibration_dataset)

            # --- Save Calibration Data ---
            try:
                os.makedirs(CALIBRATION_SAVE_DIR, exist_ok=True)
                calibration_dataset.save_to_disk(CALIBRATION_SAVE_DIR)
                print(f"\nCalibration dataset ({len(calibration_data_list)} records) saved successfully to {CALIBRATION_SAVE_DIR}")
            except Exception as e:
                print(f"\nError saving calibration dataset to {CALIBRATION_SAVE_DIR}: {e}", file=sys.stderr)
        else:
             print("\nWarning: Calibration data list became empty after filtering.", file=sys.stderr)

    except Exception as e:
        print(f"\nError converting/saving calibration data: {e}", file=sys.stderr)
        print("Try inspecting records/features definition.", file=sys.stderr)

else:
    print("\nNo calibration data loaded, skipping saving.")


# --- Create and Save DatasetDict ---
dataset_dict_content = {}
if train_dataset:
    dataset_dict_content['train'] = train_dataset
if test_dataset:
     dataset_dict_content['test'] = test_dataset # Usually want the test split for eval

if dataset_dict_content:
    sevenllm_dataset_dict = DatasetDict(dataset_dict_content)
    print("\nCreated DatasetDict:")
    print(sevenllm_dataset_dict)
    # *** Add saving logic here ***
    try:
        os.makedirs(FULL_DATASET_SAVE_DIR, exist_ok=True)
        sevenllm_dataset_dict.save_to_disk(FULL_DATASET_SAVE_DIR)
        print(f"\nFull dataset dictionary saved successfully to {FULL_DATASET_SAVE_DIR}")
    except Exception as e:
        print(f"\nError saving full dataset dictionary to {FULL_DATASET_SAVE_DIR}: {e}", file=sys.stderr)
else:
    sevenllm_dataset_dict = None
    print("\nCould not create DatasetDict as no data was loaded/converted successfully.")


print("\nScript completed.")


