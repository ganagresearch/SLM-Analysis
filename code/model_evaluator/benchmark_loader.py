import json
import os
import logging
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict
from evaluation_utils import extract_prompt_from_mutated # Import necessary util

def load_benchmark_prompts(benchmark_name, benchmark_path, cti_subset=None, num_samples=None):
    """
    Loads prompts based on benchmark name, respecting num_samples limit during loading.
    Requires cti_subset if benchmark_name is 'ctibench'.
    Returns a list of prompts or None on failure.
    """
    prompts_data = []
    dataset_cache_dir = os.path.join(os.path.dirname(benchmark_path), '.hf_cache') if benchmark_path else None

    if benchmark_name == "cyberseceval3_mitre":
        logging.info(f"Loading prompts from CyberSecEval3 MITRE file: {benchmark_path}")
        if not os.path.exists(benchmark_path):
             logging.error(f"MITRE prompt file not found at {benchmark_path}")
             return None
        try:
            with open(benchmark_path, 'r', encoding='utf-8') as f:
                 outer_data = json.load(f)
            items_to_process = outer_data
            if num_samples is not None and num_samples > 0:
                 items_to_process = outer_data[:num_samples]
                 logging.info(f"Processing the first {len(items_to_process)} items.")
            else:
                 logging.info(f"Processing all {len(outer_data)} items.")

            for i, item in enumerate(tqdm(items_to_process, desc="Parsing MITRE prompts")):
                if not isinstance(item, dict): continue
                mutated_prompt_str = item.get("mutated_prompt")
                if not mutated_prompt_str: continue
                actual_prompt_text = extract_prompt_from_mutated(mutated_prompt_str, i)
                if actual_prompt_text:
                     ttp_mapping = item.get("ttp_id_name_mapping", {})
                     item_id = ttp_mapping.get("TTP_ID") if isinstance(ttp_mapping, dict) else None
                     if item_id is None: item_id = item.get("prompt_id", f"index_{i}")
                     prompts_data.append({"id": str(item_id), "prompt": actual_prompt_text}) # Ensure ID is string

            logging.info(f"Successfully loaded {len(prompts_data)} prompts from {benchmark_name}.")

        except Exception as e:
            logging.error(f"Error loading/parsing {benchmark_name}: {e}", exc_info=True)
            return None

    elif benchmark_name == "sevenllm_bench":
        logging.info(f"Loading prompts from SEvenLLM saved dataset directory: {benchmark_path}")
        if not os.path.isdir(benchmark_path):
            logging.error(f"SEvenLLM path is not a valid directory: {benchmark_path}")
            return None
        try:
            # Load the dataset previously saved using save_to_disk
            dataset = load_from_disk(benchmark_path)
            logging.info(f"Loaded dataset: {dataset}")

            # Handle DatasetDict if necessary (though load_from_disk usually loads a single Dataset)
            if isinstance(dataset, DatasetDict):
                 # Use the first split if it's a dict, or adjust logic as needed
                 split_name = list(dataset.keys())[0]
                 logging.warning(f"Loaded a DatasetDict for SEvenLLM, using first split '{split_name}'.")
                 dataset = dataset[split_name]

            if num_samples is not None and num_samples > 0:
                 dataset = dataset.select(range(min(num_samples, len(dataset))))
                 logging.info(f"Selected first {len(dataset)} samples for evaluation.")

            # Adapt based on actual column names in your SEvenLLM saved dataset
            id_col = 'id' # Adjust if your ID column has a different name
            instruction_col = 'instruction' # Adjust if needed
            input_col = 'input' # Adjust if needed

            if not all(col in dataset.column_names for col in [id_col, instruction_col, input_col]):
                 logging.error(f"Saved SEvenLLM dataset missing required columns. Expected: '{id_col}', '{instruction_col}', '{input_col}'. Found: {dataset.column_names}")
                 return None

            for i, record in enumerate(tqdm(dataset, desc="Parsing SEvenLLM prompts")):
                # Combine instruction and input
                prompt_text = record[instruction_col]
                if record[input_col] and record[input_col].strip():
                    prompt_text += f"\n{record[input_col]}"
                item_id = record.get(id_col, f"index_{i}")
                prompts_data.append({"id": str(item_id), "prompt": prompt_text.strip()}) # Ensure ID is string

            logging.info(f"Successfully loaded {len(prompts_data)} prompts from {benchmark_name}.")

        except FileNotFoundError:
            logging.error(f"Saved SEvenLLM dataset directory not found or invalid: {benchmark_path}", exc_info=True)
            return None
        except Exception as e:
            logging.error(f"Error loading/processing saved SEvenLLM dataset from {benchmark_path}: {e}", exc_info=True)
            return None

    elif benchmark_name == "ctibench":
        if not cti_subset:
             logging.error("`--cti-subset` required for 'ctibench'.")
             return None
        logging.info(f"Loading prompts from CTIBench subset: {cti_subset}")
        try:
             dataset = load_dataset('AI4Sec/cti-bench', name=cti_subset, cache_dir=dataset_cache_dir)
             if isinstance(dataset, DatasetDict):
                 split_name = next((s for s in ['test', 'train'] if s in dataset), list(dataset.keys())[0])
                 logging.info(f"Using split '{split_name}' from CTIBench DatasetDict.")
                 dataset = dataset[split_name]

             if num_samples is not None and num_samples > 0:
                 dataset = dataset.select(range(min(num_samples, len(dataset))))
                 logging.info(f"Selected first {len(dataset)} samples.")

             # --- Determine columns based on subset ---
             id_column, prompt_column, context_column, choices_column = None, None, None, None # Initialize to None

             # Using 'Prompt' column directly as it seems pre-formatted in the dataset
             if cti_subset in ['cti-mcq', 'cti-rcm', 'cti-vsp']: # Added cti-vsp
                 prompt_column = 'Prompt'
                 id_column = None # Rely on index fallback for these subsets
                 logging.info(f"Using columns for {cti_subset}: prompt='{prompt_column}', id=index fallback")
             # Add elif blocks for other cti subsets (cti-taa, cti-ate)
             # based on inspecting their columns on Hugging Face Hub
             # Example for cti-ate (assuming 'Prompt' and index ID - VERIFY THIS)
             elif cti_subset == 'cti-ate':
                  prompt_column = 'Prompt'
                  id_column = None
                  logging.info(f"Using columns for {cti_subset}: prompt='{prompt_column}', id=index fallback (VERIFY COLUMNS)")
             # Example for cti-taa (assuming 'Prompt' and index ID - VERIFY THIS)
             elif cti_subset == 'cti-taa':
                   prompt_column = 'Prompt'
                   id_column = None
                   logging.info(f"Using columns for {cti_subset}: prompt='{prompt_column}', id=index fallback (VERIFY COLUMNS)")
             else:
                  logging.warning(f"Column mapping not explicitly defined for CTIBench subset '{cti_subset}'. Attempting default columns 'id', 'text'.")
                  # Attempt default columns or raise error if necessary
                  id_column, prompt_column = 'id', 'text' # Default guess

             # --- Check required columns ---
             required_cols = []
             if id_column: required_cols.append(id_column)
             if prompt_column: required_cols.append(prompt_column)
             # Remove check for choices_column as we use the pre-formatted Prompt
             # if cti_subset == 'cti-mcq' and choices_column: required_cols.append(choices_column)

             missing_cols = [col for col in required_cols if col not in dataset.column_names]
             if missing_cols:
                  logging.error(f"CTIBench subset '{cti_subset}' is missing expected columns: {missing_cols}. Found columns: {dataset.column_names}.")
                  # Attempt fallback if critical columns are missing?
                  if prompt_column not in dataset.column_names and 'text' in dataset.column_names:
                      logging.warning("Falling back to using 'text' as prompt column.")
                      prompt_column = 'text'
                  elif prompt_column not in dataset.column_names:
                       logging.error("Cannot determine a suitable prompt column. Aborting.")
                       return None
                  # Decide if fallback for ID is acceptable (index fallback is default)

             # --- Construct Prompt List ---
             for i, record in enumerate(tqdm(dataset, desc=f"Parsing CTIBench ({cti_subset})")):
                  item_id = record.get(id_column, f"index_{i}") if id_column else f"index_{i}" # Use index if id_column is None or key missing
                  prompt_text = record.get(prompt_column, "") if prompt_column else ""

                  # --- Apply specific formatting ---
                  # REMOVED: No special formatting needed for mcq, rcm, vsp as 'Prompt' column is used directly
                  # if cti_subset == 'cti-mcq' and choices_column and choices_column in record:
                  #     choices_text = "\nChoices:\n" + "\n".join([f"{chr(65+idx)}) {choice}" for idx, choice in enumerate(record.get(choices_column, []))])
                  #     prompt_text += choices_text

                  if not prompt_text:
                      logging.warning(f"Skipping record index {i} (ID: {item_id}) due to empty prompt text.")
                      continue
                  prompts_data.append({"id": str(item_id), "prompt": prompt_text.strip()}) # Ensure ID is string

             logging.info(f"Successfully loaded {len(prompts_data)} prompts from {benchmark_name} subset {cti_subset}.")

        except Exception as e:
            logging.error(f"Error loading/processing {benchmark_name} ({cti_subset}): {e}", exc_info=True)
            return None

    else:
        logging.error(f"Benchmark '{benchmark_name}' loading not implemented.")
        return None

    if not prompts_data:
         logging.warning(f"No prompts were loaded for benchmark '{benchmark_name}'.")

    return prompts_data
