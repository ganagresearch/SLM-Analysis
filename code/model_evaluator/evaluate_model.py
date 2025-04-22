# /workspace/code/evaluate_model.py

import argparse
import json
import os
import re # Import regex module
import time
import torch
import psutil
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel # Added AutoModel for non-causal
from tqdm import tqdm
from datasets import load_dataset, load_from_disk # Added datasets import
import datasets # Also import the base namespace for isinstance checks
try:
    import pynvml
    NVML_FOUND = True
except ImportError:
    logging.warning("pynvml not found, GPU system memory usage will not be reported. Run 'pip install nvidia-ml-py' to enable.")
    NVML_FOUND = False

# --- Global NVML Handle ---
# Keep NVML_FOUND global, but handle initialization inside main
nvml_handle = None

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def get_memory_usage():
    """
    Returns current RAM usage (MB), current PyTorch VRAM (MB),
    peak PyTorch VRAM (MB) since last reset, and current System VRAM (MB).
    """
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024) # MB

    pytorch_vram_current = 0
    pytorch_vram_peak = 0
    system_vram_current = 0

    if torch.cuda.is_available():
        pytorch_vram_current = torch.cuda.memory_allocated() / (1024 * 1024) # MB
        pytorch_vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB

        # Use the global NVML_FOUND flag and nvml_handle here
        if NVML_FOUND and nvml_handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                system_vram_current = mem_info.used / (1024 * 1024) # MB
            except pynvml.NVMLError as error:
                logging.warning(f"NVML query failed: {error}. System VRAM usage not available.")
                # Optionally disable further NVML checks
                # global NVML_FOUND
                # NVML_FOUND = False

    return ram_usage, pytorch_vram_current, pytorch_vram_peak, system_vram_current

def load_model_and_tokenizer(model_path, model_type='causal', device='cuda'):
    """
    Loads model and tokenizer, tracking resource usage.
    model_type can be 'causal' or 'encoder'.
    """
    logging.info(f"Loading {model_type} model from: {model_path}...")
    start_time = time.time()
    if torch.cuda.is_available():
        # Reset peak PyTorch memory counter ONLY
        torch.cuda.reset_peak_memory_stats()
        logging.info("Reset peak PyTorch VRAM stats.")

    # Initial memory usage
    # Use the global NVML_FOUND flag which is set before main runs
    initial_ram, _, _, initial_sys_vram = get_memory_usage()
    logging.info(f"Initial RAM usage: {initial_ram:.2f} MB")
    if NVML_FOUND:
        # Note: NVML might not be fully initialized yet if main hasn't run,
        # but get_memory_usage checks nvml_handle which will be None initially.
        # We log initial system VRAM *after* NVML init inside main for accuracy.
        pass # Initial system VRAM logging moved to main after init

    # Load tokenizer
    # Add trust_remote_code=True if needed for models like Phi-3
    try:
        # Use legacy=False for models like Phi-3 if needed
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, legacy=False)
        logging.info("Tokenizer loaded.")
    except Exception as e:
         logging.error(f"Error loading tokenizer: {e}", exc_info=True)
         # NVML shutdown will happen in main's finally block if needed
         return None, None, {}

    # Load model
    # Add trust_remote_code=True, torch_dtype=torch.bfloat16 (or float16) for larger models if needed
    try:
        if model_type == 'causal':
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                # torch_dtype=torch.bfloat16, # Consider adding for faster loading/less VRAM on >7B models
                # low_cpu_mem_usage=True, # Can help for very large models if RAM is tight during load
                device_map=device # Simple single-GPU loading
            )
            logging.info("CausalLM Model loaded.")
        elif model_type == 'encoder':
             model = AutoModel.from_pretrained(
                 model_path,
                 trust_remote_code=True,
                 device_map=device
             )
             logging.info("Encoder Model loaded.")
        else:
            logging.error(f"Unsupported model_type: {model_type}")
            # Attempt to shutdown NVML if initialized
            if NVML_FOUND:
                 try: pynvml.nvmlShutdown()
                 except: pass
            return None, None, {}

        model.eval() # Set to evaluation mode
        logging.info(f"Model placed on device: {device} and set to eval mode.")

    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        # Attempt cleanup if model loading partially succeeded before error
        if 'model' in locals(): del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Attempt to shutdown NVML if initialized
        if NVML_FOUND:
             try: pynvml.nvmlShutdown()
             except: pass
        return None, None, {}

    load_time = time.time() - start_time
    # Get final memory state AFTER loading
    final_ram, pt_vram_after_load, pt_peak_vram_load, sys_vram_after_load = get_memory_usage()

    logging.info(f"Model loaded in {load_time:.2f} seconds.")
    logging.info(f"RAM usage after load: {final_ram:.2f} MB") # No initial RAM needed here
    logging.info(f"PyTorch VRAM Current after load: {pt_vram_after_load:.2f} MB")
    logging.info(f"PyTorch VRAM Peak during load: {pt_peak_vram_load:.2f} MB")
    if NVML_FOUND and nvml_handle: # Check handle is valid now
        logging.info(f"System VRAM Current after load: {sys_vram_after_load:.2f} MB")

    load_metrics = {
        "load_time_sec": load_time,
        # Initial RAM/VRAM moved to main function's responsibility
        "ram_after_load_mb": final_ram,
        "pytorch_vram_current_after_load_mb": pt_vram_after_load,
        "pytorch_vram_peak_load_mb": pt_peak_vram_load,
        "system_vram_current_after_load_mb": sys_vram_after_load if NVML_FOUND else None,
    }
    # We'll add initial memory metrics back into the final dict in main()
    return model, tokenizer, load_metrics

def extract_prompt_from_mutated(mutated_str, item_index):
    """
    Extracts the value of the 'prompt' key from a potentially malformed
    JSON-like string using regex. Handles escaped quotes.
    Returns the extracted prompt string or None if not found/extracted.
    """
    if not isinstance(mutated_str, str):
        return None
    # Regex explanation:
    # "prompt"\s*:\s*      Match literal "prompt", optional whitespace, colon, optional whitespace
    # "                   Match the opening quote of the value
    # (                   Start capturing group 1 (the actual prompt text)
    #   (?:               Start non-capturing group for alternatives
    #      \\"            Match an escaped quote
    #      |              OR
    #      [^"]           Match any character that is NOT a quote
    #   )*                Match the preceding non-capturing group zero or more times
    # )                   End capturing group 1
    # "                   Match the closing quote of the value
    match = re.search(r'"prompt"\s*:\s*"((?:\\"|[^"])*)"', mutated_str, re.IGNORECASE | re.DOTALL) # Ignore case for "prompt"

    if match:
        extracted_text = match.group(1)
        # Basic unescaping
        extracted_text = extracted_text.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
        # Add more unescaping if needed (e.g., \t, \r, etc.)
        return extracted_text
    else:
        logging.warning(f"Item {item_index}: Regex failed to find/extract 'prompt' value from mutated_prompt_str.")
        # logging.debug(f"Failed regex on string: {mutated_str[:200]}...") # Optional: log start of failed string
        return None

def load_benchmark_prompts(benchmark_name, benchmark_path, cti_subset=None, num_samples=None):
    """
    Loads prompts based on benchmark name, respecting num_samples limit during loading.
    Requires cti_subset if benchmark_name is 'ctibench'.
    """
    prompts_data = []
    dataset_cache_dir = os.path.join(os.path.dirname(benchmark_path), '.hf_cache') if benchmark_path else None # Default cache dir near path

    if benchmark_name == "cyberseceval3_mitre":
        logging.info(f"Loading prompts from CyberSecEval3 MITRE file: {benchmark_path}")
        if not os.path.exists(benchmark_path):
             logging.error(f"MITRE prompt file not found at {benchmark_path}")
             return None
        try:
             with open(benchmark_path, 'r', encoding='utf-8') as f:
                 outer_data = json.load(f) # Load the outer list

             # Determine the maximum items to process based on num_samples
             items_to_process = outer_data
             if num_samples is not None and num_samples > 0:
                 items_to_process = outer_data[:num_samples]
                 logging.info(f"Processing the first {num_samples} items from the benchmark file.")
             else:
                 logging.info(f"Processing all {len(outer_data)} items from the benchmark file.")

             # Use tqdm only on the limited set if applicable
             for i, item in enumerate(tqdm(items_to_process, desc="Parsing MITRE prompts")):
                 if not isinstance(item, dict):
                     logging.warning(f"Item {i} is not a dictionary, skipping.")
                     continue

                 mutated_prompt_str = item.get("mutated_prompt")
                 if not mutated_prompt_str:
                     logging.warning(f"Item {i} missing 'mutated_prompt' string, skipping.")
                     continue

                 # Use the robust regex extraction function
                 actual_prompt_text = extract_prompt_from_mutated(mutated_prompt_str, i)

                 if actual_prompt_text:
                     # Try to get a specific ID, default to index if TTP_ID isn't present
                     ttp_mapping = item.get("ttp_id_name_mapping", {})
                     item_id = ttp_mapping.get("TTP_ID") if isinstance(ttp_mapping, dict) else None
                     if item_id is None:
                         item_id = item.get("prompt_id", f"index_{i}") # Fallback to prompt_id or index

                     prompts_data.append({"id": item_id, "prompt": actual_prompt_text})
                 else:
                     # Warning already logged by extract_prompt_from_mutated
                     pass # Skip this item if prompt couldn't be extracted

             logging.info(f"Successfully loaded {len(prompts_data)} actual prompts from {benchmark_name}.")

        except json.JSONDecodeError as e:
            logging.error(f"Error loading/parsing outer JSON {benchmark_path}: {e}", exc_info=True)
            return None
        except Exception as e:
             logging.error(f"Unexpected error during prompt loading for {benchmark_name}: {e}", exc_info=True)
             return None

    elif benchmark_name == "sevenllm_bench":
        logging.info(f"Loading prompts from SEvenLLM dataset directory: {benchmark_path}")
        if not os.path.isdir(benchmark_path):
            logging.error(f"SEvenLLM benchmark path is not a valid directory: {benchmark_path}")
            return None
        try:
            # Assuming benchmark_path is the directory saved by dataset.save_to_disk()
            # Adjust split name if needed (e.g., 'test', 'train') - assuming 'test' for now
            dataset = load_from_disk(benchmark_path)
            logging.info(f"Loaded dataset: {dataset}")

            # If it's a DatasetDict, try to select a default split like 'test' or 'validation'
            if isinstance(dataset, datasets.DatasetDict):
                 split_name = 'test' # Prioritize 'test'
                 if split_name not in dataset:
                     split_name = 'validation' # Fallback to 'validation'
                 if split_name not in dataset:
                     split_name = list(dataset.keys())[0] # Fallback to the first available split
                     logging.warning(f"Could not find 'test' or 'validation' split, using '{split_name}'.")
                 dataset = dataset[split_name]

            if num_samples is not None and num_samples > 0:
                 dataset = dataset.select(range(min(num_samples, len(dataset))))
                 logging.info(f"Selected first {len(dataset)} samples for evaluation.")

            # Adapt based on actual column names in your SEvenLLM dataset
            id_col = 'id' # Adjust if your ID column has a different name
            instruction_col = 'instruction' # Adjust if needed
            input_col = 'input' # Adjust if needed

            if not all(col in dataset.column_names for col in [id_col, instruction_col, input_col]):
                 logging.error(f"Dataset missing required columns. Expected: '{id_col}', '{instruction_col}', '{input_col}'. Found: {dataset.column_names}")
                 return None

            for i, record in enumerate(tqdm(dataset, desc="Parsing SEvenLLM prompts")):
                # Combine instruction and input, handle cases where input might be empty/None
                prompt_text = record[instruction_col]
                if record[input_col] and record[input_col].strip():
                    prompt_text += f"\n{record[input_col]}"
                item_id = record[id_col] if id_col in record and record[id_col] is not None else f"index_{i}"
                prompts_data.append({"id": item_id, "prompt": prompt_text.strip()})

            logging.info(f"Successfully loaded {len(prompts_data)} prompts from {benchmark_name}.")

        except FileNotFoundError:
            logging.error(f"Dataset directory not found or invalid: {benchmark_path}", exc_info=True)
            return None
        except Exception as e:
            logging.error(f"Error loading/processing SEvenLLM dataset from {benchmark_path}: {e}", exc_info=True)
            return None

    elif benchmark_name == "ctibench":
        if not cti_subset:
             logging.error("`--cti-subset` argument is required when `benchmark-name` is 'ctibench'.")
             return None
        logging.info(f"Loading prompts from CTIBench dataset: AI4Sec/cti-bench, subset: {cti_subset}, cache: {dataset_cache_dir}")
        try:
             # Use benchmark_path as cache_dir if provided, otherwise default HF cache
             dataset = load_dataset('AI4Sec/cti-bench', name=cti_subset, cache_dir=dataset_cache_dir)
             logging.info(f"Loaded dataset: {dataset}")

             # CTIBench seems to often use 'train' or 'test' splits. Assume 'test'.
             split_name = 'test'
             if isinstance(dataset, datasets.DatasetDict):
                 if split_name not in dataset:
                     split_name = 'train' # Fallback to train
                 if split_name not in dataset:
                      split_name = list(dataset.keys())[0] # Fallback to first split
                      logging.warning(f"Could not find 'test' or 'train' split in CTIBench subset, using '{split_name}'.")
                 dataset = dataset[split_name]

             if num_samples is not None and num_samples > 0:
                 dataset = dataset.select(range(min(num_samples, len(dataset))))
                 logging.info(f"Selected first {len(dataset)} samples for evaluation.")

             # --- Determine ID and Prompt columns (REQUIRES INSPECTION of CTIBench subsets) ---
             # This is a guess - you MUST verify the actual column names for each subset you use.
             id_column = 'id' # Common but might be different
             prompt_column = 'text' # Common, but could be 'question', 'sentence', 'prompt', etc.
             context_column = 'context' # For RCM tasks
             choices_column = 'choices' # For MCQ tasks

             # Example adaptation logic (needs verification per subset)
             if cti_subset == 'cti-mcq':
                 prompt_column = 'question'
                 id_column = 'idx' # Or maybe 'id'? Verify.
                 if choices_column not in dataset.column_names: logging.warning(f"MCQ subset missing expected '{choices_column}' column.")
             elif cti_subset == 'cti-rcm':
                 prompt_column = 'question'
                 id_column = 'idx' # Or maybe 'id'? Verify.
                 if context_column not in dataset.column_names: logging.warning(f"RCM subset missing expected '{context_column}' column.")
             # Add more elif for other subsets (cti-vsp, cti-taa, cti-ate) based on their structure

             # Check if guessed columns exist
             required_cols = [id_column, prompt_column]
             if cti_subset == 'cti-rcm' and context_column in dataset.column_names: required_cols.append(context_column)
             # Add other required columns based on subset logic

             missing_cols = [col for col in required_cols if col not in dataset.column_names]
             if missing_cols:
                 logging.error(f"CTIBench subset '{cti_subset}' is missing expected columns: {missing_cols}. Found: {dataset.column_names}. Please verify column names.")
                 # Attempt to use a fallback if standard columns are missing?
                 if 'text' in dataset.column_names:
                     logging.warning(f"Falling back to using 'text' column as prompt.")
                     prompt_column = 'text'
                 elif 'sentence' in dataset.column_names:
                     logging.warning(f"Falling back to using 'sentence' column as prompt.")
                     prompt_column = 'sentence'
                 else:
                      logging.error("Cannot determine a suitable prompt column.")
                      return None
                 # Fallback for ID
                 if 'id' not in dataset.column_names and 'idx' not in dataset.column_names:
                      logging.warning("Cannot determine a suitable ID column. Using index.")
                      id_column = None # Use index later

             for i, record in enumerate(tqdm(dataset, desc=f"Parsing CTIBench ({cti_subset}) prompts")):
                 item_id = record.get(id_column, f"index_{i}") if id_column else f"index_{i}"
                 prompt_text = record.get(prompt_column, "")

                 # Specific formatting for certain tasks (examples, adjust as needed)
                 if cti_subset == 'cti-mcq' and choices_column in record:
                     choices_text = "\nChoices:\n" + "\n".join([f"{chr(65+idx)}) {choice}" for idx, choice in enumerate(record.get(choices_column, []))])
                     prompt_text += choices_text
                 elif cti_subset == 'cti-rcm' and context_column in record:
                     prompt_text = f"Context: {record.get(context_column, '')}\n\nQuestion: {prompt_text}"

                 if not prompt_text:
                     logging.warning(f"Skipping record index {i} (ID: {item_id}) due to empty prompt text.")
                     continue

                 prompts_data.append({"id": item_id, "prompt": prompt_text.strip()})

             logging.info(f"Successfully loaded {len(prompts_data)} prompts from {benchmark_name} subset {cti_subset}.")

        except Exception as e:
            logging.error(f"Error loading/processing CTIBench dataset (subset: {cti_subset}): {e}", exc_info=True)
            return None

    else:
        logging.error(f"Benchmark '{benchmark_name}' loading not implemented yet.")
        return None

    if not prompts_data:
         logging.warning(f"No prompts were loaded for benchmark '{benchmark_name}'.")
         # No need to return None here, an empty list is valid but will be caught later

    return prompts_data

def run_inference(model, tokenizer, prompts_data, device='cuda', max_new_tokens=100):
    """Runs inference on the provided prompts_data, collects metrics."""
    results = []
    total_inference_time_sec = 0
    total_tokens_generated = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize() # Ensure previous operations complete
        # Reset peak PyTorch VRAM counter specifically for inference phase
        torch.cuda.reset_peak_memory_stats()
        logging.info("Reset peak PyTorch VRAM stats for inference.")

    # Get memory usage *after* load but *before* inference loop
    ram_before_inference, pt_vram_before_inference, _, sys_vram_before_inference = get_memory_usage()
    logging.info(f"Memory before inference loop - RAM: {ram_before_inference:.2f} MB")
    logging.info(f"Memory before inference loop - PyTorch VRAM: {pt_vram_before_inference:.2f} MB")
    if NVML_FOUND and nvml_handle:
        logging.info(f"Memory before inference loop - System VRAM: {sys_vram_before_inference:.2f} MB")

    logging.info(f"Starting inference on {len(prompts_data)} samples...")
    inference_loop_start_time = time.time() # Overall timer for inference block

    # Ensure the tokenizer has a pad token defined; if not, use eos_token
    if tokenizer.pad_token is None:
        logging.warning("Tokenizer does not have a pad token. Using eos_token as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id # Ensure model config also knows

    # Initialize peak system VRAM for the inference phase
    peak_sys_vram_inference = sys_vram_before_inference if NVML_FOUND else 0

    for item in tqdm(prompts_data, desc="Inference Progress"):
        prompt_text = item.get("prompt")
        item_id = item.get("id", "N/A") # Get ID or use placeholder

        if not prompt_text: # Skip empty prompts
            logging.warning(f"Skipping empty prompt for item id: {item_id}")
            results.append({"item_id": item_id, "prompt": prompt_text, "response": "Skipped empty prompt", "inference_time_sec": 0, "tokens_generated": 0})
            continue

        # --- Apply Chat Template ---
        messages = [{"role": "user", "content": prompt_text}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
             logging.warning(f"Could not apply chat template for item id {item_id}: {e}. Using raw prompt. Model may not respond as expected.")
             formatted_prompt = prompt_text # Fallback to raw prompt

        try:
            # Tokenize the *formatted* prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096 - max_new_tokens)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else None

            if attention_mask is None:
                logging.warning(f"Attention mask not generated by tokenizer for item id {item_id}. Generation might be suboptimal.")

        except Exception as e:
             logging.error(f"Error tokenizing formatted prompt for item id {item_id}: {e}", exc_info=True)
             results.append({"item_id": item_id, "prompt": prompt_text, "response": f"Tokenization Error: {e}", "inference_time_sec": 0, "tokens_generated": 0})
             continue

        iter_start_time = time.time()
        if torch.cuda.is_available(): torch.cuda.synchronize()

        try:
            with torch.no_grad(): # Ensure no gradients are computed
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available(): torch.cuda.synchronize()
            iter_inference_time = time.time() - iter_start_time

            num_input_tokens = input_ids.shape[-1]
            response_ids = outputs[0][num_input_tokens:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            num_generated = len(response_ids)

            if num_generated == 1 and response_ids[0] in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                logging.warning(f"Model only generated EOS/PAD token for item id: {item_id}. Response is empty.")
                response_text = ""

            # Get system VRAM usage *after* this specific generation and update peak
            if NVML_FOUND and nvml_handle:
                _, _, _, current_sys_vram = get_memory_usage()
                peak_sys_vram_inference = max(peak_sys_vram_inference, current_sys_vram)

            results.append({
                "item_id": item_id,
                "prompt": prompt_text,
                "response": response_text,
                "inference_time_sec": iter_inference_time,
                "tokens_generated": num_generated
            })
            total_inference_time_sec += iter_inference_time
            effective_tokens = num_generated
            if num_generated > 0 and response_ids[-1] in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                 effective_tokens = max(0, num_generated -1)
            total_tokens_generated += effective_tokens

        except Exception as e:
            logging.error(f"Error during model generation for item id {item_id}: {e}", exc_info=True)
            results.append({"item_id": item_id, "prompt": prompt_text, "response": f"Generation Error: {e}", "inference_time_sec": time.time() - iter_start_time, "tokens_generated": 0})
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                logging.warning("Potential OOM detected, clearing CUDA cache.")
                torch.cuda.empty_cache()

    inference_loop_end_time = time.time()
    overall_inference_duration_sec = inference_loop_end_time - inference_loop_start_time

    ram_after_inference, pt_vram_after_inference, pt_peak_vram_inference, sys_vram_after_inference = get_memory_usage()

    num_samples_processed = len(prompts_data)
    avg_inference_time_per_sample = total_inference_time_sec / num_samples_processed if num_samples_processed > 0 else 0
    tokens_per_sec = total_tokens_generated / total_inference_time_sec if total_inference_time_sec > 0 else 0

    logging.info(f"\n--- Inference Summary ---")
    logging.info(f"Processed {num_samples_processed} samples.")
    logging.info(f"Total 'generate' time (sum): {total_inference_time_sec:.2f} sec")
    logging.info(f"Overall inference loop duration: {overall_inference_duration_sec:.2f} sec")
    logging.info(f"Average 'generate' time per sample: {avg_inference_time_per_sample:.4f} sec")
    logging.info(f"Total effective tokens generated (excluding final EOS/PAD): {total_tokens_generated}")
    logging.info(f"Overall effective tokens per second: {tokens_per_sec:.2f}")
    logging.info(f"RAM after inference loop: {ram_after_inference:.2f} MB (Delta during inference: {ram_after_inference - ram_before_inference:.2f} MB)")
    logging.info(f"PyTorch VRAM after inference loop: {pt_vram_after_inference:.2f} MB")
    logging.info(f"PyTorch VRAM Peak during inference loop: {pt_peak_vram_inference:.2f} MB (Delta over loaded state: {pt_peak_vram_inference - pt_vram_before_inference:.2f} MB)")
    if NVML_FOUND and nvml_handle:
        logging.info(f"System VRAM after inference loop: {sys_vram_after_inference:.2f} MB")
        logging.info(f"System VRAM Peak approx. during inference loop: {peak_sys_vram_inference:.2f} MB (Delta over loaded state: {peak_sys_vram_inference - sys_vram_before_inference:.2f} MB)")

    inference_metrics = {
        "total_generate_time_sec": total_inference_time_sec,
        "overall_inference_duration_sec": overall_inference_duration_sec,
        "avg_generate_time_per_sample_sec": avg_inference_time_per_sample,
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": tokens_per_sec,
        "ram_before_inference_mb": ram_before_inference,
        "ram_after_inference_mb": ram_after_inference,
        "pytorch_vram_before_inference_mb": pt_vram_before_inference,
        "pytorch_vram_after_inference_mb": pt_vram_after_inference,
        "pytorch_vram_peak_inference_mb": pt_peak_vram_inference,
        "system_vram_before_inference_mb": sys_vram_before_inference if NVML_FOUND else None,
        "system_vram_after_inference_mb": sys_vram_after_inference if NVML_FOUND else None,
        "system_vram_peak_inference_approx_mb": peak_sys_vram_inference if NVML_FOUND else None, # Approximate peak
        "num_samples_run": num_samples_processed
    }

    return results, inference_metrics

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmark evaluation.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=str, required=True, help="Path to the local Hugging Face model directory.")
    parser.add_argument("--model-type", type=str, default="causal", choices=['causal', 'encoder'], help="Type of model architecture.")
    parser.add_argument("--benchmark-name", type=str, required=True, choices=['cyberseceval3_mitre', 'sevenllm_bench', 'ctibench'], help="Name identifier for the benchmark task.")
    parser.add_argument("--benchmark-path", type=str, required=True, help="Path to the benchmark data file (e.g., prompt JSON, dataset dir) or cache directory for HF datasets.")
    parser.add_argument("--results-dir", type=str, default="/workspace/results", help="Directory to save results.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on ('cuda' or 'cpu').")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for model generation.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to load/run (loads/runs all if None or <= 0).")
    parser.add_argument("--cti-subset", type=str, default=None, help="Specify the CTIBench subset name (e.g., 'cti-mcq', 'cti-rcm') if using benchmark-name 'ctibench'.")

    args = parser.parse_args()

    # NVML Initialization moved inside main, after args are parsed
    global nvml_handle, NVML_FOUND # Use global flags/handle
    nvml_initialized_successfully = False
    if NVML_FOUND: # Only proceed if pynvml was imported
        try:
            pynvml.nvmlInit()
            device_id = 0 # Default
            if args.device.startswith('cuda:'):
                try:
                    device_id = int(args.device.split(':')[1])
                except (ValueError, IndexError):
                    logging.warning(f"Could not parse device ID from '{args.device}'. Defaulting to device 0 for NVML.")
            elif args.device != 'cuda':
                NVML_FOUND = False # Disable NVML if not explicitly cuda
                logging.info("Target device is not CUDA, NVML reporting disabled.")

            if NVML_FOUND: # If still enabled after device check
                nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                logging.info(f"Initialized NVML for device {device_id}.")
                nvml_initialized_successfully = True # Flag success

        except pynvml.NVMLError as error:
            logging.warning(f"Failed to initialize NVML: {error}. GPU system memory usage will not be reported.")
            NVML_FOUND = False # Ensure flag is False if init fails
            nvml_handle = None # Ensure handle is None

    initial_ram = 0
    initial_sys_vram = 0
    try: # Wrap core logic in try...finally for NVML shutdown
        # Get initial memory usage *after* potential NVML initialization
        initial_ram, _, _, initial_sys_vram = get_memory_usage()

        # --- Handle num_samples=0 case ---
        num_samples_to_load = args.num_samples
        if num_samples_to_load is not None and num_samples_to_load <= 0:
            num_samples_to_load = None

        # --- Create results directory ---
        os.makedirs(args.results_dir, exist_ok=True)
        model_name_for_file = os.path.basename(args.model_path).replace('/','_')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        subset_tag = f"_{args.cti_subset}" if args.benchmark_name == 'ctibench' and args.cti_subset else ""
        results_filename_base = f"{args.benchmark_name}{subset_tag}_{model_name_for_file}_{timestamp}"
        results_filepath = os.path.join(args.results_dir, f"{results_filename_base}_outputs.json")
        metrics_filepath = os.path.join(args.results_dir, f"{results_filename_base}_metrics.json")

        # --- Log Run Arguments ---
        logging.info("--- Starting Evaluation ---")
        logging.info(f"Model Path: {args.model_path}")
        logging.info(f"Model Type: {args.model_type}")
        logging.info(f"Benchmark Name: {args.benchmark_name}")
        if args.benchmark_name == 'ctibench':
            logging.info(f"CTIBench Subset: {args.cti_subset}")
        logging.info(f"Benchmark Path/Cache Dir: {args.benchmark_path}")
        logging.info(f"Device: {args.device}")
        logging.info(f"Max New Tokens: {args.max_new_tokens}")
        logging.info(f"Num Samples to Load/Run: {'All' if num_samples_to_load is None else num_samples_to_load}")
        logging.info(f"Results Dir: {args.results_dir}")
        logging.info(f"Metrics file: {metrics_filepath}")
        logging.info(f"Outputs file: {results_filepath}")
        logging.info(f"NVML Reporting Enabled: {NVML_FOUND and nvml_initialized_successfully}")
        logging.info(f"Initial RAM usage: {initial_ram:.2f} MB")
        if NVML_FOUND and nvml_initialized_successfully:
            logging.info(f"Initial System VRAM usage: {initial_sys_vram:.2f} MB")

        # --- Validate args ---
        if args.benchmark_name == 'ctibench' and not args.cti_subset:
            logging.error("Missing required argument: --cti-subset must be provided when benchmark-name is 'ctibench'.")
            # No return here, finally block will handle shutdown
            raise ValueError("Missing --cti-subset for ctibench") # Raise error to trigger finally

        # --- Load Model ---
        model, tokenizer, load_metrics_partial = load_model_and_tokenizer(args.model_path, args.model_type, args.device)
        if model is None or tokenizer is None:
            logging.error("Failed to load model/tokenizer. Exiting.")
            raise RuntimeError("Model/Tokenizer loading failed") # Raise error to trigger finally

        # Add initial metrics to load_metrics
        load_metrics = {
            "ram_initial_mb": initial_ram,
            "system_vram_current_initial_mb": initial_sys_vram if NVML_FOUND else None,
            **load_metrics_partial # Combine with metrics from the function
        }
        # Log delta VRAM during load now that we have initial value
        if NVML_FOUND and load_metrics.get("system_vram_current_after_load_mb") is not None:
             sys_vram_delta_load = load_metrics["system_vram_current_after_load_mb"] - initial_sys_vram
             logging.info(f"System VRAM Delta during load: {sys_vram_delta_load:.2f} MB")

        # --- Load Benchmark Prompts ---
        prompts_data = load_benchmark_prompts(args.benchmark_name, args.benchmark_path, args.cti_subset, num_samples_to_load)
        if prompts_data is None:
            logging.error("Failed to load benchmark prompts (returned None). Exiting.")
            # Clean up model/tokenizer before raising
            del model, tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            raise RuntimeError("Prompt loading failed")
        if not prompts_data:
             logging.warning("No prompts were loaded. Check benchmark data and arguments. Exiting.")
             del model, tokenizer
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             # Don't raise error, just exit cleanly if no prompts found
             return # Exit via finally block

        # --- Run Inference ---
        if args.model_type == 'encoder':
            logging.warning(f"Model type is '{args.model_type}'. Skipping generation-based inference loop.")
            inference_results = []
            inference_metrics = {}
        else:
            inference_results, inference_metrics = run_inference(
                model, tokenizer, prompts_data, args.device, args.max_new_tokens
            )

        # --- Combine Metrics ---
        all_metrics = {
            "run_args": vars(args),
            "load_metrics": load_metrics,
            "inference_metrics": inference_metrics
        }

        # --- Save Results ---
        try:
            with open(results_filepath, 'w') as f:
                json.dump(inference_results, f, indent=4)
            logging.info(f"Saved inference outputs ({len(inference_results)} items) to: {results_filepath}")
        except Exception as e:
            logging.error(f"Error saving inference outputs: {e}", exc_info=True)

        try:
            with open(metrics_filepath, 'w') as f:
                json.dump(all_metrics, f, indent=4)
            logging.info(f"Saved evaluation metrics to: {metrics_filepath}")
        except Exception as e:
            logging.error(f"Error saving metrics: {e}", exc_info=True)

        # --- Clean up Model/Tokenizer --- (before finally block)
        logging.info("Cleaning up model and tokenizer from memory...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("CUDA cache cleared.")

    except Exception as e:
         logging.error(f"An error occurred during execution: {e}", exc_info=True)
         # Ensure cleanup happens even if error occurs mid-run

    finally:
        # --- Shutdown NVML ---
        # Ensure NVML is shut down regardless of success or failure, if it was initialized
        if nvml_initialized_successfully:
            try:
                pynvml.nvmlShutdown()
                logging.info("NVML shut down.")
            except pynvml.NVMLError as error:
                logging.warning(f"Failed to shutdown NVML: {error}")

    logging.info("--- Evaluation Complete ---")


if __name__ == "__main__":
    main()
