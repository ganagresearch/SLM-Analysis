import time
import torch
import logging
from tqdm import tqdm
from evaluation_utils import get_memory_usage, NVML_FOUND, nvml_handle # Import necessary utils

def run_inference(model, tokenizer, prompts_data, device='cuda', max_new_tokens=100, batch_size=1):
    """
    Runs inference on the provided prompts_data in batches, collects metrics.
    Returns results list and inference_metrics dict.
    """
    results = []
    total_generate_time_sec = 0 # Renamed for clarity: only tracks model.generate time
    total_tokens_generated = 0 # Effective tokens (excluding final special token)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        logging.info("Reset peak PyTorch VRAM stats for inference.")

    # Get memory usage *before* inference loop
    ram_before, pt_vram_before, _, sys_vram_before = get_memory_usage()

    logging.info(f"Starting inference on {len(prompts_data)} samples with batch size {batch_size}...")
    inference_loop_start_time = time.time()

    if tokenizer.pad_token is None:
        logging.warning("Tokenizer missing pad token, setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id # Ensure model config matches

    peak_sys_vram_inference = sys_vram_before if NVML_FOUND and nvml_handle else 0

    # --- Batch Processing Loop ---
    num_batches = (len(prompts_data) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Inference Batches"):
        batch_start_index = i * batch_size
        batch_end_index = min((i + 1) * batch_size, len(prompts_data))
        current_batch_data = prompts_data[batch_start_index:batch_end_index]

        if not current_batch_data: continue # Skip empty batches (shouldn't happen with logic above)

        batch_prompts = [item.get("prompt", "") for item in current_batch_data]
        batch_item_ids = [item.get("id", f"batch_{i}_item_{j}") for j, item in enumerate(current_batch_data)]

        # --- Apply Chat Template (per item in batch) ---
        formatted_prompts = []
        for idx, prompt_text in enumerate(batch_prompts):
            messages = [{"role": "user", "content": prompt_text}]
            try:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                formatted_prompts.append(formatted)
            except Exception as e:
                logging.warning(f"Chat template failed for item {batch_item_ids[idx]}: {e}. Using raw prompt.")
                formatted_prompts.append(prompt_text) # Fallback to raw prompt for this item

        # --- Tokenization (batch) ---
        try:
            # Use padding=True for batching
            inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True, # Enable padding for batching
                truncation=True,
                max_length=4096 - max_new_tokens # Ensure space for generation
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
        except Exception as e:
             logging.error(f"Batch tokenization failed (Batch {i}): {e}", exc_info=True)
             # Add error entries for all items in this failed batch
             batch_iter_start_time = time.time() # Need a time for error reporting
             for item_idx, item_id in enumerate(batch_item_ids):
                 results.append({"item_id": item_id, "prompt": batch_prompts[item_idx], "response": f"Tokenization Error: {e}", "inference_time_sec": 0, "tokens_generated": 0})
             continue # Move to the next batch

        # --- Generation (batch) ---
        batch_iter_start_time = time.time()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        try:
            with torch.no_grad():
                # Generate for the whole batch
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # Add other generation params if needed (e.g., do_sample=False)
                )
            if torch.cuda.is_available(): torch.cuda.synchronize()
            batch_inference_time = time.time() - batch_iter_start_time
            total_generate_time_sec += batch_inference_time # Accumulate total generate time

            # --- Decoding & VRAM Peak Update (per item in batch) ---
            num_input_tokens_batch = input_ids.shape[-1] # Input length is same for all due to padding
            # outputs shape is (batch_size, sequence_length)
            response_ids_batch = outputs[:, num_input_tokens_batch:]

            # Decode each response in the batch
            response_texts = tokenizer.batch_decode(response_ids_batch, skip_special_tokens=True)

            if NVML_FOUND and nvml_handle:
                _, _, _, current_sys_vram = get_memory_usage()
                peak_sys_vram_inference = max(peak_sys_vram_inference, current_sys_vram)

            # --- Store results (per item in batch) ---
            for item_idx, response_text in enumerate(response_texts):
                item_id = batch_item_ids[item_idx]
                prompt_text = batch_prompts[item_idx]
                response_ids = response_ids_batch[item_idx]
                num_generated = len(response_ids)

                # Handle empty generation or only EOS/PAD token
                cleaned_response_text = response_text.strip()
                # This check needs care with batching - decode cleans it mostly
                # if num_generated == 1 and response_ids[0] in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                #      cleaned_response_text = ""

                # Calculate effective tokens (handle potential trailing EOS/PAD)
                effective_tokens = max(0, num_generated -1) if num_generated > 0 and response_ids[-1] in [tokenizer.eos_token_id, tokenizer.pad_token_id] else num_generated
                total_tokens_generated += effective_tokens

                results.append({
                    "item_id": item_id, "prompt": prompt_text, "response": cleaned_response_text,
                    # Assign average batch time per item for simplicity, or 0 if batch failed
                    "inference_time_sec": batch_inference_time / len(current_batch_data) if len(current_batch_data) > 0 else 0,
                    "tokens_generated": num_generated
                })


        except Exception as e:
            logging.error(f"Generation failed for Batch {i}: {e}", exc_info=True)
            batch_inference_time = time.time() - batch_iter_start_time # Time spent before failure
            # Add error entries for all items in this failed batch
            for item_idx, item_id in enumerate(batch_item_ids):
                 results.append({"item_id": item_id, "prompt": batch_prompts[item_idx], "response": f"Generation Error: {e}", "inference_time_sec": batch_inference_time / len(current_batch_data) if len(current_batch_data) > 0 else 0, "tokens_generated": 0})
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                logging.warning("OOM detected, clearing CUDA cache.")
                torch.cuda.empty_cache()
            # Continue to the next batch even if one fails

    # --- Post-Loop Metrics ---
    inference_loop_end_time = time.time()
    overall_inference_duration_sec = inference_loop_end_time - inference_loop_start_time
    ram_after, pt_vram_after, pt_peak_vram_inf, sys_vram_after = get_memory_usage()

    num_samples_processed = len(results) # Count actual results stored (includes errors)
    # Calculate average time based on total *generate* time and number of *successfully processed prompts*
    # This is a bit tricky if some batches fail. Using len(prompts_data) might be more stable.
    num_initial_samples = len(prompts_data)
    avg_time_per_sample = total_generate_time_sec / num_initial_samples if num_initial_samples > 0 else 0
    # Calculate TPS based on total *generate* time
    tps = total_tokens_generated / total_generate_time_sec if total_generate_time_sec > 0 else 0

    inference_metrics = {
        "total_generate_time_sec": total_generate_time_sec, # Sum of model.generate calls
        "overall_inference_duration_sec": overall_inference_duration_sec, # Wall time for the whole loop
        "avg_generate_time_per_sample_sec": avg_time_per_sample,
        "total_tokens_generated": total_tokens_generated, # Sum of effective tokens generated
        "tokens_per_second": tps, # Based on generate time
        "ram_before_inference_mb": ram_before, "ram_after_inference_mb": ram_after,
        "pytorch_vram_before_inference_mb": pt_vram_before, "pytorch_vram_after_inference_mb": pt_vram_after,
        "pytorch_vram_peak_inference_mb": pt_peak_vram_inf,
        "system_vram_before_inference_mb": sys_vram_before if NVML_FOUND else None,
        "system_vram_after_inference_mb": sys_vram_after if NVML_FOUND else None,
        "system_vram_peak_inference_approx_mb": peak_sys_vram_inference if NVML_FOUND else None,
        "num_samples_run": num_initial_samples, # Report initial number planned
        "num_results_produced": num_samples_processed # Report actual number in results list
    }

    return results, inference_metrics 
