import time
import torch
import logging
from tqdm import tqdm
from evaluation_utils import get_memory_usage, NVML_FOUND, nvml_handle # Import necessary utils

def run_inference(model, tokenizer, prompts_data, device='cuda', max_new_tokens=100):
    """
    Runs inference on the provided prompts_data, collects metrics.
    Returns results list and inference_metrics dict.
    """
    results = []
    total_inference_time_sec = 0
    total_tokens_generated = 0 # Effective tokens (excluding final special token)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        logging.info("Reset peak PyTorch VRAM stats for inference.")

    # Get memory usage *before* inference loop
    ram_before, pt_vram_before, _, sys_vram_before = get_memory_usage()
    # Logging moved to main CLI

    logging.info(f"Starting inference on {len(prompts_data)} samples...")
    inference_loop_start_time = time.time()

    if tokenizer.pad_token is None:
        logging.warning("Using eos_token as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    peak_sys_vram_inference = sys_vram_before if NVML_FOUND and nvml_handle else 0

    for item in tqdm(prompts_data, desc="Inference Progress"):
        prompt_text = item.get("prompt")
        item_id = item.get("id", "N/A")
        if not prompt_text: continue

        # --- Apply Chat Template ---
        messages = [{"role": "user", "content": prompt_text}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
             logging.warning(f"Chat template failed for {item_id}: {e}. Using raw prompt.")
             formatted_prompt = prompt_text

        # --- Tokenization ---
        try:
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096 - max_new_tokens)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else None
        except Exception as e:
             logging.error(f"Tokenization failed for {item_id}: {e}", exc_info=True)
             results.append({"item_id": item_id, "prompt": prompt_text, "response": f"Tokenization Error: {e}", "inference_time_sec": 0, "tokens_generated": 0})
             continue

        # --- Generation ---
        iter_start_time = time.time()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available(): torch.cuda.synchronize()
            iter_inference_time = time.time() - iter_start_time

            # --- Decoding & VRAM Peak Update ---
            num_input_tokens = input_ids.shape[-1]
            response_ids = outputs[0][num_input_tokens:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            num_generated = len(response_ids)

            if num_generated == 1 and response_ids[0] in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                response_text = ""

            if NVML_FOUND and nvml_handle:
                _, _, _, current_sys_vram = get_memory_usage()
                peak_sys_vram_inference = max(peak_sys_vram_inference, current_sys_vram)

            # --- Store results ---
            results.append({
                "item_id": item_id, "prompt": prompt_text, "response": response_text,
                "inference_time_sec": iter_inference_time, "tokens_generated": num_generated
            })
            total_inference_time_sec += iter_inference_time
            effective_tokens = max(0, num_generated -1) if num_generated > 0 and response_ids[-1] in [tokenizer.eos_token_id, tokenizer.pad_token_id] else num_generated
            total_tokens_generated += effective_tokens

        except Exception as e:
            logging.error(f"Generation failed for {item_id}: {e}", exc_info=True)
            results.append({"item_id": item_id, "prompt": prompt_text, "response": f"Generation Error: {e}", "inference_time_sec": time.time() - iter_start_time, "tokens_generated": 0})
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                logging.warning("OOM detected, clearing CUDA cache.")
                torch.cuda.empty_cache()

    # --- Post-Loop Metrics ---
    inference_loop_end_time = time.time()
    overall_inference_duration_sec = inference_loop_end_time - inference_loop_start_time
    ram_after, pt_vram_after, pt_peak_vram_inf, sys_vram_after = get_memory_usage()
    num_samples_processed = len(prompts_data)
    avg_time = total_inference_time_sec / num_samples_processed if num_samples_processed > 0 else 0
    tps = total_tokens_generated / total_inference_time_sec if total_inference_time_sec > 0 else 0

    inference_metrics = {
        "total_generate_time_sec": total_inference_time_sec,
        "overall_inference_duration_sec": overall_inference_duration_sec,
        "avg_generate_time_per_sample_sec": avg_time,
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": tps,
        "ram_before_inference_mb": ram_before, "ram_after_inference_mb": ram_after,
        "pytorch_vram_before_inference_mb": pt_vram_before, "pytorch_vram_after_inference_mb": pt_vram_after,
        "pytorch_vram_peak_inference_mb": pt_peak_vram_inf,
        "system_vram_before_inference_mb": sys_vram_before if NVML_FOUND else None,
        "system_vram_after_inference_mb": sys_vram_after if NVML_FOUND else None,
        "system_vram_peak_inference_approx_mb": peak_sys_vram_inference if NVML_FOUND else None,
        "num_samples_run": num_samples_processed
    }
    # Logging of summary moved to main CLI

    return results, inference_metrics 
