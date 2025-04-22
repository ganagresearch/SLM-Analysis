import argparse
import json
import os
import time
import torch
import logging

# Import modular components
from evaluation_utils import setup_logging, init_nvml, shutdown_nvml, get_memory_usage, NVML_FOUND
from model_loader import load_model_and_tokenizer
from benchmark_loader import load_benchmark_prompts
from inference_runner import run_inference

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmark evaluation.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=str, required=True, help="Path to the local Hugging Face model directory.")
    parser.add_argument("--model-type", type=str, default="causal", choices=['causal', 'encoder'], help="Type of model architecture.")
    parser.add_argument("--benchmark-name", type=str, required=True, choices=['cyberseceval3_mitre', 'sevenllm_bench', 'ctibench'], help="Name identifier for the benchmark task.")
    parser.add_argument("--benchmark-path", type=str, required=True, help="Path to the benchmark data file/dir or cache directory for HF datasets.")
    parser.add_argument("--results-dir", type=str, default="/workspace/results", help="Directory to save results.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on ('cuda' or 'cpu').")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for model generation.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to load/run (loads/runs all if None or <= 0).")
    parser.add_argument("--cti-subset", type=str, default=None, help="Specify the CTIBench subset name if using benchmark-name 'ctibench'.")
    args = parser.parse_args()

    setup_logging() # Configure logging first

    nvml_handle, nvml_active = init_nvml(args.device) # Initialize NVML

    model = None # Define model/tokenizer outside try block for cleanup
    tokenizer = None
    all_metrics = {} # Initialize metrics dict

    try:
        # --- Initial Setup & Arg Validation ---
        initial_ram, _, _, initial_sys_vram = get_memory_usage()

        num_samples_to_load = args.num_samples
        if num_samples_to_load is not None and num_samples_to_load <= 0:
            num_samples_to_load = None

        os.makedirs(args.results_dir, exist_ok=True)
        model_name_for_file = os.path.basename(args.model_path).replace('/','_')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        subset_tag = f"_{args.cti_subset}" if args.benchmark_name == 'ctibench' and args.cti_subset else ""
        results_filename_base = f"{args.benchmark_name}{subset_tag}_{model_name_for_file}_{timestamp}"
        results_filepath = os.path.join(args.results_dir, f"{results_filename_base}_outputs.json")
        metrics_filepath = os.path.join(args.results_dir, f"{results_filename_base}_metrics.json")

        logging.info("--- Starting Evaluation ---")
        logging.info(f"Args: {vars(args)}")
        logging.info(f"Results Dir: {args.results_dir}")
        logging.info(f"Metrics file: {metrics_filepath}")
        logging.info(f"Outputs file: {results_filepath}")
        logging.info(f"NVML Reporting Active: {nvml_active}")
        logging.info(f"Initial RAM usage: {initial_ram:.2f} MB")
        if nvml_active: logging.info(f"Initial System VRAM usage: {initial_sys_vram:.2f} MB")

        if args.benchmark_name == 'ctibench' and not args.cti_subset:
            raise ValueError("Missing --cti-subset for ctibench")

        # --- Load Model ---
        model, tokenizer, load_metrics_partial = load_model_and_tokenizer(args.model_path, args.model_type, args.device)
        if model is None or tokenizer is None:
            raise RuntimeError("Model/Tokenizer loading failed")

        load_metrics = {
            "ram_initial_mb": initial_ram,
            "system_vram_current_initial_mb": initial_sys_vram if nvml_active else None,
            **load_metrics_partial
        }
        if nvml_active and load_metrics.get("system_vram_current_after_load_mb") is not None:
             sys_vram_delta = load_metrics["system_vram_current_after_load_mb"] - initial_sys_vram
             logging.info(f"System VRAM Delta during load: {sys_vram_delta:.2f} MB")

        # --- Load Benchmark ---
        prompts_data = load_benchmark_prompts(args.benchmark_name, args.benchmark_path, args.cti_subset, num_samples_to_load)
        if prompts_data is None: raise RuntimeError("Prompt loading failed (returned None)")
        if not prompts_data:
             logging.warning("No prompts loaded. Exiting.")
             inference_results = []
             inference_metrics = {} # No inference if no prompts
        else:
            # --- Run Inference ---
            if args.model_type == 'encoder':
                logging.warning(f"Model type is '{args.model_type}'. Skipping generation.")
                inference_results = []
                inference_metrics = {}
            else:
                inference_results, inference_metrics = run_inference(
                    model, tokenizer, prompts_data, args.device, args.max_new_tokens
                )
                # Log inference summary here
                logging.info(f"\n--- Inference Summary ---")
                logging.info(f"Processed {inference_metrics.get('num_samples_run', 0)} samples.")
                logging.info(f"Total 'generate' time (sum): {inference_metrics.get('total_generate_time_sec', 0):.2f} sec")
                logging.info(f"Overall inference loop duration: {inference_metrics.get('overall_inference_duration_sec', 0):.2f} sec")
                logging.info(f"Average 'generate' time per sample: {inference_metrics.get('avg_generate_time_per_sample_sec', 0):.4f} sec")
                logging.info(f"Total effective tokens generated: {inference_metrics.get('total_tokens_generated', 0)}")
                logging.info(f"Overall effective tokens per second: {inference_metrics.get('tokens_per_second', 0):.2f}")
                ram_delta_inf = inference_metrics.get('ram_after_inference_mb', 0) - inference_metrics.get('ram_before_inference_mb', 0)
                logging.info(f"RAM Delta during inference: {ram_delta_inf:.2f} MB")
                pt_vram_delta_peak_inf = inference_metrics.get('pytorch_vram_peak_inference_mb', 0) - inference_metrics.get('pytorch_vram_before_inference_mb', 0)
                logging.info(f"PyTorch VRAM Peak Delta during inference: {pt_vram_delta_peak_inf:.2f} MB")
                if nvml_active:
                    sys_vram_delta_peak_inf = inference_metrics.get('system_vram_peak_inference_approx_mb', 0) - inference_metrics.get('system_vram_before_inference_mb', 0)
                    logging.info(f"System VRAM Peak Approx. Delta during inference: {sys_vram_delta_peak_inf:.2f} MB")


        # --- Combine & Save ---
        all_metrics = { "run_args": vars(args), "load_metrics": load_metrics, "inference_metrics": inference_metrics }

        try:
            with open(results_filepath, 'w') as f: json.dump(inference_results, f, indent=4)
            logging.info(f"Saved outputs ({len(inference_results)} items) to: {results_filepath}")
        except Exception as e: logging.error(f"Error saving outputs: {e}", exc_info=True)

        try:
            with open(metrics_filepath, 'w') as f: json.dump(all_metrics, f, indent=4)
            logging.info(f"Saved metrics to: {metrics_filepath}")
        except Exception as e: logging.error(f"Error saving metrics: {e}", exc_info=True)

    except Exception as e:
         logging.error(f"An error occurred during evaluation pipeline: {e}", exc_info=True)

    finally:
        # --- Cleanup ---
        logging.info("Cleaning up resources...")
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("CUDA cache cleared.")
        shutdown_nvml() # Attempt NVML shutdown

    logging.info("--- Evaluation Complete ---")

if __name__ == "__main__":
    main() 
