# SLM-Analysis: Evaluating Small Language Models for Security Operations

## Project Goal

This project aims to evaluate the performance and resource efficiency of various Small Language Models (SLMs) for tasks relevant to security operations. The primary motivation is to assess their suitability for local deployment in resource-constrained environments, overcoming potential privacy, security, and cost limitations of larger cloud-based models.

The evaluation focuses on:

* **Security-Specific Benchmarks:** Assessing performance on tasks like threat modeling, incident response assistance, and CTI analysis.
* **Standard NLP Benchmarks:** Establishing baseline language understanding capabilities.
* **Resource Consumption:** Measuring inference speed, VRAM usage, and RAM usage.
* **Model Compression:** Investigating the impact of techniques like Post-Training Quantization (AutoGPTQ) on performance and efficiency trade-offs.

## Repository Structure

```plaintext
.
├── .git/                     # Git internal data
├── .gitignore                # Specifies intentionally untracked files by Git
├── code/                     # Core Python scripts for evaluation and analysis
│   ├── model_evaluator/      # Modules for custom security benchmark evaluation
│   │   ├── benchmark_loader.py # Loads and prepares benchmark datasets
│   │   ├── evaluate_cli.py     # CLI interface for running individual evaluations
│   │   ├── evaluation_utils.py # Utility functions (logging, metrics)
│   │   ├── inference_runner.py # Handles model inference execution
│   │   ├── load_sevenllm.py    # Preprocesses raw SEvenLLM data for evaluation/calibration
│   │   └── model_loader.py     # Loads Hugging Face models (base and quantized)
│   ├── quantize_models.py    # Script for applying AutoGPTQ quantization
│   ├── run_benchmark.py      # Main script to run security benchmarks evaluation suite
│   ├── run_nlp.py            # Main script to run standard NLP benchmarks via lm-eval-harness
│   └── run_benchmark_quantized.py # Main script to run security benchmarks on QUANTIZED models
├── datasets/                 # Placeholder for data (raw data ignored by git)
│   ├── SEVENLLM_raw/         # Example: Location for raw SEvenLLM data (add to .gitignore)
│   └── .hf_cache/            # Hugging Face datasets cache (add to .gitignore)
├── lm-evaluation-harness/    # Required Git submodule or clone (see Setup)
├── models/                   # Placeholder for model weights (ignored by git)
│   ├── Mistral-7B-Instruct-v0.3/
│   │   └── quantized-gptq-4bit/ # Example: Location for quantized weights
│   └── ...                   # Other model directories
├── results/                  # Output directory for metrics and logs (ignored by git)
│   ├── bare/48/              # Security benchmark outputs
│   ├── nlp/bare/48/          # NLP benchmark outputs
│   └── quantization_logs/    # Logs from the quantization process
└── README.md                 # This file
```

**Note:** Large files and generated output (`models/`, `results/`, raw datasets, `.hf_cache/`, `*venv*/`) should be included in `.gitignore` and are not tracked in this repository.

## Setup

1. **Clone the Repository:**

    ```bash
    git clone <repository-url>
    cd SLM-Analysis
    ```

2. **Set up Python Environment:**
    It's recommended to use a virtual environment (e.g., `venv`, `conda`).

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**
    *(Assumption: A `requirements.txt` file will be created)*

    ```bash
    pip install -r requirements.txt
    ```

    Key dependencies include `transformers`, `datasets`, `torch`, `accelerate`, `auto-gptq`, `optimum`, `psutil`, `pynvml` (for NVIDIA GPU monitoring).

4. **Install `lm-evaluation-harness`:**
    This project uses EleutherAI's `lm-evaluation-harness` for standard NLP benchmarks. Clone it into the project directory (or adjust paths in `code/run_nlp.py` accordingly):

    ```bash
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    pip install -e .
    cd ..
    ```

5. **Download Models:**
    The required models (e.g., `Mistral-7B-Instruct-v0.3`, `TinyLlama-1.1b-chat-v1.0`, `Phi-3-mini-4k-instruct`, `google/electra-small-discriminator`) are not included. Download them from the Hugging Face Hub and place them in the `models/` directory (or update paths in the scripts).

    ```bash
    # Example (requires huggingface_hub CLI)
    # huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/Mistral-7B-Instruct-v0.3 --local-dir-use-symlinks False
    # ... repeat for other models ...
    ```

6. **Prepare Datasets:**
    * **Security Benchmarks:**
        * `cyberseceval3_mitre`: Assumed to be available in a specific JSON format expected by `benchmark_loader.py`. Place the data where the script can find it (e.g., within a `data/` directory not tracked by git, and update paths if necessary). The original benchmark can be found within the [PurpleLlama](https://github.com/facebookresearch/PurpleLlama) repository.
        * `sevenllm_bench`: Requires preprocessing from its raw `.jsonl` format. Place the raw data (e.g., in `datasets/SEVENLLM_raw/`) and run `code/model_evaluator/load_sevenllm.py` to generate the processed Hugging Face `datasets` version used by the evaluation scripts. This script also generates a calibration subset (`calibration_data/sevenllm_instruct_subset_manual/`).
        * `ctibench`: Loaded directly from the Hugging Face Hub using `datasets.load_dataset` within `benchmark_loader.py`. Ensure you have internet connectivity during the first run or use a cached version.
    * **NLP Benchmarks:** Handled automatically by `lm-evaluation-harness`. Data will be downloaded to the Hugging Face cache (`.hf_cache/` or default location).

## Running Evaluations & Quantization

Scripts are located in the `code/` directory. Execute them from the root `SLM-Analysis` directory. Ensure your virtual environment is activated.

* **Security Benchmarks (Base Models):**
    Uses `code/run_benchmark.py`. Edit the script to configure models, benchmarks, batch sizes, and output paths.

    ```bash
    python code/run_benchmark.py
    ```

    Outputs are saved in `results/bare/48/` by default, containing model generations (`*_outputs.json`) and performance metrics (`*_metrics.json`).

* **Security Benchmarks (Quantized Models):**
    Uses `code/run_benchmark_quantized.py`. This script is specifically designed to load and evaluate AutoGPTQ models. Ensure the quantized models (e.g., in `models/Mistral-7B-Instruct-v0.3/quantized-gptq-4bit/`) exist. Edit the script to configure models, benchmarks, etc.

    ```bash
    python code/run_benchmark_quantized.py
    ```

    Outputs are saved similarly to the base model runs (adjust paths in the script if needed).

* **Standard NLP Benchmarks (Base Models):**
    Uses `code/run_nlp.py`, which wraps `lm-evaluation-harness`. Edit the script to configure models and tasks.

    ```bash
    python code/run_nlp.py
    ```

    Outputs are saved in `results/nlp/bare/48/<model_name>/<task_name>/results.json`.

* **Model Quantization (AutoGPTQ):**
    Uses `code/quantize_models.py`. Edit the script to specify the base model path, the dataset to use for calibration (e.g., the SEvenLLM subset created by `load_sevenllm.py`), quantization parameters (bits, group size), and the output path for the quantized model.

    ```bash
    python code/quantize_models.py
    ```

    Quantized model files will be saved to the specified output directory (e.g., `models/<base_model_name>/quantized-gptq-4bit/`). Logs may be saved to `results/quantization_logs/`.
    *Note: Quantization, especially for larger models, can be very memory-intensive (both VRAM and RAM).*

## Interpreting Output

* **Security Benchmark Results (`results/bare/48/`):**
  * `*_outputs.json`: Contains the prompts given to the model and the generated responses.
  * `*_metrics.json`: Contains run arguments (model, benchmark, batch size, etc.) and performance metrics (load time, inference time, tokens/sec, RAM/VRAM usage).
* **NLP Benchmark Results (`results/nlp/bare/48/`):**
  * Follows the structure defined by `lm-evaluation-harness`. `results.json` files contain standard NLP metrics (e.g., accuracy, F1) for each task.
* **Quantization Logs (`results/quantization_logs/`):**
  * Contain logs generated during the `quantize_models.py` script execution, useful for debugging the quantization process.

## Models Evaluated

* Mistral 7B Instruct v0.3 (`mistralai/Mistral-7B-Instruct-v0.3`)
* TinyLlama 1.1B Chat v1.0 (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
* Phi-3 Mini 4k Instruct (`microsoft/Phi-3-mini-4k-instruct`)
* Electra Small Discriminator (`google/electra-small-discriminator`)

4-bit AutoGPTQ quantized versions were created and evaluated for Mistral, TinyLlama, and Phi-3 using the SEvenLLM dataset for calibration.

## Benchmarks Used

* **Security:**
  * `cyberseceval3_mitre` (subset of PurpleLlama CyberSecEval v3)
  * `sevenllm_bench` (various security tasks)
  * `ctibench` (various CTI tasks from HF Hub)
* **NLP (via lm-evaluation-harness):**
  * `arc_challenge`
  * `hellaswag`
  * `winogrande`
  * `gsm8k`
  * `mmlu_elementary_mathematics`
  * `truthfulqa_mc2`

## Development Note

Most development and testing were performed on cloud instances (NVIDIA A40, H200) using a specific environment setup (`/workspace/`, `testbedvenv/`). Paths and configurations in the scripts may need adjustment for different environments. The code aims to be generally runnable where dependencies are met.
