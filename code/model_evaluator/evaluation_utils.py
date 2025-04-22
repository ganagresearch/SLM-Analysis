import psutil
import torch
import logging
import os
import re

try:
    import pynvml
    NVML_FOUND = True
except ImportError:
    NVML_FOUND = False

nvml_handle = None
nvml_initialized_successfully = False

def setup_logging():
    """Configures basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def init_nvml(device_str='cuda'):
    """Initializes NVML and gets handle for the specified device."""
    global nvml_handle, nvml_initialized_successfully, NVML_FOUND

    if not NVML_FOUND:
        logging.warning("pynvml not found, GPU system memory usage will not be reported. Run 'pip install nvidia-ml-py' to enable.")
        return None, False

    try:
        pynvml.nvmlInit()
        device_id = 0 # Default
        if device_str.startswith('cuda:'):
            try:
                device_id = int(device_str.split(':')[1])
            except (ValueError, IndexError):
                logging.warning(f"Could not parse device ID from '{device_str}'. Defaulting to device 0 for NVML.")
        elif device_str != 'cuda':
            logging.info("Target device is not CUDA, NVML reporting disabled.")
            NVML_FOUND = False # Ensure flag is False if not CUDA
            pynvml.nvmlShutdown() # Shutdown if we initialized but aren't using CUDA
            return None, False

        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        logging.info(f"Initialized NVML for device {device_id}.")
        nvml_initialized_successfully = True
        return nvml_handle, True

    except pynvml.NVMLError as error:
        logging.warning(f"Failed to initialize NVML: {error}. GPU system memory usage will not be reported.")
        NVML_FOUND = False # Ensure flag is False if init fails
        nvml_handle = None
        nvml_initialized_successfully = False
        return None, False

def shutdown_nvml():
    """Shuts down NVML if it was successfully initialized."""
    global nvml_initialized_successfully
    if nvml_initialized_successfully:
        try:
            pynvml.nvmlShutdown()
            logging.info("NVML shut down.")
            nvml_initialized_successfully = False
        except pynvml.NVMLError as error:
            logging.warning(f"Failed to shutdown NVML: {error}")

def get_memory_usage():
    """
    Returns current RAM usage (MB), current PyTorch VRAM (MB),
    peak PyTorch VRAM (MB) since last reset, and current System VRAM (MB).
    Uses the global nvml_handle.
    """
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024) # MB

    pytorch_vram_current = 0
    pytorch_vram_peak = 0
    system_vram_current = 0

    if torch.cuda.is_available():
        pytorch_vram_current = torch.cuda.memory_allocated() / (1024 * 1024) # MB
        pytorch_vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB

        if NVML_FOUND and nvml_handle: # Check handle validity
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                system_vram_current = mem_info.used / (1024 * 1024) # MB
            except pynvml.NVMLError as error:
                logging.warning(f"NVML query failed: {error}. System VRAM usage not available.")

    return ram_usage, pytorch_vram_current, pytorch_vram_peak, system_vram_current

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
        return None 
