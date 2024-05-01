import requests
from bs4 import BeautifulSoup
import json
import urllib

_GGUF_QUANTS = {
    "Q2_K": 3.35,
    "Q3_K_S": 3.5,
    "Q3_K_M": 3.91,
    "Q3_K_L": 4.27,
    "Q4_0": 4.55,
    "Q4_K_S": 4.58,
    "Q4_K_M": 4.85,
    "Q5_0": 5.54,
    "Q5_K_S": 5.54,
    "Q5_K_M": 5.69,
    "Q6_K": 6.59,
    "Q8_0": 8.5,
}

def get_gguf_quants():
    """
    Retrieves the available GGUF quantization sizes.

    Returns:
    - list: A list containing the keys of available GGUF quantization sizes.
    """
    return list(_GGUF_QUANTS.keys())

def _model_config(hf_model: str) -> dict:
    """
    Retrieves the configuration of the specified model from the Hugging Face Model Hub.

    Args:
    - hf_model (str): The name of the model on the Hugging Face Model Hub.

    Returns:
    - dict: The configuration of the model.
    """
    config_response = requests.get(f"https://huggingface.co/{hf_model}/raw/main/config.json")
    config = config_response.json()
    model_size = 0
    try:
        model_size_response = requests.get(f"https://huggingface.co/{hf_model}/resolve/main/model.safetensors.index.json")
        model_size = model_size_response.json()["metadata"]["total_size"] / 2
        if not model_size:
            raise ValueError("no size in safetensors metadata")
    except Exception as e:
        try:
            model_size_response = requests.get(f"https://huggingface.co/{hf_model}/resolve/main/pytorch_model.bin.index.json")
            model_size = model_size_response.json()["metadata"]["total_size"] / 2
            if not model_size:
                raise ValueError("no size in pytorch metadata")
        except Exception as e:
            model_page_response = requests.get(f"https://corsproxy.io/?{urllib.parse.quote('https://huggingface.co/' + hf_model)}")
            model_page = model_page_response.text
            soup = BeautifulSoup(model_page, 'html.parser')
            params_el = soup.find('div', {'data-target': 'ModelSafetensorsParams'})
            if params_el is not None:
                model_size = json.loads(params_el.attrs.get("data-props", "{}"))["safetensors"]["total"]
            else:
                params_el = soup.find('div', {'data-target': 'ModelHeader'})
                model_size = json.loads(params_el.attrs.get("data-props", "{}"))["model"]["safetensors"]["total"]
    if not model_size:
        raise ValueError("no size in pytorch metadata")
    config["parameters"] = model_size
    return config

def _input_buffer(context: int, model_config: dict, bsz: int) -> float:
    """
    Calculates the input buffer size.

    Args:
    - context (int): The context size.
    - model_config (dict): The configuration of the model.
    - bsz (int): Batch size.

    Returns:
    - float: The input buffer size.
    """
    inp_tokens = bsz
    inp_embd = model_config["hidden_size"] * bsz
    inp_pos = bsz
    inp_KQ_mask = context * bsz
    inp_K_shift = context
    inp_sum = bsz
    return inp_tokens + inp_embd + inp_pos + inp_KQ_mask + inp_K_shift + inp_sum

def _compute_buffer(context: int, model_config: dict, bsz: int) -> float:
    """
    Calculates the compute buffer size.

    Args:
    - context (int): The context size.
    - model_config (dict): The configuration of the model.
    - bsz (int): Batch size.

    Returns:
    - float: The compute buffer size.
    """
    if bsz != 512:
        print("batch size other than 512 is currently not supported for the compute buffer, using batchsize 512 for compute buffer calculation, end result result will be an overestimation")
    return (context / 1024 * 2 + 0.75) * model_config["num_attention_heads"] * 1024 * 1024

def _kv_cache(context: int, model_config: dict, cache_bit: int) -> float:
    """
    Calculates the key-value cache size.

    Args:
    - context (int): The context size.
    - model_config (dict): The configuration of the model.
    - cache_bit (int): Size of cache in bits.

    Returns:
    - float: The key-value cache size.
    """
    n_gqa = model_config["num_attention_heads"] / model_config["num_key_value_heads"]
    n_embd_gqa = model_config["hidden_size"] / n_gqa
    n_elements = n_embd_gqa * (model_config["num_hidden_layers"] * context)
    size = 2 * n_elements
    return size * (cache_bit / 8)

def _context_size(context: int, model_config: dict, bsz: int, cache_bit: int) -> float:
    """
    Calculates the total context size.

    Args:
    - context (int): The context size.
    - model_config (dict): The configuration of the model.
    - bsz (int): Batch size.
    - cache_bit (int): Size of cache in bits.

    Returns:
    - float: The total context size.
    """
    return round(_input_buffer(context, model_config, bsz) + _kv_cache(context, model_config, cache_bit) + _compute_buffer(context, model_config, bsz), 2)

def _model_size(model_config, bpw: float) -> float:
    """
    Calculates the size of the model.

    Args:
    - model_config (dict): The configuration of the model.
    - bpw (float): Bits per weight.

    Returns:
    - float: The size of the model.
    """
    return round(model_config["parameters"] * bpw / 8, 2)

def compute_sizes_exl2(hf_model: str, context: int, cache_bit: int = 16, bpw: float = 4.5) -> dict:
    """
    Computes the sizes (model size, context size, and total size) excluding L2 cache.

    Args:
    - hf_model (str): The name of the model on the Hugging Face Model Hub.
    - context (int): The context size.
    - cache_bit (int): Size of cache in bits.
    - bpw (float): Bits per weight.

    Returns:
    - dict: Dictionary containing model size, context size, and total size.
    """
    batch_size = 512

    model_config_data = _model_config(hf_model)
    model_sz = _model_size(model_config_data, bpw) / (2**30)
    context_sz = _context_size(context, model_config_data, batch_size, cache_bit) / (2**30)
    total_sz = (model_sz + context_sz)
    
    return {"model_size": model_sz, "context_size": context_sz, "total_size": total_sz}

def compute_sizes_gguf(hf_model: str, context: int, quant_size: str = "") -> dict:
    """
    Computes the sizes (model size, context size, and total size) using GGUF quantization.

    Args:
    - hf_model (str): The name of the model on the Hugging Face Model Hub.
    - context (int): The context size.
    - quant_size (str): Quantization size.

    Returns:
    - dict: Dictionary containing model size, context size, and total size. All in GB.
    """
    batch_size = 512

    model_config_data = _model_config(hf_model)
    bpw = _GGUF_QUANTS.get(quant_size, 0)
    model_sz = _model_size(model_config_data, bpw) / (2**30)
    context_sz = _context_size(context, model_config_data, batch_size, 16) / (2**30)
    total_sz = (model_sz + context_sz)

    return {"model_size": model_sz, "context_size": context_sz, "total_size": total_sz}