# LLM VRAM Calculator

The `LLMVRAMCalculator.py` provides a tool for estimating the VRAM requirements for running large language models (LLMs) on GPUs. It calculates the sizes of the model, context, and total VRAM required for EXL2 and GGUF quantization.

## Kudos to NyxKrage
This wouldnt be possible without the [LLM Model VRAM Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) from NyxKrage, so thank you and all credits to him!

## Install
Install using pip:
```bash
pip install git+https://github.com/pandora-s-git/LLMVRAMCalculator.git
```

## Example

```python
import json
import LLMVRAMCalculator

exl2 = LLMVRAMCalculator.compute_sizes_exl2("Nexusflow/Starling-LM-7B-beta", 8192, cache_bit = 16, bpw = 4.5)
print(json.dumps(exl2, indent = 4))

print(LLMVRAMCalculator.get_gguf_quants())

gguf = LLMVRAMCalculator.compute_sizes_gguf("Nexusflow/Starling-LM-7B-beta", 8192, quant_size = "Q4_K_S")
print(json.dumps(gguf, indent = 4))
```
Output:
```shell
{
    "model_size": 3.793727159500122,
    "context_size": 1.5293059349060059,
    "total_size": 5.323033094406128
}
['Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L', 'Q4_0', 'Q4_K_S', 'Q4_K_M', 'Q5_0', 'Q5_K_S', 'Q5_K_M', 'Q6_K', 'Q8_0']
{
    "model_size": 3.8611711978912355,  
    "context_size": 1.5293059349060059,
    "total_size": 5.390477132797241    
}
```