import gc

import ray
import torch
from vllm import LLM, SamplingParams

# from vllm.distributed.parallel_state import destroy_model_parallel

# Sample prompts.
prompts = "Hello, my name is"
# Create a sampling params object.
sampling_params = SamplingParams()

# Create an LLM.
print("IS PROBLEM FROM BELOW?")
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=13456,
)
print("IS PROBLEM FROM ABOVE?")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# destroy_model_parallel()
del llm.llm_engine.model_executor
del llm
gc.collect()

# torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
ray.shutdown()

if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(0)
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - reserved_memory

    print(f"Total GPU memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"Reserved memory: {reserved_memory / (1024 ** 3):.2f} GB")
    print(f"Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
    print(f"Remaining available memory: {free_memory / (1024 ** 3):.2f} GB")
else:
    print(
        "CUDA is not available. Make sure you have a CUDA-capable device and PyTorch is installed with CUDA."
    )


llm = LLM(model="tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1")
