"""

"""
from vllm import LLM, SamplingParams
from loguru import logger
from vllm.platforms import current_platform

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
if __name__ == "__main__":
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    logger.info(f"vLLM platform: {current_platform.device_name} / {current_platform.device_type}")
    # "cuda" means GPU; "cpu" means the CPU backend
    llm = LLM(model="facebook/opt-125m")
    logger.info("llm model loaded successfully")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        logger.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")