from vllm import LLM, SamplingParams

model = LLM("EleutherAI/gpt-neo-2.7B")
output = model.generate("What is the capital of France?", SamplingParams(max_tokens=50))
print(output)