print("START WITH CUDA DEVICE COUNT")
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import pandas as pd
from datetime import datetime
import os
from peft import LoraConfig, get_peft_model, PeftModel
import huggingface_hub
from vllm import LLM, SamplingParams


hf_token = "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"
write_token = "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"

huggingface_hub.login(token = hf_token)
'''
cache_dir = "/vast/palmer/scratch/odea/das293/huggingface/"
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['HF_TOKEN'] = hf_token

sampling_params = SamplingParams(temperature=0.1, top_p=0.95, top_k=60, max_tokens = 512)

task = 'compare'
topic = "Frequency"
device = "auto"

# Path to the LoRA weights exported from Predibase
if task=='summary':
    ft_model = "danascott329/mixtral-document-summaries-telework"
if task=='compare':
    ft_model = "danascott329/mixtral-telework-compare"
    #ft_model = 'telework-compare-mixtral'

# fine-tuned linewise model
base_model = "mistralai/Mixtral-8x7B-Instruct-v0.1" 
tokenizer = AutoTokenizer.from_pretrained(base_model)       

# Prompt should be in this style due to how the data was created
#prompt = "#### Human: What is the capital of Australia?#### Assistant:"

model = LLM(model=ft_model, tokenizer = base_model, tensor_parallel_size=4)
