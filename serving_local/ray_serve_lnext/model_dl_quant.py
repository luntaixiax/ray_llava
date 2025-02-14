from transformers import (LlavaNextProcessor, 
                          LlavaNextForConditionalGeneration, 
                          BitsAndBytesConfig
                          )
import torch
from PIL import Image
import requests

import logging
import warnings

from constants import llava_next_template


logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=DeprecationWarning) 


base_f = '#'*40
formatting = f'\n\n{base_f}\n\n'
quant = True

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")


#otherwise a ValueError no template found is thrown with the latest transformers update
processor.chat_template = llava_next_template
if quant:
    #available on a a single consumer gpu, single node
    load_in_4bit=True
    torch_dtype=torch.float32
    attn_implementation=None

else:
    #requires multiple gpus
    load_in_4bit=False
    torch_dtype=torch.float16
    attn_implementation="flash_attention_2"
    quantization_config=None



model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", 
                                                        torch_dtype=torch_dtype, 
                                                        quantization_config=quantization_config,
                                                        attn_implementation=attn_implementation,
                                                        low_cpu_mem_usage=True) 

model.save_pretrained("llava_quant", from_pt=True) 
model = LlavaNextForConditionalGeneration.from_pretrained("llava_quant")

