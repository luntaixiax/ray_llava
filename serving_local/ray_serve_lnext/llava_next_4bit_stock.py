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
quant = False

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



if not quant:
    model.to("cuda:0")

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What is shown in this image?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)


logging.info(f'{formatting}{processor.decode(output[0], skip_special_tokens=True)}{formatting}')
