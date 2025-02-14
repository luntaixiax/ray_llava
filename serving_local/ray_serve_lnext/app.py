import torch
from transformers import (LlavaNextProcessor, 
                          LlavaNextForConditionalGeneration, 
                          BitsAndBytesConfig
                          )

from constants import llava_next_template



import ray
from ray import serve
#from ray.serve.drivers import DAGDriver
from starlette.requests import Request


from PIL import Image

import requests
import logging
import warnings

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=DeprecationWarning) 


"""

serve run --non-blocking app_challenge:lnext_serve - prod
serve run app_challenge:lnext_serve - debug

-----
Docker

!!!ensure that your run model_dl_quant.py first 

sudo docker build -t ray-up .
#shm-size should be 30% of available mem -  Ray is mem hungry
sudo docker run -p 8000:8000 -p 8265:8265 --shm-size=20gb --gpus all ray-up



"""

# Asynchronous function to resolve incoming JSON requests
async def json_resolver(request: Request) -> dict:
    """
    Resolve incoming JSON requests asynchronously.

    Args:
        request: The incoming HTTP request containing JSON data.

    Returns:
        A dictionary representing the parsed JSON data.
    """
    return await request.json()
    
    
ray.init(address="auto", namespace="example")
serve.start(detached=True, http_options={"host": "0.0.0.0"})



@serve.deployment(
    ray_actor_options={
        "num_gpus": 1,
        "num_cpus": 6
    }
)
class LNXT:
        def __init__(self):
                self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
                self.processor.chat_template = llava_next_template
                self.model = LlavaNextForConditionalGeneration.from_pretrained("llava_quant", 
                                                        low_cpu_mem_usage=True)
                

        async def __call__(self, http_request: Request)  -> str:

                request = await http_request.json() 
                
                prompt, url = request["prompt"], request["url"]

                assert url is not None, 'please provide an image url!'                
                inputs = self.preprocess(prompt, url)
                assert torch.is_tensor(inputs['input_ids'])

                output = self.model.generate(**inputs, max_new_tokens=100)


                return self.processor.decode(output[0], skip_special_tokens=True)


        def eval(self, txt: str):
                encoded_text = self.tokenizer(txt, return_tensors="pt", padding=True, truncation = True)
                generated_tokens = self.model.generate(**encoded_text,forced_bos_token_id=self.tokenizer.get_lang_id(self.trg_lang))
                return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        def preprocess(self, prompt, url):
                """
                
                """

                conversation = [
                        {

                        "role": "user",
                        "content": [
                                {"type": "text", "text": prompt},
                                 {"type": "image"},
                                ],
                        },
                        ]
                
                image = Image.open(requests.get(url, stream=True).raw)
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(prompt, image, return_tensors="pt").to("cuda:0")

                return inputs
                        

#app = DAGDriver.bind(LNXT)
app = LNXT.bind()
serve.run(app)
