import argparse
import yaml
import os
import logging
import shutil
import numpy as np
from PIL import Image 
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from diffusers import DDIMScheduler
from datasets import load_dataset
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.wmpatch import GTWatermark, GTWatermarkMulti
from main.utils import *
from loss.loss import LossProvider
from loss.pytorch_ssim import ssim
import json
import requests 
import subprocess
import re
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
from pathlib import Path
from cleanfid import fid
from collections import defaultdict
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

os.environ["CUDA_VISIBLE_DEVICES"]='5'
device = torch.device('cuda')
print(device)

paraphrase_model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
pipeline_text2image = AutoPipelineForText2Image.from_pretrained(paraphrase_model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False).to(device)
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to(device)



# pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )



logging.info(f'===== Load Config =====')
with open('./example/config/config.yaml', 'r') as file:
    cfgs = yaml.safe_load(file)
logging.info(cfgs)

logging.info(f'===== Init Pipeline =====')
if cfgs['w_type'] == 'single':
    wm_pipe = GTWatermark(device, w_channel=cfgs['w_channel'], w_radius=cfgs['w_radius'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))
elif cfgs['w_type'] == 'multi':
    wm_pipe = GTWatermarkMulti(device, w_settings=cfgs['w_settings'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))

scheduler = DDIMScheduler.from_pretrained(cfgs['model_id'], subfolder="scheduler")
pipe = WMDetectStableDiffusionPipeline.from_pretrained(cfgs['model_id'], scheduler=scheduler).to(device)
pipe.set_progress_bar_config(disable=True)

with open('captions_train2014_new_format.json','r') as file : 
    new = json.load(file)

img_dir = '/raid/home/ashhar21137/watermarking_final_tests/zodiac/zodiac_watermarked_samples_with_AE'
img_ids = []
img_pths = []
imgs = os.listdir(img_dir)
for i in imgs : 
    if '.png' in i :
        id = re.split('[_.]',i)[1]
        img_ids.append(id) 
        img_pths.append(os.path.join(img_dir,i))

print("Imgs ids : ", img_ids)
print("Images : ",imgs)
wmis = os.listdir(img_dir) 
print("Wmis : ",wmis)

save_dir = '/raid/home/ashhar21137/watermarking_final_tests/zodiac/zodiac_paraphrases_gui'

strength_values = [0.10, 0.20, 0.30, 0.40, 0.50]

for strength in strength_values : 
    print(f"---------- Diffusion with strength value {strength} -------------")
    paraphrase_detection = defaultdict(lambda: defaultdict(dict))

    count = 1 
    for i in tqdm(range(len(wmis))) :
        # if count > 1 : break 
        print(f"---------- Count = {count} : Visual Paraphrasing for the watermarked version of {img_ids[i]}-------------")
        print()

        print("Image name : ",wmis[i])
        id = wmis[i].split('_')[1]
        print("Image id : ",img_ids[i])

        paraphrase_detection[id]["name"] = wmis[i]

        captions = new['annotations'][id]

        post_img = os.path.join(img_dir,wmis[i]) 
        image = Image.open(post_img)
        init_image = load_image(image)

        gen_image = pipeline(captions, image=init_image, strength=strength, guidance_scale=6).images
        
        directory = f'{save_dir}/{strength}/{id}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        for k in range(len(gen_image)):
            gen_save_dir = os.path.join(directory, f'{id}_gen_{k}.png')
            gen_image[k].save(gen_save_dir)
            print(f"Generated Image saved to {gen_save_dir}")

        num_images = len(gen_image) 
        print()
        print("Number of images generated : ",num_images)
        print("********** Watermark detection for generated images without Captions ***************")     

        avg_no_prompt = 0 
        num_detected = 0 
        for j in range(num_images):
            post_img = os.path.join(directory,f'{id}_gen_{j}.png')
            tester_prompt = '' 
            text_embeddings = pipe.get_text_embedding(tester_prompt)
            det_prob = 1 - watermark_prob(post_img, pipe, wm_pipe, text_embeddings)
            
            avg_no_prompt = avg_no_prompt + det_prob
            num_detected = num_detected = (det_prob>0.9)
            
            print(f'Watermark Presence Prob.: {det_prob}')

        paraphrase_detection[id]["without_captions_avg_det_prob"] = avg_no_prompt/num_images
        paraphrase_detection[id]["without_captions_det_rate"] = num_detected/num_images

        avg_with_prompt = 0 
        num_detected = 0

        print()
        print("********** Watermark detection for generated images with Captions ***************")  

        for j in range(num_images):
            post_img = os.path.join(directory,f'{id}_gen_{j}.png')
            print(f"caption : {captions[j]}")

            tester_prompt = captions[j]
            text_embeddings = pipe.get_text_embedding(tester_prompt)
            det_prob = 1 - watermark_prob(post_img, pipe, wm_pipe, text_embeddings)

            avg_with_prompt = avg_with_prompt + det_prob
            num_detected = num_detected + (det_prob>0.9)
            print(f'Watermark Presence Prob.: {det_prob}')
        
        paraphrase_detection[id]["with_captions_avg_det_prob"] = avg_with_prompt/num_images
        paraphrase_detection[id]["with_captions_det_rate"] = num_detected/num_images

        count = count + 1  

    with open(f"/raid/home/ashhar21137/watermarking_final_tests/zodiac/results_gui/zodiac_{strength}_paraphrased.json", "w") as file:
        json.dump(paraphrase_detection, file, indent=4)