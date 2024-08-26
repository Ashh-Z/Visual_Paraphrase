import os 
import json 
from PIL import Image 
import numpy as np 
from tqdm import tqdm 
from PIL import Image, ImageEnhance
from skimage.util import random_noise
import cv2
import tempfile
from torchvision import transforms
from collections import defaultdict
import re
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

from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.wmpatch import GTWatermark, GTWatermarkMulti
from main.utils import *
from loss.loss import LossProvider
from loss.pytorch_ssim import ssim
import json
import requests 
import subprocess
import re


def brightness_attack(img_path, out_path, multi=False):
    brightness = 2
    if os.path.exists(out_path) and not multi:
        return
    
    img = Image.open(img_path)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness)
    img.save(out_path)


def gaussian_noise_attack(img_path, out_path, multi=False):
        std = 0.1
        
        if os.path.exists(out_path) and not multi:
            return 
        
        image = cv2.imread(img_path)
        image = image / 255.0
        # Add Gaussian noise to the image
        noise_sigma = std  # Vary this to change the amount of noise
        noisy_image = random_noise(image, mode='gaussian', var=noise_sigma ** 2)
        # Clip the values to [0, 1] range after adding the noise
        noisy_image = np.clip(noisy_image, 0, 1)
        noisy_image = np.array(255 * noisy_image, dtype='uint8')
        cv2.imwrite(out_path, noisy_image)


def rotate_attack(img_path, out_path, multi=False):
    degree = 30
    expand = 1
    if os.path.exists(out_path) and not multi:
        return 
    
    img = Image.open(img_path)
    img = img.rotate(degree, expand=expand)
    img = img.resize((512,512))
    img.save(out_path)


def jpeg_attack(img_path, out_path, multi=False):
    quality = 50
    if os.path.exists(out_path) and not multi:
        return 
    
    img = Image.open(img_path)
    img.save(out_path, "JPEG", quality=quality)

parent = '/raid/home/ashhar21137/watermarking_final_tests/zodiac/zodiac_watermarked_samples_with_AE'

ids = os.listdir(parent)

attacks = ["brightness","gaussian_noise","jpeg","rotation"]
attacks_op_parent = "/raid/home/ashhar21137/watermarking_final_tests/zodiac/attacked_zodiac"

print("----------------- Performing Attacks ------------------")
    
for image in tqdm(ids):
    # if ".png" in image : 
    img_pth = os.path.join(parent, image)
    
    # print("img_pth : ", img_pth)
    
    img_name, img_ext = os.path.splitext(image)
    
    # print("img_name : ",img_name)
    # print("img_ext : ",img_ext)

    for attack in attacks: 
        output_path = os.path.join(f"{attacks_op_parent}/{attack}", f"{attack}_{img_name + img_ext}")
        # print("output path : ",output_path)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if attack == "brightness":
            brightness_attack(img_pth, output_path)  # Pass individual paths

        elif attack == "gaussian_noise":
            gaussian_noise_attack(img_pth, output_path)

        elif attack == "rotation" :
            rotate_attack(img_pth,output_path)

        elif attack == "jpeg" : 
            jpeg_attack(img_pth,output_path)
        

logging.info(f'===== Load Config =====')
device = torch.device('cuda')
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


# post_img = os.path.join(wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}.png")

# post_img = '/raid/home/ashhar21137/watermarking/ZoDiac_one/attacked_zodiac/brightness/brightness_Image_8649_SSIM0.92.png'

# tester_prompt = '' # assume at the detection time, the original prompt is unknown
# text_embeddings = pipe.get_text_embedding(tester_prompt)


# logging.info(f'===== Testing the Watermarked Images {post_img} =====')
# det_prob = 1 - watermark_prob(post_img, pipe, wm_pipe, text_embeddings)
# logging.info(f'Watermark Presence Prob.: {det_prob}')


print("----------------- Generating detection results ------------------")
attacked_path = os.path.join(attacks_op_parent,attack) 
attacked_ids = os.listdir(attacked_path)

attack_detection = defaultdict(lambda: defaultdict(dict))


for attack in attacks : 
    print(f" ----- Attack : {attack} -----")

    attacked_path = os.path.join(attacks_op_parent,attack) 
    attacked_ids = os.listdir(attacked_path)

    for img in tqdm(attacked_ids) :
        print(f" *** Image : {img} ***")
        l = img.split('_')

        id = f"{l[1]}_{l[2]}"

        print("Image id : ",id)
        
        image_path = os.path.join(attacked_path,img)

        tester_prompt = ""
        text_embeddings = pipe.get_text_embedding(tester_prompt)

        logging.info(f'===== Testing the Watermarked Images {image_path} =====')
        det_prob = 1 - watermark_prob(image_path, pipe, wm_pipe, text_embeddings)
        logging.info(f'Watermark Presence Prob.: {det_prob}')

        print("Detection Probability : ", det_prob)

        attack_detection[id][f"{attack}_det_prob"] = det_prob


        # id_prob = 0 
        # detection = 0 
        

print("Preparing results json ")
import json 
with open('zodiac_attack_detection.json','w') as file : 
    json.dump(attack_detection,file,indent=4)







# count += 1
