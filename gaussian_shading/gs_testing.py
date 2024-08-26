import os 
os.environ["CUDA_VISIBLE_DEVICES"]='6'

from tqdm import tqdm 
from PIL import Image, ImageEnhance
from skimage.util import random_noise
import cv2
import numpy as np 
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline, DDIMScheduler
# from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DDIMInverseScheduler
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import torch 
import tempfile
from torchvision import transforms
from collections import defaultdict
import hashlib

import copy
from transformers import CLIPModel, CLIPTokenizer
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *
import re
from types import SimpleNamespace
from collections import defaultdict

model_id = 'stabilityai/stable-diffusion-2-1-base'

if torch.cuda.is_available() : 
    device = "cuda"
else :
    device = "cpu"

print("device : ",device)


reference_model = None 
reference_model_pretrain = None 
model_path = "stabilityai/stable-diffusion-2-1-base"
channel_copy = 1 
hw_copy = 8 
fpr = 0.000001
num = 1000
user_number = 1000000
output_path = "/raid/home/ashhar21137/watermarking2/Gaussian-Shading/watermarked_images"
gen_seed = 0 
num_inference_steps = 50 
guidance_scale = 7.5
image_length = 512
chacha = True
num_inversion_steps = 50

args = SimpleNamespace(
    reference_model = None ,
    reference_model_pretrain = None ,
    model_path = "stabilityai/stable-diffusion-2-1-base",
    channel_copy = 1 ,
    hw_copy = 8 ,
    fpr = 0.000001,
    num = 1000,
    user_number = 1000000,
    output_path = "/raid/home/ashhar21137/watermarking2/Gaussian-Shading/watermarked_images",
    gen_seed = 0 ,
    num_inference_steps = 50 ,
    guidance_scale = 7.5,
    image_length = 512,
    chacha = True,
    num_inversion_steps = 50,
    jpeg_ratio = 0.25,
    random_crop_ratio =None ,
    random_drop_ratio = None, 
    gaussian_blur_r = None, 
    median_blur_k = None, 
    resize_ratio = None, 
    gaussian_std = None, 
    sp_prob = None, 
    brightness_factor = None
)


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
    expand = 0
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


parent = '/raid/home/ashhar21137/watermarking2/Gaussian-Shading/gs_watermarked_images'

ids = os.listdir(parent)
# ids

attacks = ["brightness","gaussian_noise","jpeg","rotation"]
attacks_op_parent = "/raid/home/ashhar21137/watermarking2/Gaussian-Shading/gs_attacked"

attack_detection = defaultdict(lambda: defaultdict(dict))


print("----------------- Performing Attacks ------------------")
count = 0 
for id in tqdm(ids): 
    # if count > 4: 
    #     break  # Only process the first ID for testing

    path = os.path.join(parent, id)
    
    for image in os.listdir(path):
        # if ".png" in image : 
        img_pth = os.path.join(path, image)
        img_name, img_ext = os.path.splitext(image)

        for attack in attacks: 
            output_path = os.path.join(f"{attacks_op_parent}/{attack}", f"{id}/{img_name + img_ext}")
            # print(output_path)
            
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
            

    # count += 1


print("----------------- Generating detection results ------------------")
for attack in tqdm(attacks) : 
    print(f" ----- Attack : {attack} -----")
    attacked_path = os.path.join(attacks_op_parent,attack) 
    attacked_ids = os.listdir(attacked_path)

    for id in attacked_ids :
        print(f" *** Image Id. : {id} ***")
        id_pth = os.path.join(attacked_path,id)
        

        images = os.listdir(id_pth)

        id_prob = 0 
        detection = 0 
        
        for image in images : 
            scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                    model_path,
                    scheduler=scheduler,
                    torch_dtype=torch.float16,
                    revision='fp16',
            )
            pipe.safety_checker = None
            pipe = pipe.to(device)

            # class for watermark
            if chacha:
                watermark = Gaussian_Shading_chacha(channel_copy, hw_copy, fpr, user_number)
            else:
                #a simple implement,
                watermark = Gaussian_Shading(channel_copy, hw_copy, fpr, user_number)

            img_path = os.path.join(id_pth,image)

            image_w_distortion = Image.open(img_path)



            # is_watermarked, probability = detect(img, pipeline1)

            tester_prompt = ''
            text_embeddings = pipe.get_text_embedding(tester_prompt)

            image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
            image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
            reversed_latents_w = pipe.forward_diffusion(
                latents=image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=num_inversion_steps,
            )

            #acc metric
            acc_metric = watermark.eval_watermark(reversed_latents_w)
            print("acc_metric : ",acc_metric)
            # acc.append(acc_metric)





            print(f"{attack} : {id} : {image} : Prob = {acc_metric}")
            print(f"{attack} : {id} : {image} : Watermarked? = {acc_metric>0.9}")

            id_prob += acc_metric
            detection += (acc_metric>0.9)

        print(id_prob/len(images))
        print(detection/len(images))

        attack_detection[id][attack]['avg_probability'] = id_prob/len(images)
        attack_detection[id][attack]['detection_rate'] = detection/len(images)

    

print("Preparing results json ")
import json 
with open('tree_ring_zeroes_detection_results.json','w') as file : 
    json.dump(attack_detection,file,indent=4)
