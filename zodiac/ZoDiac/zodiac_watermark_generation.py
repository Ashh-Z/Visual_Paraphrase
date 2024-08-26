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
from tqdm import tqdm 
import json 


os.environ["CUDA_VISIBLE_DEVICES"]='4'
device = torch.device('cuda')
print(device)

parent_img_dir = '/raid/home/ashhar21137/watermarking/original_images'
original_images = os.listdir(parent_img_dir)

img_ids = []
img_pths = []
for img in original_images : 
    id = re.split('[_.]',img)[1]
    img_ids.append(id) 
    img_pths.append(os.path.join(parent_img_dir,img))

with open('/raid/home/ashhar21137/watermarking_final_tests/zodiac/ZoDiac/captions_train2014_new_format.json','r') as file : 
    data = json.load(file)

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

# imagename = 'Image_57870.jpg'
# imagename = original_images[0]
# print(imagename)
print(img_pths[0])
gt_img_tensor = get_img_tensor(img_pths[0], device)
print(gt_img_tensor.shape)
wm_path = cfgs['save_img']

# Step 1: Get init noise
def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
    # DDIM inversion from the given image
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    )
    return reversed_latents

def binary_search_theta(threshold, lower=0., upper=1., precision=1e-6, max_iter=1000):
    for i in range(max_iter):
        mid_theta = (lower + upper) / 2
        img_tensor = (gt_img_tensor-wm_img_tensor)*mid_theta+wm_img_tensor
        ssim_value = ssim(img_tensor, gt_img_tensor).item()

        if ssim_value <= threshold:
            lower = mid_theta
        else:
            upper = mid_theta
        if upper - lower < precision:
            break
    return lower


empty_text_embeddings = pipe.get_text_embedding('')
init_latents_approx = get_init_latent(gt_img_tensor, pipe, empty_text_embeddings)

print(original_images[0].split('.')[0])

save_path1 = '/raid/home/ashhar21137/watermarking_final_tests/zodiac/zodiac_watermarked_samples'
save_path2 = '/raid/home/ashhar21137/watermarking_final_tests/zodiac/zodiac_watermarked_samples_with_AE'

detection_probs = dict()

# count = 0 
for img in tqdm(original_images) : 
    # if count > 2 : break 
    print(f" ----- Watermarking {img} -----")
    path = [pth for pth in img_pths if img in pth][0]
    id = img.split('.')[0]

    gt_img_tensor = get_img_tensor(path,device)
    print("Tensor Size :",gt_img_tensor.shape)

    wm_path = cfgs['save_img']

    # Step 1: Get init noise
    empty_text_embeddings = pipe.get_text_embedding('') ## Watermarking with empty prompts
    init_latents_approx = get_init_latent(gt_img_tensor, pipe, empty_text_embeddings)

    # Step 2:
    init_latents = init_latents_approx.detach().clone()
    init_latents.requires_grad = True
    optimizer = optim.Adam([init_latents], lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.3) 

    totalLoss = LossProvider(cfgs['loss_weights'], device)
    loss_lst = [] 

    # Step 3: train the init latents
    for iter in range(cfgs['iters']):
        logging.info(f'iter {iter}:')
        init_latents_wm = wm_pipe.inject_watermark(init_latents)
        # Runs with empty prompt
        if cfgs['empty_prompt']:
            pred_img_tensor = pipe('', guidance_scale=1.0, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm).images
        else:
            pred_img_tensor = pipe(prompt, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm).images
        loss = totalLoss(pred_img_tensor, gt_img_tensor, init_latents_wm, wm_pipe)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_lst.append(loss.item())
        # save watermarked image
        if (iter+1) in cfgs['save_iters']:
            save1 = os.path.join(save_path1,f"{id}.png")
            print("save1 : ",save1)
            save_img(save1, pred_img_tensor, pipe)
            print(f"Watermarked image {img} saved at : {save1}")
    torch.cuda.empty_cache()
    #####################################################################################################

    ## Postprocessing with Adaptive Enhancement
    # hyperparameter
    ssim_threshold = cfgs['ssim_threshold']

    # wm_img_path = os.path.join(wm_path, f"{img.split('.')[0]}_{cfgs['save_iters'][-1]}.png")
    wm_img_path = save1
    print("wm_img_path : ", wm_img_path)

    wm_img_tensor = get_img_tensor(wm_img_path, device)
    ssim_value = ssim(wm_img_tensor, gt_img_tensor).item()
    
    logging.info(f'Original SSIM {ssim_value}')

    optimal_theta = binary_search_theta(ssim_threshold, precision=0.01)
    logging.info(f'Optimal Theta {optimal_theta}')

    img_tensor = (gt_img_tensor-wm_img_tensor)*optimal_theta+wm_img_tensor

    ssim_value = ssim(img_tensor, gt_img_tensor).item()
    psnr_value = compute_psnr(img_tensor, gt_img_tensor)

    tester_prompt = '' 
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    det_prob = 1 - watermark_prob(img_tensor, pipe, wm_pipe, text_embeddings)

    save2 = os.path.join(save_path2, f"{id}_SSIM{ssim_threshold}.png")
    # save2 = os.path.join(wm_path, f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold}.png")
    print("save 2 : ", save2  )
    save_img(save2, img_tensor, pipe)
    logging.info(f'SSIM {ssim_value}, PSNR, {psnr_value}, Detect Prob: {det_prob} after postprocessing')
    ##############################################################################################################
    detection_probs[img] = det_prob

    # count = count + 1 


print("Saving detection probabilites in json")

with open("og_zodiac_watermarked_probs.json","w") as file : 
    json.dump(detection_probs,file,indent=4)

print("Finised")