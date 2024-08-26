import os 
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import json 
import subprocess
import requests
import re 
import sys
import torch
import torchvision
from copy import deepcopy
from PIL import Image
import requests
from io import BytesIO
import tempfile
from typing import Union, List, Tuple
import numpy as np
import os
import matplotlib.pyplot as plt
import hashlib
from torchvision import transforms
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline, DDIMScheduler
# from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DDIMInverseScheduler
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from tqdm import tqdm 




import torch
if torch.cuda.is_available() : 
    device = "cuda"
else :
    device = "cpu"

print(f"Device : {device}")

with open('captions_train2014_new_format.json','r') as file : 
    data = json.load(file)

new = data

# def download_image(url, file_name,save_dir):
#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()  # Raise an error for bad status codes
#         with open(os.path.join(save_dir,file_name), 'wb') as file:
#             for chunk in response.iter_content(1024):
#                 file.write(chunk)
#         return True
#     except requests.RequestException as e:
#         print(f"Failed to download from {url}: {e}")
#         return False
    

# save_dir = '/raid/home/ashhar21137/watermarking/tree-ring-watermark/test_samples_ring/original_images'
# count = 200
# for i in range(len(new['images'])) : 
#     id = new['images'][i]['id'] 
#     file_name = f"Image_{new['images'][i]['id']}.jpg"
#     if(not download_image(new['images'][i]['coco_url'], file_name,save_dir)) :
#         if not download_image(new['images'][i]['flickr_url'], file_name,save_dir):
#             print("Failed to download the image from both URLs")
#             continue

#     print(new['annotations'][f'{id}'])

#     # print("Use the image now")

#     # print("Delete the image now")
#     #     # Delete the downloaded image
#     # try:
#     #     os.remove(file_name)
#     # except OSError as e:
#     #     print(f"Error deleting file {file_name}: {e}")

#     count = count - 1 
#     if(count == 0 ) : 
#         break


def _circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size))
    y = torch.flip(y, dims=[0])

    return ((x - x0)**2 + (y-y0)**2) <= r**2

def _get_pattern(shape, w_pattern='ring', generator=None):
    gt_init = torch.randn(shape, generator=generator)

    if 'rand' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'ring' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        for i in range(shape[-1] // 2, 0, -1):
            tmp_mask = _circle_mask(gt_init.shape[-1], r=i)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch[0, j, 0, i].item()

    return gt_patch

def _transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def load_keys(cache_dir):
    arrays = {}
    for file_name in os.listdir(cache_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(cache_dir, file_name)
            arrays[file_name] = np.load(file_path)
    return arrays

def probability_from_distance(dist, threshold=77, k=0.1):
    raw_prob = 1 / (1 + torch.exp(k * (dist - threshold)))
    if dist > threshold:
        return 1 - raw_prob
    return raw_prob

def detect(image, pipe):
    detection_time_num_inference = 50
    prob_threshold=0.9
    keys = load_keys(tempfile.gettempdir())

    # ddim inversion
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    img = _transform_img(image).unsqueeze(0).to(pipe.unet.dtype).to(pipe.device)
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.18215
    inverted_latents = pipe(
            prompt='',
            latents=image_latents,
            guidance_scale=1,
            num_inference_steps=detection_time_num_inference,
            output_type='latent',
        )
    inverted_latents = inverted_latents.images.float().cpu()

    # check if one key matches
    shape = image_latents.shape
    for filename, w_key in keys.items():
        w_channel, w_radius = filename.split(".npy")[0].split("_")[1:3]

        np_mask = _circle_mask(shape[-1], r=int(w_radius))
        torch_mask = torch.tensor(np_mask)
        w_mask = torch.zeros(shape, dtype=torch.bool)
        w_mask[:, int(w_channel)] = torch_mask

        # calculate the distance
        inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
        dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

        prob = probability_from_distance(torch.tensor(dist))

        # If the computed probability is above the threshold, return True
        if prob > prob_threshold:
            pipe.scheduler = curr_scheduler
            return True, prob.item()

    return False, prob.item()


def get_noise(pipe):
    w_channel = 3 # id for watermarked channel
    w_radius = 10 # watermark radius
    w_pattern = 'ring' # watermark pattern
    shape = (1, 4, 64, 64)
    generator = None

    # Assuming the device is defined elsewhere in your code
    # If not, add the following line:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get watermark key and mask
    torch_mask = _circle_mask(shape[-1], r=w_radius).to(dtype=torch.bool, device=device)
    w_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    w_mask[:, w_channel] = torch_mask

    w_key = _get_pattern(shape, w_pattern=w_pattern, generator=generator).to(device)
    width = 512
    height = 512

    # Adjusted the deprecated attribute access
    in_channels = pipe.unet.config.in_channels

    init_latents = torch.randn(
            (1, in_channels, height // 8, width // 8),
            generator=generator,
            device=device
    )

    clean_latents = deepcopy(init_latents)
    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    init_latents_fft[w_mask] = w_key[w_mask].clone()
    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real
    tensor_bytes = init_latents.cpu().numpy().tobytes()

    # generate a hash from the bytes
    hash_object = hashlib.sha256(tensor_bytes)
    hex_dig = hash_object.hexdigest()

    file_name = "_".join([hex_dig, str(w_channel), str(w_radius), w_pattern]) + ".npy"
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file_name)
    np.save(file_path, w_key.cpu())

    return clean_latents, init_latents

model_id = 'stabilityai/stable-diffusion-2-1-base'
watermark_path = '/raid/home/ashhar21137/watermarking_final_tests/tree_ring/tr_sample_RING'

img_dir = '/raid/home/ashhar21137/watermarking_final_tests/original_images'
img_ids = []
img_pths = []
imgs = os.listdir(img_dir)
for i in imgs : 
    id = re.split('[_.]',i)[1]
    img_ids.append(id) 
    img_pths.append(os.path.join(img_dir,i))

print(f'img_ids : {img_ids}')

captions = []
for id in img_ids : 
    captions.append(new['annotations'][id])

original_avg_detection_prob = dict()

for i in tqdm(range(len(img_ids))) : 
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipeline1 = DiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    # pipeline2 = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline1 = pipeline1.to(device)
    # pipeline2 = pipeline2.to(device)

    print(f'Image Id : {img_ids[i]}')

    clean_latents, wm_latents = get_noise(pipeline1)
    neg_prompt = 'deformity, bad anatomy, cloned face, amputee, people in background, asymmetric, disfigured, extra limbs, text, missing legs, missing arms, Out of frame, low quality, Poorly drawn feet'
    # print(f"Prompt[i] : {prompt[i]}")

    det_prob = 0 

    for j in range(len(captions[i])) :
        with torch.autocast("cuda"):
            watermarked_image = pipeline1(
                captions[i][j],
                latents = wm_latents,
                num_inference_steps=250,
                negative_prompt = neg_prompt,
            ).images[0]

        print(f"Initial watermarked image generated for {img_ids[i]}")
        print(watermarked_image.size)
        print(watermarked_image)

        is_watermarked, probability = detect(watermarked_image, pipeline1)
        det_prob = det_prob + probability 

        print(f'is_watermarked: {is_watermarked}, probability : {probability}')
        print()

            # Ensure the image is in PIL format
        if not isinstance(watermarked_image, Image.Image):
            watermarked_image = Image.fromarray(watermarked_image)

        # print()

        # Define the directory and filename
        directory = f'{watermark_path}/{img_ids[i]}'
        

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the image to the specified directory with the specified filename
        # for copy_count in range(4) :
        # for j in range(5) : 
        filename = f'{img_ids[i]}_watermarked_caption_{j}.png'
        watermarked_image.save(os.path.join(directory, filename))
        print(f"Image saved to {os.path.join(directory, filename)}")

    original_avg_detection_prob[img_ids[i]] = det_prob/5

    # steps = 20
    # scale = 7
    # strength = 0.15
    # num_images_per_prompt = 1

    # print(f"Generating vis-paraphrased images")
    # with torch.autocast("cuda"):
    #     gen_image = pipeline2(
    #         captions[i][1:] ,
    #         latents = wm_latents,
    #         num_inference_steps=250,
    #         image=[watermarked_image]*4,
    #         negative_prompt = [neg_prompt]*4, 
    #         guidance_scale=scale, 
    #         strength=strength, 
    #         num_images_per_prompt=num_images_per_prompt, 
    #         # generator=generator
    #     ).images
    
    # print('Done')
    # print(gen_image)
    # print()
    # print(image_grid(gen_image, 2, 2))
    # print()

    # is_watermarked
    # for j in range(num_images):
    #     is_watermarked, probability = detect(gen_image[j], pipeline1)
    #     print(f'is_watermarked: {is_watermarked}, probability : {probability}')

    # directory =  f'generated_images/{img_ids[i]}'
    # print(f"Saving generated images at {directory}")
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # for k in range(len(gen_image)) :
    #     gen_save_dir = os.path.join(directory,f'{img_ids[i]}_gen_{k}.png')
    #     gen_image[k].save(gen_save_dir)
    #     print(f"Generated Image saved to {gen_save_dir}")

print('Finished, writing results to json file')

with open(r'tr_RING_initial_watermarks.json','w') as file : 
    json.dump(original_avg_detection_prob,file,indent=4)

print('done')


        
    


    
    