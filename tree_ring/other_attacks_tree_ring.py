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
from copy import deepcopy
model_id = 'stabilityai/stable-diffusion-2-1-base'



if torch.cuda.is_available() : 
    device = "cuda"
else :
    device = "cpu"

print("device : ",device)

def load_keys(cache_dir):
    arrays = {}
    for file_name in os.listdir(cache_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(cache_dir, file_name)
            arrays[file_name] = np.load(file_path)
    return arrays


scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
pipeline1 = DiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
# pipeline2 = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline1 = pipeline1.to(device)


def _circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size))
    y = torch.flip(y, dims=[0])

    return ((x - x0)**2 + (y-y0)**2) <= r**2

def _get_pattern(shape, w_pattern='rand', generator=None):
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


parent = '/raid/home/ashhar21137/watermarking_final_tests/tree_ring/tr_sample_RING'

ids = os.listdir(parent)
# ids

attacks = ["brightness","gaussian_noise","jpeg","rotation"]
attacks_op_parent = "/raid/home/ashhar21137/watermarking_final_tests/tree_ring/tree_ring_attacked"

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
            img_path = os.path.join(id_pth,image)

            img = Image.open(img_path)
            is_watermarked, probability = detect(img, pipeline1)

            print(f"{attack} : {id} : {image} : Prob = {probability}")
            print(f"{attack} : {id} : {image} : Watermarked? = {is_watermarked}")

            id_prob += probability
            detection += is_watermarked

        print(id_prob/len(images))
        print(detection/len(images))

        attack_detection[id][attack]['avg_probability'] = id_prob/len(images)
        attack_detection[id][attack]['detection_rate'] = detection/len(images)

    

print("Preparing results json ")
import json 
with open('tree_ring_attacked_results_new.json','w') as file : 
    json.dump(attack_detection,file,indent=4)
