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
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
import random
import json 
import re
from copy import deepcopy


if torch.cuda.is_available : 
    device = 'cuda'
else : 
    device = 'cpu'

print(device)

model_id = 'stabilityai/stable-diffusion-2-1-base'


paraphrase_model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
pipeline_text2image = AutoPipelineForText2Image.from_pretrained(paraphrase_model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False).to(device)
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to(device)

with open('captions_train2014_new_format.json','r') as file : 
    data = json.load(file)

watermark_img_dir = "/raid/home/ashhar21137/watermarking_final_tests/tree_ring/tr_sample_RING"
wm_images = os.listdir(watermark_img_dir)
print(wm_images)


img_ids = wm_images
print(img_ids)

img_dir = watermark_img_dir
# img_ids = []
img_pths = []
imgs = os.listdir(img_dir)
for i in imgs : 
    # id = re.split('[_.]',i)[1]
    # img_ids.append(id) 
    img_pths.append(os.path.join(img_dir,i))

print(f'img_ids : {img_ids}')
print("paths : ",img_pths)

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


# print(data['annotations'])

# random.seed(42)

def select_random_excluding_index(lst, i):
    # Remove the element at the i-th index
    excluded_list = lst[:i] + lst[i+1:]
    # Select a random element from the remaining list
    selected_element = random.choice(excluded_list)
    # Find the index of the selected element in the original list
    original_index = lst.index(selected_element)
    return selected_element, original_index


def list_excluding_index(lst, i):
    # Return a list excluding the element at the i-th index
    return lst[:i] + lst[i+1:]

scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
pipeline1 = DiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
# pipeline2 = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline1 = pipeline1.to(device)


# paraphrase_detection = defaultdict(lambda: defaultdict(dict))


strength_values = [0.20,0.40,0.60,0.80,1.0]
save_dir = '/raid/home/ashhar21137/watermarking_final_tests/tree_ring/ring_paraphrased'

from diffusers import StableDiffusionXLImg2ImgPipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16
)
pipe = pipe.to(device)

neg_prompt = 'deformity, bad anatomy, cloned face, amputee, people in background, asymmetric, disfigured, extra limbs, text, missing legs, missing arms, Out of frame, low quality, Poorly drawn feet'

# neg_prompt = ['deformity', 'bad anatomy', 'cloned face', 'amputee', 'people in background', 'asymmetric', 'disfigured', 'extra limbs', 'text', 'missing legs', 'missing arms', 'Out of frame', 'low quality', 'Poorly drawn feet']


for strength in strength_values: 
    print(f"---- Strength = {strength} -----")
    paraphrase_detection = defaultdict(lambda: defaultdict(dict))
    # count = 1 
    for id in wm_images: 
        # if count > 100: 
        #     break 

        id_dir = os.path.join(watermark_img_dir, id)

        avg_prob_id = 0 
        avg_rate_id = 0

        id_imgs = os.listdir(id_dir)

        captions = data['annotations'][id]

        for i in range(len(id_imgs)):
            image_path = os.path.join(id_dir, id_imgs[i])

            gen_caption_idx = re.split('[_.]', id_imgs[i])[3]
            paraphrase_caption,_ = select_random_excluding_index(captions, int(gen_caption_idx))
            print(paraphrase_caption)

            wm_image = Image.open(image_path)

            # for para_cap in paraphrase_captions:
            #     print(para_cap)
            gen_image = pipeline(paraphrase_caption, image=wm_image, negative_prompt=neg_prompt, strength=strength, guidance_scale=7.5).images[0]
            # gen_image = pipe(paraphrase_caption, image=wm_image, negative_prompt = neg_prompt, strength=strength, guidance_scale=7.5).images[0]

            directory_paraphrased = f'{save_dir}/{strength}_para/{id}'
            print(f"Saving generated images at {directory_paraphrased}")
            if not os.path.exists(directory_paraphrased):
                os.makedirs(directory_paraphrased)

            paraphrased_name = f'gen_{id}_{i}.png'
            gen_image.save(os.path.join(directory_paraphrased, paraphrased_name))
            print(f"Paraphrased Image saved to {os.path.join(directory_paraphrased, paraphrased_name)}")

            is_watermarked, probability = detect(gen_image, pipeline1)
            print(f"Watermarked: {is_watermarked} || Probability: {probability}")
            
            avg_prob_id += probability
            avg_rate_id += is_watermarked

        # Now compute the averages after processing all images for the current id
        num_images = len(id_imgs)
        paraphrase_detection[id]["avg_prob"] = avg_prob_id / num_images
        paraphrase_detection[id]["avg_det"] = avg_rate_id / num_images

        # count += 1 

    with open(f"/raid/home/ashhar21137/watermarking_final_tests/tree_ring/ring_paraphrased/ring_{strength}_paraphrased.json", "w") as file:
        json.dump(paraphrase_detection, file, indent=4)


