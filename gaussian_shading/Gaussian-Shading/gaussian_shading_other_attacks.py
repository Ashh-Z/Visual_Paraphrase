import argparse
import copy
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPTokenizer
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *
import re
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
import random
from types import SimpleNamespace
from collections import defaultdict


os.environ["CUDA_VISIBLE_DEVICES"]= "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

with open('/raid/home/ashhar21137/watermarking2/captions_train2014_new_format.json','r') as file : 
    data = json.load(file)

img_dir = '/raid/home/ashhar21137/watermarking_final_tests/original_images'
img_ids = []
img_pths = []
imgs = os.listdir(img_dir)
for i in imgs : 
    id = re.split('[_.]',i)[1]
    img_ids.append(id) 
    img_pths.append(os.path.join(img_dir,i))

print(f'img_ids : {img_ids}')


reference_model = None 
reference_model_pretrain = None 
# model_path = "stabilityai/stable-diffusion-2-1-base"
channel_copy = 1 
hw_copy = 8 
fpr = 0.000001
num = 1000
user_number = 1000000
output_path = "/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/gaussian_shading_attacked"
gen_seed = 0 
num_inference_steps = 50 
guidance_scale = 7.5
image_length = 512
chacha = True
num_inversion_steps = 50

args_brightness = SimpleNamespace(
    reference_model = None ,
    reference_model_pretrain = None ,
    model_path = "stabilityai/stable-diffusion-2-1-base",
    channel_copy = 1 ,
    hw_copy = 8 ,
    fpr = 0.000001,
    num = 1000,
    user_number = 1000000,
    output_path = "/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/gaussian_shading_attacked",
    gen_seed = 0 ,
    num_inference_steps = 50 ,
    guidance_scale = 7.5,
    image_length = 512,
    chacha = True,
    num_inversion_steps = 50,
    jpeg_ratio = None,
    random_crop_ratio =None ,
    random_drop_ratio = None, 
    gaussian_blur_r = None, 
    median_blur_k = None, 
    resize_ratio = None, 
    gaussian_std = None, 
    sp_prob = None, 
    brightness_factor = 2,
    rotate_deg = None 
)

args_jpeg = SimpleNamespace(
    reference_model = None ,
    reference_model_pretrain = None ,
    model_path = "stabilityai/stable-diffusion-2-1-base",
    channel_copy = 1 ,
    hw_copy = 8 ,
    fpr = 0.000001,
    num = 1000,
    user_number = 1000000,
    output_path = "/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/gaussian_shading_attacked",
    gen_seed = 0 ,
    num_inference_steps = 50 ,
    guidance_scale = 7.5,
    image_length = 512,
    chacha = True,
    num_inversion_steps = 50,
    jpeg_ratio = 50,
    random_crop_ratio =None ,
    random_drop_ratio = None, 
    gaussian_blur_r = None, 
    median_blur_k = None, 
    resize_ratio = None, 
    gaussian_std = None, 
    sp_prob = None, 
    brightness_factor = None,
    rotate_deg = None
)

args_gaussian_noise = SimpleNamespace(
    reference_model = None ,
    reference_model_pretrain = None ,
    model_path = "stabilityai/stable-diffusion-2-1-base",
    channel_copy = 1 ,
    hw_copy = 8 ,
    fpr = 0.000001,
    num = 1000,
    user_number = 1000000,
    output_path = "/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/gaussian_shading_attacked",
    gen_seed = 0 ,
    num_inference_steps = 50 ,
    guidance_scale = 7.5,
    image_length = 512,
    chacha = True,
    num_inversion_steps = 50,
    jpeg_ratio = None,
    random_crop_ratio =None ,
    random_drop_ratio = None, 
    gaussian_blur_r = None, 
    median_blur_k = None, 
    resize_ratio = None, 
    gaussian_std = 0.1, 
    sp_prob = None, 
    brightness_factor = None,
    rotate_deg = None
)

args_rotate = SimpleNamespace(
    reference_model = None ,
    reference_model_pretrain = None ,
    model_path = "stabilityai/stable-diffusion-2-1-base",
    channel_copy = 1 ,
    hw_copy = 8 ,
    fpr = 0.000001,
    num = 1000,
    user_number = 1000000,
    output_path = "/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/gaussian_shading_attacked",
    gen_seed = 0 ,
    num_inference_steps = 50 ,
    guidance_scale = 7.5,
    image_length = 512,
    chacha = True,
    num_inversion_steps = 50,
    jpeg_ratio = None,
    random_crop_ratio =None ,
    random_drop_ratio = None, 
    gaussian_blur_r = None, 
    median_blur_k = None, 
    resize_ratio = None, 
    gaussian_std = None, 
    sp_prob = None, 
    brightness_factor = None,
    rotate_deg = 30
)


def select_random_excluding_index(lst, i):
    # Remove the element at the i-th index
    excluded_list = lst[:i] + lst[i+1:]
    # Select a random element from the remaining list
    selected_element = random.choice(excluded_list)
    return selected_element


count = 1 
print("----------------- Performing Attacks ------------------")

model_path = "stabilityai/stable-diffusion-2-1-base"
watermarked_images_parent = "/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/gs_watermarked_images"
attacks_op_parent = "/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/gaussian_shading_attacked"
attacks = ["brightness", "gaussian_noise", "rotation", "jpeg"]

# Initialize the pipeline and watermark
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')
pipe = InversableStableDiffusionPipeline.from_pretrained(
    model_path,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    revision='fp16',
).to(device)
pipe.safety_checker = None

# Initialize watermark (adjust parameters as needed)
# class for watermark
if chacha:
    watermark = Gaussian_Shading_chacha(channel_copy, hw_copy, fpr, user_number)
else:
    #a simple implement,
    watermark = Gaussian_Shading(channel_copy, hw_copy, fpr, user_number)

attack_detection = defaultdict(lambda: defaultdict(dict))

for id in tqdm(os.listdir(watermarked_images_parent)):
    id_path = os.path.join(watermarked_images_parent, id)
    
    for i, image in enumerate(os.listdir(id_path)):
        if image.endswith('.png'):
            img_path = os.path.join(id_path, image)
            image_w = Image.open(img_path)
            
            # Get text embeddings (empty prompt)
            text_embeddings = pipe.get_text_embedding("")

            for attack in attacks:
                seed = random.randint(0, 1000000)  # Generate a random seed for each attack
                
                if attack == "brightness":
                    image_w_distortion = image_distortion(image_w, seed, args_brightness)
                elif attack == "jpeg":
                    image_w_distortion = image_distortion(image_w, seed, args_jpeg)
                elif attack == "gaussian_noise":
                    image_w_distortion = image_distortion(image_w, seed, args_gaussian_noise)
                elif attack == "rotation":
                    image_w_distortion = image_distortion(image_w, seed, args_rotate)

                print(f"attack : {attack}")

                if not isinstance(image_w_distortion, Image.Image):
                    image_w_distortion = Image.fromarray(image_w_distortion)

                # Create the directory if it doesn't exist
                att_save = f"{attacks_op_parent}/{attack}/{id}"
                if not os.path.exists(att_save):
                    os.makedirs(att_save)

                filename_att = f'{attack}_{id}_watermarked_caption_{i}.png'
                image_w_distortion.save(os.path.join(att_save, filename_att))
                print(f" {attack} Image saved to {os.path.join(att_save, filename_att)}")

                image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
                image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
                reversed_latents_w = pipe.forward_diffusion(
                    latents=image_latents_w,
                    text_embeddings=text_embeddings,
                    guidance_scale=1,
                    num_inference_steps=50,  # Adjust as needed
                )

                # acc metric
                acc_metric = watermark.eval_watermark(reversed_latents_w)
                print(f" attack : {attack} | acc_metric : {acc_metric}")

                if id not in attack_detection[attack]:
                    attack_detection[attack][id] = {'avg_probability': 0, 'detection_rate': 0, 'count': 0}
                attack_detection[attack][id]['avg_probability'] += acc_metric
                attack_detection[attack][id]['detection_rate'] += (acc_metric > 0.9)
                attack_detection[attack][id]['count'] += 1

# Calculate final averages
for attack in attack_detection:
    for id in attack_detection[attack]:
        count = attack_detection[attack][id]['count']
        attack_detection[attack][id]['avg_probability'] /= count
        attack_detection[attack][id]['detection_rate'] /= count
        del attack_detection[attack][id]['count']

print("Preparing results json")
with open('gs_attacked_results.json', 'w') as file:
    json.dump(attack_detection, file, indent=4)
