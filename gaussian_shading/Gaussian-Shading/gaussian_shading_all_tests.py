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


os.environ["CUDA_VISIBLE_DEVICES"]= "5"
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
model_path = "stabilityai/stable-diffusion-2-1-base"
channel_copy = 1 
hw_copy = 8 
fpr = 0.000001
num = 1000
user_number = 1000000
output_path = "/raid/home/ashhar21137/watermarking2/Gaussian-Shading/gs_watermarked_images_2"
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
    output_path = "/raid/home/ashhar21137/watermarking2/Gaussian-Shading/watermarked_images",
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
    output_path = "/raid/home/ashhar21137/watermarking2/Gaussian-Shading/watermarked_images",
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
    output_path = "/raid/home/ashhar21137/watermarking2/Gaussian-Shading/watermarked_images",
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
    output_path = "/raid/home/ashhar21137/watermarking2/Gaussian-Shading/watermarked_images",
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

attacks = ["brightness","gaussian_noise","jpeg","rotation"]
attacks_op_parent = f"/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/gaussian_shading_attacked/"

paraphrase_model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
pipeline_text2image = AutoPipelineForText2Image.from_pretrained(paraphrase_model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False).to(device)
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to(device)

def select_random_excluding_index(lst, i):
    # Remove the element at the i-th index
    excluded_list = lst[:i] + lst[i+1:]
    # Select a random element from the remaining list
    selected_element = random.choice(excluded_list)
    return selected_element

og_detection_dict = defaultdict(lambda: defaultdict(dict))

attack_detection = defaultdict(lambda: defaultdict(dict))
# paraphrase_detection = defaultdict(lambda: defaultdict(lambda: {'avg_probability': 0, 'detection_rate': 0}))
paraphrase_detection = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'avg_probability': 0, 'detection_rate': 0}))))

count = 1 
strength_values = [0.2,0.4,0.6,0.8,1.0]

for id in tqdm(img_ids):
    if count > 1 : break 

    print(f"******** Count : {count} ********")

    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    #reference model for CLIP Score
    if reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(reference_model,
                                                                                    pretrained=reference_model_pretrain,
                                                                                    device=device)
        ref_tokenizer = open_clip.get_tokenizer(reference_model)

    # dataset
    # dataset, prompt_key = get_dataset(args)

    # class for watermark
    if chacha:
        watermark = Gaussian_Shading_chacha(channel_copy, hw_copy, fpr, user_number)
    else:
        #a simple implement,
        watermark = Gaussian_Shading(channel_copy, hw_copy, fpr, user_number)

    os.makedirs(output_path, exist_ok=True)

    # assume at the detection time, the original prompt is unknown
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    #acc
    acc = []
    #CLIP Scores
    clip_scores = []

    captions = data['annotations'][id]

    print(f"len captions : {len(captions)}")

    for attack in attacks: 
        attack_detection[id][attack]['avg_probability'] = 0 
        attack_detection[id][attack]['detection_rate'] = 0 

    #test
    for i in tqdm(range(len(captions))):
        seed = i + gen_seed
        current_prompt = captions[i]

        #generate with watermark
        set_random_seed(seed)
        init_latents_w = watermark.create_watermark_and_return_w()
        outputs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=image_length,
            width=image_length,
            latents=init_latents_w,
        )
        image_w = outputs.images[0]

        watermarked_image = image_w

        if not isinstance(watermarked_image, Image.Image):
            watermarked_image = Image.fromarray(image_w)

        directory = f'/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/gs_watermarked_images/{id}'

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = f'{id}_watermarked_caption_{i}.png'
        watermarked_image.save(os.path.join(directory, filename))
        print(f"Image saved to {os.path.join(directory, filename)}")

        # image_w_distortion = image_distortion(image_w, seed, args)

        image_og = image_w

        # reverse img
        image_og = transform_img(image_og).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(image_og, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=num_inversion_steps,
        )

        #acc metric
        acc_metric = watermark.eval_watermark(reversed_latents_w)
        print("acc_metric : ", acc_metric)
        acc.append(acc_metric)

        og_detection_dict[id][f"caption{i}"] = acc_metric

        for attack in attacks:
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
                num_inference_steps=num_inversion_steps,
            )

            #acc metric
            acc_metric = watermark.eval_watermark(reversed_latents_w)
            print(f" attack : {attack} | acc_metric : {acc_metric}")
            # acc.append(acc_metric)

            attack_detection[id][attack]['avg_probability'] += acc_metric / len(captions)
            attack_detection[id][attack]['detection_rate'] += (acc_metric > 0.9) / len(captions)

        # SAMPLE PARAPHRASING
        # paraphrase_caption = captions[i]
        paraphrase_caption = select_random_excluding_index(captions,i)

        for strength in strength_values :
            paraphrase_detection[strength][id]['with_caption']['avg_probability'] = 0
            paraphrase_detection[strength][id]['with_caption']['detection_rate'] = 0
            paraphrase_detection[strength][id]['without_caption']['avg_probability'] = 0
            paraphrase_detection[strength][id]['without_caption']['detection_rate'] = 0
            gen_image = pipeline(paraphrase_caption, image=image_w, strength=strength, guidance_scale=7.5).images

            directory_paraphrased = f'/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/gs_paraphrases/{strength}/{id}'
            print(f"Saving generated images at {directory_paraphrased}")
            if not os.path.exists(directory_paraphrased):
                os.makedirs(directory_paraphrased)

            paraphrased_name = f'gen_{id}_{i}.png'
            # print(len(gen_image))

            gen_image = gen_image[0]

            gen_image.save(os.path.join(directory_paraphrased, paraphrased_name))
            print(f" Paraphrased Image saved to {os.path.join(directory_paraphrased, paraphrased_name)}")

            # Ensure gen_image is a PIL Image
            if not isinstance(gen_image, Image.Image):
                gen_image = Image.fromarray(gen_image)

            # *** Testing with captions **
            tester_prompt = ''
            text_embeddings = pipe.get_text_embedding(paraphrase_caption)

            gen_image1 = transform_img(gen_image).unsqueeze(0).to(text_embeddings.dtype).to(device)
            gen_image_latents_w = pipe.get_image_latents(gen_image1, sample=False)
            gen_reversed_latents_w = pipe.forward_diffusion(
                latents=gen_image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=num_inversion_steps,
            )

            #acc metric
            gen_acc_metric = watermark.eval_watermark(gen_reversed_latents_w)
            print(f" attack : Paraphrase | acc_metric : {gen_acc_metric}")
            # acc.append(acc_metric)

            paraphrase_detection[strength][id]['with_caption']['avg_probability'] += gen_acc_metric / len(captions)
            paraphrase_detection[strength][id]['with_caption']['detection_rate'] += (gen_acc_metric > 0.9) / len(captions)

            # *** Testing without captions **
            tester_prompt = ''
            text_embeddings = pipe.get_text_embedding(tester_prompt)

            gen_image2 = transform_img(gen_image).unsqueeze(0).to(text_embeddings.dtype).to(device)
            gen_image_latents_w = pipe.get_image_latents(gen_image2, sample=False)
            gen_reversed_latents_w = pipe.forward_diffusion(
                latents=gen_image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=num_inversion_steps,
            )

            #acc metric
            gen_acc_metric = watermark.eval_watermark(gen_reversed_latents_w)
            print(f" attack : Paraphrase | acc_metric : {gen_acc_metric}")
            # acc.append(acc_metric)

            paraphrase_detection[strength][id]['without_caption']['avg_probability'] += gen_acc_metric / len(captions)
            paraphrase_detection[strength][id]['without_caption']['detection_rate'] += (gen_acc_metric > 0.9) / len(captions)

    # break 

    count = count + 1 
    # break

print('Writting results in json : ')
for strength in strength_values : 
    with open(f'/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/results/gs_paraphrase_detection_strength_{strength}.json','w') as file : 
        json.dump(paraphrase_detection[strength],file,indent=4)

with open(f'/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/results/gs_attack_detection.json','w') as file : 
    json.dump(attack_detection,file,indent=4)

print('Done')