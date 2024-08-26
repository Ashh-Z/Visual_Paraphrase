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
from types import SimpleNamespace
from collections import defaultdict


os.environ["CUDA_VISIBLE_DEVICES"]= "4"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

with open('/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/Gaussian-Shading/captions_train2014_new_format.json','r') as file : 
    data = json.load(file)


img_dir = '/raid/home/ashhar21137/watermarking2/original_images'
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
    captions.append(data['annotations'][id])

print(captions)


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

detection_dict = defaultdict(lambda: defaultdict(dict))

count = 1 
for id in tqdm(img_ids) :

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

        # distortion
        # image_w_distortion = image_distortion(image_w, seed, args)

        image_w_distortion = image_w

        # reverse img
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
        acc.append(acc_metric)

        detection_dict[id][f"caption{i}"] = acc_metric

        #CLIP Score
        if reference_model is not None:
            socre = measure_similarity([image_w], current_prompt, ref_model,
                                                ref_clip_preprocess,
                                                ref_tokenizer, device)
            clip_socre = socre[0].item()
        else:
            clip_socre = 0
        clip_scores.append(clip_socre)

    #tpr metric
    tpr_detection, tpr_traceability = watermark.get_tpr()
    #save metrics
    save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores)


    count = count + 1 
    # break


print("Preparing json file")

with open(r'/raid/home/ashhar21137/watermarking_final_tests/gaussian_shading/original_gaussian_shading_sample_results.json','w') as file : 
    json.dump(detection_dict,file,indent=4)
