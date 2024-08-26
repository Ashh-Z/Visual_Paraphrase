# %cd hidden
import os 
import site 
# import sys 
# os.chdir('hidden')

# # Add the project root to the site-packages
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# site.addsitedir(project_root)



# try:
#     os.chdir('hidden')
#     print("Directory changed successfully")
#     print("New Working Directory: ", os.getcwd())
# except FileNotFoundError as e:
#     print(f"Error: {e}")
# except NotADirectoryError as e:
#     print(f"Error: {e}")
# except PermissionError as e:
#     print(f"Error: {e}")

os.environ["CUDA_VISIBLE_DEVICES"]='0'

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# you should run this notebook in the root directory of the hidden project for the following imports to work
# %cd ..
from models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
from attenuations import JND
from PIL import Image 
import numpy as np 
from tqdm import tqdm 
from skimage.util import random_noise
from torchvision import transforms
from collections import defaultdict
import re
import cv2
from PIL import Image, ImageEnhance
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
import json


def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]


class Params():
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int,
                attenuation:str, scale_channels:bool, scaling_i:float, scaling_w:float):
        # encoder and decoder parameters
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        # attenuation parameters
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])

params = Params(
    encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
    attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
)

decoder = HiddenDecoder(
    num_blocks=params.decoder_depth,
    num_bits=params.num_bits,
    channels=params.decoder_channels
)
encoder = HiddenEncoder(
    num_blocks=params.encoder_depth,
    num_bits=params.num_bits,
    channels=params.encoder_channels
)
attenuation = JND(preprocess=UNNORMALIZE_IMAGENET) if params.attenuation == "jnd" else None
encoder_with_jnd = EncoderWithJND(
    encoder, attenuation, params.scale_channels, params.scaling_i, params.scaling_w
)


ckpt_path = "ckpts/hidden_replicate.pth"

state_dict = torch.load(ckpt_path, map_location='cpu')['encoder_decoder']
encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'encoder' in k}
decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}

encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)

encoder_with_jnd = encoder_with_jnd.to(device).eval()
decoder = decoder.to(device).eval()


paraphrase_model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
# model_id = 'stabilityai/stable-diffusion-2-1-base'

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(paraphrase_model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False).to(device)
pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to(device)


img_dir = "/raid/home/ashhar21137/watermarking_final_tests/stable_signature/stable_sig_watermarked"


with open('/raid/home/ashhar21137/watermarking3/stable_signature/captions_train2014_new_format.json','r') as file : 
    new = json.load(file)


img_ids = []
img_pths = []
imgs = os.listdir(img_dir)
for i in imgs : 
    if '.png' or '.jpg' in i :
        id = re.split('[_.]',i)[1]
        img_ids.append(id) 
        img_pths.append(os.path.join(img_dir,i))

print("Imgs ids : ", img_ids)

print("Images : ",imgs)

cnt = 0 
for i in range(len(img_ids)) : 
    id = re.split('[_.]',imgs[i])[1]
    if(img_ids[i] != id ) : 
        print(img_ids)
        cnt += 1

print(cnt)

if(cnt == 0) : 
    print("All fine, please proceed")
else : 
    print("Ordering does not match ")
    quit() 


wmis = os.listdir(img_dir) 
print("Wmis : ",wmis)

save_dir = "/raid/home/ashhar21137/watermarking_final_tests/stable_signature/stable_signature_paraphrases"


def load_image(image_path):
    """
    Load an image from the specified file path using PIL.

    Args:
    image_path (str): Path to the image file.

    Returns:
    PIL.Image.Image: Loaded image object.
    """
    return Image.open(image_path)



strength_values = [0.10, 0.20, 0.30, 0.40, 0.50]

for strength in strength_values : 
    count = 1

    print(f'----- Strength = {strength} -----')
    paraphrase_detection = defaultdict(lambda: defaultdict(dict))

    for i in tqdm(range(len(wmis))) :

        if count > 1 : break 

        print(f"---------- Count = {count} : Visual Paraphrasing for the watermarked version of {img_ids[i]}-------------")
        print()

        # print(f"records_512_st_0.35/{img_ids[0]}.txt")
        print("Image name : ",wmis[i])
        
        id = re.split('[_.]',wmis[i])[1]

        print("Image id :",id)
        # print("Image id : ",img_ids[i])

        paraphrase_detection[id]["name"] = wmis[i]

        captions = new['annotations'][id]


        post_img = os.path.join(img_dir,wmis[i]) # watermarked image path 
        image = Image.open(post_img)
        # init_image = load_image(image)
        init_image = image

        gen_image = pipeline(captions, image=init_image, strength=strength, guidance_scale=7.5).images
        
        # image
        # make_image_grid([init_image, image], rows=2, cols=2)

        directory =  f'{save_dir}/{strength}/{id}'
        print(f"Saving generated images at {directory}")
        if not os.path.exists(directory):
            os.makedirs(directory)

        for k in range(len(gen_image)) :
            gen_save_dir = os.path.join(directory,f'{id}_gen_{k}.png')
            gen_image[k].save(gen_save_dir)
            print(f"Generated Image saved to {gen_save_dir}")

        # is_watermarked
        num_images = len(gen_image) # 5 images generated from 5 captions for each input image
        print()
        print("Number of images generated : ",num_images)
        print("********** Watermark detection for generated images without Captions ***************")     

        # avg_no_prompt = 0 
        # num_detected = 0 
        # for j in range(num_images):
        #     post_img = os.path.join(directory,f'{id}_gen_{j}.png')
        #     tester_prompt = '' # assume at the detection time, the original prompt is unknown
        #     text_embeddings = pipe.get_text_embedding(tester_prompt)
        #     det_prob = 1 - watermark_prob(post_img, pipe, wm_pipe, text_embeddings)
            
        #     avg_no_prompt = avg_no_prompt + det_prob
        #     num_detected = num_detected = (det_prob>0.9)
            
        #     # logging.info(f'Watermark Presence Prob.: {det_prob}')
        #     print(f'Watermark Presence Prob.: {det_prob}')

        # paraphrase_detection[id]["without_captions_avg_det_prob"] = avg_no_prompt/num_images
        # paraphrase_detection[id]["without_captions_det_rate"] = num_detected/num_images


        avg_with_prompt = 0 
        num_detected = 0


        print()
        print("********** Watermark detection for generated images ***************")  

        for j in range(num_images):
            post_img = os.path.join(directory, f'{id}_gen_{j}.png')
            # tester_prompt = '' # assume at the detection time, the original prompt is unknown
            print(f"caption : {captions[j]}")

            # Load the image
            img = Image.open(post_img).convert('RGB')
            img = img.resize((512, 512), Image.BICUBIC)
            img_pt = default_transform(img).unsqueeze(0).to(device)
            
            # Decode the watermark
            ft = decoder(img_pt)
            decoded_msg = ft > 0  # b k -> b k
            
            # Assume you have the original message or compute its bit accuracy
            msg_ori = torch.Tensor(str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0).to(device)  # Example message
            accs = (~torch.logical_xor(decoded_msg, msg_ori))  # b k -> b k
            bit_accuracy = accs.sum().item() / params.num_bits

            print(f"Message: {msg2str(msg_ori.squeeze(0).cpu().numpy())}")
            print(f"Decoded: {msg2str(decoded_msg.squeeze(0).cpu().numpy())}")
            print(f"Bit Accuracy: {bit_accuracy}")
            
            detection_result = 1 if bit_accuracy > 0.90 else 0
            avg_with_prompt = avg_with_prompt + bit_accuracy
            num_detected = num_detected + detection_result

            print(f'Watermark Presence Prob.: {bit_accuracy}')    
            paraphrase_detection[id]["avg_det_prob"] = avg_with_prompt/num_images
            paraphrase_detection[id]["avg_det_rate"] = num_detected/num_images

        # break
        count = count + 1  


    with open(f"/raid/home/ashhar21137/watermarking_final_tests/stable_signature/stable_signature_paraphrases/stable_signature_{strength}_paraphrased.json","w") as file : 
        json.dump(paraphrase_detection,file,indent=4)


# if project_root in sys.path:
#     sys.path.remove(project_root)


