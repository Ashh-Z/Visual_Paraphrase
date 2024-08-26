import os 
os.environ["CUDA_VISIBLE_DEVICES"]='3'

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

parent = '/raid/home/ashhar21137/watermarking_final_tests/stable_signature/stable_sig_watermarked'

ids = os.listdir(parent)

attacks = ["brightness","gaussian_noise","jpeg","rotation"]
attacks_op_parent = "/raid/home/ashhar21137/watermarking_final_tests/stable_signature/stable_signature_attacked"


print("----------------- Performing Attacks ------------------")
count = 0 
for image in tqdm(ids):
    # if count > 0 : break 
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
        

    # count += 1

original_message = "111010110101000001010111010011010100010000100111"
attack_detection = defaultdict(lambda: defaultdict(dict))

for attack in attacks:
    # if attack != "gaussian_noise" : 
        # continue 
    
    print(f" ----- Attack : {attack} -----")
    
    attacked_path = os.path.join(attacks_op_parent, attack)
    attacked_ids = os.listdir(attacked_path)
    
    for img in tqdm(attacked_ids):
        print(f" *** Image : {img} ***")
        l = img.split('_')
        if attack == "gaussian_noise":
            id = f"{l[2]}_{l[3]}"
            print("Image id : ", id)
        
        else :
            id = f"{l[1]}_{l[2]}"
            print("Image id : ", id)
        
        image_path = os.path.join(attacked_path, img)
        img = Image.open(image_path).convert('RGB')
        img = img.resize((512, 512), Image.BICUBIC)
        img_pt = default_transform(img).unsqueeze(0).to(device)
        
        # create message
        # random_msg = False
        # if random_msg:
        #     msg_ori = torch.randint(0, 2, (1, params.num_bits), device=device).bool().to(device) # b k
        # else:
        #     msg_ori = torch.Tensor(str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0).to(device)
        
        # msg = 2 * msg_ori.type(torch.float) - 1  # b k
        # img_w = encoder_with_jnd(img_pt, msg)
        # clip_img = torch.clamp(UNNORMALIZE_IMAGENET(img_w), 0, 1)
        # clip_img = torch.round(255 * clip_img) / 255
        # clip_img = transforms.ToPILImage()(clip_img.squeeze(0).cpu())
        
        # diff = np.abs(np.asarray(img).astype(int) - np.asarray(clip_img).astype(int)) / 255 * 10
        # msg_ori = torch.Tensor(str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0).to(device)  # Example message
        # accs = (~torch.logical_xor(decoded_msg, msg_ori))  # b k -> b k
        # bit_accuracy = accs.sum().item() / params.num_bits
        
        ft = decoder(img_pt)
        decoded_msg = ft > 0  # b k -> b k
        msg_ori = torch.Tensor(str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0).to(device)  # Example message
        accs = (~torch.logical_xor(decoded_msg, msg_ori))  # b k -> b k
        bit_accuracy = accs.sum().item() / params.num_bits
        
        print(f"Message: {msg2str(msg_ori.squeeze(0).cpu().numpy())}")
        print(f"Decoded: {msg2str(decoded_msg.squeeze(0).cpu().numpy())}")
        print(f"Bit Accuracy: {bit_accuracy}")
        detection_result = 1 if bit_accuracy > 0.90 else 0
        
        detection_result = 1 if bit_accuracy > 0.90 else 0
        attack_detection[id][f"{attack}_det_prob"] = bit_accuracy
        attack_detection[id][f"{attack}_det_result"] = detection_result
        print("Detection Result: ", detection_result)

print("Preparing results json ")
import json 
with open('/raid/home/ashhar21137/watermarking_final_tests/stable_signature/stable_signature/stable_signature_attack_result.json','w') as file : 
    json.dump(attack_detection,file,indent=4)
