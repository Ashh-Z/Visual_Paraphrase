import os 
os.environ["CUDA_VISIBLE_DEVICES"]='4'

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import json 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# you should run this notebook in the root directory of the hidden project for the following imports to work
# %cd ..
from models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
from attenuations import JND

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

img = Image.open("/raid/home/ashhar21137/watermarking2/original_images/Image_2024.jpg").convert('RGB')
img = img.resize((512, 512), Image.BICUBIC)
img_pt = default_transform(img).unsqueeze(0).to(device)

# create message
random_msg = False
if random_msg:
    msg_ori = torch.randint(0, 2, (1, params.num_bits), device=device).bool().to(device) # b k
else:
    msg_ori = torch.Tensor(str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0).to(device)
msg = 2 * msg_ori.type(torch.float) - 1 # b k
# encode
img_w = encoder_with_jnd(img_pt, msg)
clip_img = torch.clamp(UNNORMALIZE_IMAGENET(img_w), 0, 1)
clip_img = torch.round(255 * clip_img)/255
clip_img = transforms.ToPILImage()(clip_img.squeeze(0).cpu())
diff = np.abs(np.asarray(img).astype(int) - np.asarray(clip_img).astype(int)) / 255 * 10

# # decode
# ft = decoder(default_transform(clip_img).unsqueeze(0).to(device))
# decoded_msg = ft > 0 # b k -> b k
# accs = (~torch.logical_xor(decoded_msg, msg_ori)) # b k -> b k
# print(f"Message: {msg2str(msg_ori.squeeze(0).cpu().numpy())}")
# print(f"Decoded: {msg2str(decoded_msg.squeeze(0).cpu().numpy())}")
# print(f"Bit Accuracy: {accs.sum().item() / params.num_bits}")

input_dir = "/raid/home/ashhar21137/watermarking_final_tests/original_images"
output_dir = "/raid/home/ashhar21137/watermarking_final_tests/stable_signature/stable_sig_watermarked"
results_file = "/raid/home/ashhar21137/watermarking_final_tests/stable_signature/stable_signature/initial_stable_sig_results.json"
params = {'num_bits': 48}  # Example, adjust accordingly

if not os.path.exists(output_dir):
    os.makedirs(output_dir)




image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

results_data = []

for img_file in image_files:
    img_path = os.path.join(input_dir, img_file)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    img_pt = default_transform(img).unsqueeze(0).to(device)

    # create message
    random_msg = False
    if random_msg:
        msg_ori = torch.randint(0, 2, (1, params['num_bits']), device=device).bool().to(device)  # b k
    else:
        msg_ori = torch.Tensor(str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0).to(device)
    msg = 2 * msg_ori.type(torch.float) - 1  # b k

    # encode
    img_w = encoder_with_jnd(img_pt, msg)
    clip_img = torch.clamp(UNNORMALIZE_IMAGENET(img_w), 0, 1)
    clip_img = torch.round(255 * clip_img) / 255
    clip_img = transforms.ToPILImage()(clip_img.squeeze(0).cpu())
    
    # save watermarked image
    watermarked_img_path = os.path.join(output_dir, img_file)
    clip_img.save(watermarked_img_path)

    # decode
    ft = decoder(default_transform(clip_img).unsqueeze(0).to(device))
    decoded_msg = ft > 0  # b k -> b k
    accs = (~torch.logical_xor(decoded_msg, msg_ori))  # b k -> b k
    bit_accuracy = accs.sum().item() / params['num_bits']
    detection_result = 1 if bit_accuracy > 0.90 else 0
    
    # store results
    results_data.append({
        "image": img_file,
        "bit_accuracy": bit_accuracy,
        "detection_result": detection_result
    })
    print(f"Processed {img_file}: Bit Accuracy = {bit_accuracy:.4f}, Detection Result = {detection_result}")

# Save results to JSON file
with open(results_file, 'w') as json_file:
    json.dump(results_data, json_file, indent=4)

print("Watermarking completed. Results saved to", results_file)
