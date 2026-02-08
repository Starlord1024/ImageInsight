import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

# Check if GPU is enabled
import torch
import transformers
# print("CUDA available:", torch.cuda.is_available())
# print("Num GPUs Available: ", torch.cuda.device_count())

import requests
import os
import json
from transformers import pipeline

urls = [
    'https://www.ti.com/ds_dgm/images/fbd_sllsfu5a_2.gif',
    'https://www.ti.com/ds_dgm/images/fbd_scls128g.gif',
    'https://www.ti.com/ds_dgm/images/fbd_sllsfu5a.png',
    'https://www.ariat-tech.com/upfile/images/94/20231205162236766.png',
    'https://e2e.ti.com/resized-image/__size/1230x0/__key/communityserver-discussions-components-files/151/image1.jpg',
    'https://cdn.sparkfun.com/assets/learn_tutorials/1/1/8/3/Highlighted_Heatsink_Graph.jpg',
    'https://arduinodiy.wordpress.com/wp-content/uploads/2016/11/ttp233.png',
    'https://cdn.vox-cdn.com/thumbor/sKOjfrm0_K3yJTkWPGXgaOX6PPY=/1400x1400/filters:format(jpeg)/cdn.vox-cdn.com/uploads/chorus_asset/file/24933002/1689125936.jpg'
]


model_names = [
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-image-captioning-base",
    "moranyanuka/blip-image-captioning-large-mocha",
    "noamrot/FuseCap_Image_Captioning",
    "nnpy/blip-image-captioning",
    "Sof22/image-caption-large-copy",
    # "microsoft/Phi-3-vision-128k-instruct"
]

# Iterate over the list of models and generate captions
captions = {}

for url in urls:
    for model_name in model_names:
        img_captioner = pipeline(model=model_name)

        # Generate caption for the image
        caption = img_captioner(url)

        # Store the caption with the model name
        captions[model_name] = caption


    # Print the captions
    for model_name, caption in captions.items():
        print(f"Caption from {model_name}: {caption[0]['generated_text']}" , end='\n')
        print ('', end='\n')