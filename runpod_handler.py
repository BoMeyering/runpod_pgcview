"""
runpod_handler.py
RunPod.io serverless handler script
BoMeyering 2025
"""

import runpod
import time
import torch
import os
import sys
import numpy as np
from pathlib import Path
from torchvision.transforms import ToTensor
from src.models import load_model
from src.utils import decode_img, encode_bbox_arr, encode_out_map
from src.transforms import get_marker_transforms, get_seg_transforms

# Get model paths
effdet_model_path = os.getenv('EFFDET_MODEL_PATH')
dlv3p_model_path = os.getenv('DLV3P_MODEL_PATH')

if effdet_model_path is None or dlv3p_model_path is None:
    sys.exit("Model paths were not defined in the ENV")

# Set the torch device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Set Tensor transforms
effdet_transforms = get_marker_transforms()
dlv3p_transforms = get_seg_transforms()

# Load models
try:
    effdet_model = load_model(file_path=effdet_model_path, device=device)['model']
    dlv3p_model = load_model(file_path=dlv3p_model_path, device=device)['model']

    effdet_model = effdet_model.to(device)
    dlv3p_model = dlv3p_model.to(device)
except Exception as e:
    sys.exit(f"Call to load_model() failed: {str(e)}")

def handler(request):
    """ Handles API requests and processes them """

    print(f"[INFO] Received request at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()

    input_data = request.get('input', {})
    b64_img = input_data.get('image')

    # Return error if image is missing
    if b64_img is None:
        return {
            "errors": "Missing 'image' key in request payload."
        }

    # Decode image and collect errors
    img_dict = decode_img(b64_img)
    if img_dict.get('errors') is not None:
        return {
            'errors': img_dict.get('errors')
        }
    
    # Create a numpy array image
    rgb_img = np.array(img_dict.get('image'))

    # Create normalized, tensors for each model
    effdet_tensor = effdet_transforms(image=rgb_img)['image'].unsqueeze(0).to(device)
    dlv3p_tensor = dlv3p_transforms(image=rgb_img)['image'].unsqueeze(0).to(device)

    # Forward pass for effdet model
    try:
        with torch.no_grad():
            effdet_out = effdet_model(effdet_tensor).squeeze().detach().cpu().numpy()
            effdet_dict = encode_bbox_arr(effdet_out)
            
            if effdet_dict.get('errors') is not None:
                return {
                    'errors': effdet_dict.get('errors')
                }
    except Exception as e:
        error_msg = f"Server encountered error processing the image through the EffDet model: {str(e)}"
        return {
            'errors': error_msg
        }
    
    p_time = time.time() - t0
    

    return {
        'effdet': effdet_dict,
        'server_inference_time': p_time
    }

# Start the Serverless function when the container runs the script
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})