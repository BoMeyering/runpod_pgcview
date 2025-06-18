"""
src/utils.py
Handler utility functions
BoMeyering 2025
"""

import os
import io
import base64
import numpy
import traceback
import numpy as np
from PIL import Image


def decode_img(b64_img: str):
    """ Decode a base64 encoded image """
    try:
        img_bytes = base64.b64decode(b64_img)
        img_buffer = io.BytesIO(img_bytes)

        # Vertfy the image
        img = Image.open(img_buffer)
        img.verify()

        # Load and decode
        img_buffer.seek(0) # Reset the pointer
        img = Image.open(img_buffer).convert('RGB')
        img_format = img.format

        return {
            'image': img,
            'format': img_format,
            'errors': None
        }
    except Exception as e:
        error_msg = f"Server encountered an error decoding the base64 image in 'decode_img()': {str(e)}"
        return {
            'image': None,
            'errors': error_msg
        }

def encode_out_map(out_map: numpy.ndarray):
    """ Encode the output map from the segmentation model """

    try: 
        out_map = Image.fromarray(out_map)
        buffer = io.BytesIO()
        out_map.save(buffer, format='PNG')
        b64_map = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            'out_map': b64_map,
            'errors': None
        }
    except Exception as e:
        error_msg = f"Server encountered an error encoding the segmentation output map in 'encode_out_map()': {str(e)}"
        return {
            'out_map': None,
            'errors': error_msg
        }

def encode_bbox_arr(bbox_arr: numpy.ndarray):
    """ Encode the bounding box array as base64 """
    try:
        bbox_arr = bbox_arr.astype(np.float32)
        encoded = base64.b64encode(bbox_arr.tobytes()).decode('utf-8')

        return {
            'bboxes': encoded,
            'dtype': str(bbox_arr.dtype),
            'shape': bbox_arr.shape,
            'errors': None
        }
    except Exception as e:
        error_msg = f"Server encountered an error encoding the bbox array in 'encode_bbox_arr()': {str(e)}"

        return {
            'bboxes': None,
            'dtype': None,
            'shape': None,
            'errors': error_msg
        }


