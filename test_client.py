from PIL import Image
import io
import os
import base64
import requests
import cv2
from glob import glob
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

api_key = os.getenv("API_KEY")
endpoint_id = os.getenv("ENDPOINT_ID")

print(endpoint_id)

def encode_image_to_base64(image_path):
	""" Load an image and convert to base64 """
	try:
		with Image.open(image_path) as img:
			# Convert to RGB mode if needed
			if img.mode != "RGB":
				img = img.convert("RGB")

			img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
			# Save to bytes buffer
			buffer = io.BytesIO()
			img.save(buffer, format="JPEG")

			# Encode to base64
			encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

			return encoded_image
	except Exception as e:
		print(f"Error encoding image {image_path}: {str(e)}")
		return None

def decode_b64_to_image(response, out_path):
	""" Load a b64 encoded image and save to PNG """
	try:
		output = response.get('output', {})  # Get response output object
		b64_img = output.get('output_image')  # Grab the output image

		if b64_img is None:
			return {"error": "Missing 'output_image' in API response"}

		# Decode the image
		output_bytes = base64.b64decode(b64_img)
		output_img = Image.open(io.BytesIO(output_bytes)).convert('RGB')

		output_img.save(out_path, 'PNG')

		np_image = np.array(output_img)
		print(np.unique(np_image))

		return 'Image processed and saved!'

	except Exception as e:
		return f"Failed to decode image: {str(e)}"

def send_request_sync(image_path, args):
	""" Send a synchronous request to the API """
	try:
		print(f"Processing image {os.path.basename(image_path)}")

		# Encode the image
		b64_img = encode_image_to_base64(image_path)
		if not b64_img:
			return

		# Prepare the request
		url = f"https://api.runpod.ai/v2/{args['endpoint_id']}/runsync"
		print(url)
		headers = {
			"Authorization": f"Bearer {args['api_key']}",
			"Content-Type": "application/json"
		}
		payload = {
			"input": {
				"image": b64_img
			}
		}

		# Send request and await response
		response = requests.post(url, headers=headers, json=payload)
		response.raise_for_status()
		result = response.json()

		# Report output
		return result

	except Exception as e:
		print(f"Error sending request for image {image_path}: {str(e)}")

args = {
	'endpoint_id': endpoint_id,
	'api_key': api_key
}

FILE_NAMES = glob("*", root_dir='images')
for file_name in FILE_NAMES:
	FILE_PATH = Path('images') / file_name
	response_json = send_request_sync(image_path=FILE_PATH, args=args)
	response_message = decode_b64_to_image(response_json, out_path=Path('output') / os.path.basename(FILE_PATH))

	print(response_json['output'].keys(), response_json['output']['server_inference_time'])

	output = response_json['output']['effdet']['bboxes']

	output_bytes = base64.b64decode(output)
	out_array = np.frombuffer(output_bytes, dtype=np.float32).reshape((100, 6))

	print(out_array)
	
