from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import torch

# Load BLIP model (instruction-following variant)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def process_image(image_url):
    try:
        print(f"Processing image: {image_url}")
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')

        prompt = "Describe the image with focus on skin concerns and face features."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        output = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(output[0], skip_special_tokens=True)
        print(f"Caption: {caption}")
        return caption
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Failed to process image."
