from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((256, 256))
    #image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
