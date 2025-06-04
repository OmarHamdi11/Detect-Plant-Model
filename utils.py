from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model once
model = load_model("model/cnn_model.h5")

# Define your class labels (update as per your model)
class_labels = ['Aloe Vera', 'Basil', 'Mint', 'Rosemary', 'Snake Plant']

def predict_plant(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Change if your model uses another size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if you did during training

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    return class_labels[predicted_class]
