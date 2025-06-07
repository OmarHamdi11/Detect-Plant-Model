# from flask import Flask, request, jsonify
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import io

# # Initialize Flask app
# app = Flask(__name__)

# # Load model
# model = tf.keras.models.load_model("model/plantDiseaseClassification2.keras")

# # Define class names
# class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
#                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
#                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
#                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
#                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
#                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
#                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
#                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
#                'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
#                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
#                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# # Preprocess image
# def preprocess_image(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     image = image.resize((224, 224))  # Resize to match model input
#     image = np.array(image) / 255.0   # Normalize
#     image = np.expand_dims(image, axis=0)
#     return image

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     file = request.files['image']
#     image_bytes = file.read()
#     processed_image = preprocess_image(image_bytes)

#     prediction = model.predict(processed_image)
#     predicted_class = class_names[np.argmax(prediction)]
#     confidence = float(np.max(prediction))

#     return jsonify({
#         'predicted_class': predicted_class,
#         'confidence': confidence
#     })

# # Health check route
# @app.route('/', methods=['GET'])
# def index():
#     return jsonify({'message': 'Plant Disease Classification API is running!'})

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from utils import preprocess_image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/plantDiseaseClassification2.keras")

# Class names...
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
               'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'API is running!'})

if __name__ == '__main__':
    app.run(debug=True)
