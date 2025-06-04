from flask import Flask, request, jsonify
from utils import predict_plant
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        plant_name = predict_plant(file_path)
        return jsonify({'plant_name': plant_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
