from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# --- SETUP ---
app = Flask(__name__)
model = tf.keras.models.load_model('cattle_breed_classifier.h5')
with open('class_labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]
IMG_HEIGHT, IMG_WIDTH = 224, 224

# --- PREPROCESSING FUNCTION ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        prediction = model.predict(processed_image)[0]
        
        top_indices = prediction.argsort()[-3:][::-1]
        results = [{'breed': class_labels[i], 'confidence': f"{float(prediction[i]) * 100:.2f}%"} for i in top_indices]
        return jsonify(results)
    return jsonify({'error': 'An unexpected error occurred'}), 500

# --- RUN SERVER ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
