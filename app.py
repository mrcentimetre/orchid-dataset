import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# --- INITIAL SETUP ---
app = Flask(__name__)

# --- LOAD THE TRAINED MODEL ---
# Load the model that Thisara created
model = tf.keras.models.load_model('orchid_classifier_model.h5')
print("Model loaded successfully.")

# --- LOAD STATIC CARE DATA ---
with open('care_data.json', 'r') as f:
    care_data = json.load(f)
print("Care data loaded successfully.")

# --- CLASS NAMES (must match the training order) ---
# IMPORTANT: This list must be in the same order as the training output
CLASS_NAMES = ['Cattleya', 'Dendrobium', 'Oncidium', 'Phalaenopsis', 'Vanda']

# --- IMAGE PRE-PROCESSING FUNCTION ---
def preprocess_image(image_path):
    """Loads and preprocesses an image for model prediction."""
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    
    # Normalize the image
    img_array = img_array / 255.0
    
    return img_array

# --- PING ENDPOINT FOR HEALTH CHECK ---
@app.route('/ping', methods=['GET'])
def ping():
    """A simple endpoint to check if the service is alive."""
    return jsonify({'response': 'pong'})

# --- API ENDPOINT FOR PREDICTION ---
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        # 2. Save the file securely
        filename = secure_filename(file.filename)
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        # 3. Pre-process the image
        processed_image = preprocess_image(filepath)

        # 4. Make a prediction
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(prediction))

        # 5. Get care and decoration info
        species_info = care_data.get(predicted_class_name, {
            "care": "Information not available.",
            "decoration": "Information not available."
        })

        # 6. Prepare the JSON response
        response = {
            'species': predicted_class_name,
            'confidence': f"{confidence:.2%}",
            'care_instructions': species_info['care'],
            'decoration_ideas': species_info['decoration']
        }
        
        # 7. Clean up the uploaded file
        os.remove(filepath)

        return jsonify(response)

    return jsonify({'error': 'An unknown error occurred'}), 500

# --- RUN THE FLASK APP ---
if __name__ == '__main__':
    app.run(debug=True)