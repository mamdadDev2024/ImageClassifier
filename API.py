from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU devices:", gpus)
else:
    print("No GPU devices found, training will run on CPU.")


app = Flask(__name__)

model = tf.keras.models.load_model('my_model.keras')

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return False

    file = request.files['image']
    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)[0][0]
    return jsonify({'score': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True , port=5050)
