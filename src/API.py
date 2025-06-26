"""
این کد برای ارائه API فلسک به کاربر هست که میتواند با ارسال فرم با متد POST تا تصویر دریافتی از آن را طبقه بندی و در صورت امتیاز بالای 60 درصد درست و در غیر اینصورت غلط بر میگرداند
"""
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import logging
import os

# برای حذف لاگ های غیر ضروری
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('werkzeug').setLevel(logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# اطمینان از استفاده از GPU و اعمال محدودیت حافظه
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

app = Flask(__name__)

# مدل آموزش دیده شده
model = tf.keras.models.load_model('simple_model.keras')

# دریافت تصویر و انجام پیش پردازش
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/process-image', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print(request.files)
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    image_bytes = file.read()

    try:
        processed_image = preprocess_image(image_bytes)
        probability = float(model.predict(processed_image)[0][0])
        prediction = probability < 0.6

        return jsonify({
            'prediction': prediction,
            'probability': probability
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# اجرا وب سرور در صورت اجرای فایل پایتون به صورت مستقیم
if __name__ == '__main__':
    app.run(debug=False, port=5050)
