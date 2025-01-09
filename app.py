import os
import imghdr
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Comment out the following two lines to disable GPU support
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#tf.config.set_visible_devices([], 'GPU')

# Supported image extensions
supported_image_exts = ['jpeg', 'jpg', 'png']

# Load the trained model
model = load_model('trained_model.keras')  # Load the full model from the new .keras file
# Function to predict the emotion from an image
def predict_emotion(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make a prediction
    predictions = model.predict(img_array)
    emotion_labels = ['angry', 'happy', 'neutral', 'sad']
    predicted_emotion = emotion_labels[np.argmax(predictions)]

    return predicted_emotion

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file:
                # Check if the file is an image and has a supported extension
                file_type = imghdr.what(file)
                if file_type in supported_image_exts:
                    # Ensure the static directory exists
                    if not os.path.exists('static'):
                        os.makedirs('static')
                    file_path = os.path.join('static', 'uploaded_image.jpg')
                    file.save(file_path)
                    return render_template('index.html', prediction=None, image_path='uploaded_image.jpg', error=None)
                else:
                    error = "Unsupported file type. Please upload an image with a supported extension: jpeg, jpg, bmp, png."
        elif 'predict' in request.form:
            file_path = os.path.join('static', 'uploaded_image.jpg')
            prediction = predict_emotion(file_path)
            return render_template('index.html', prediction=prediction, image_path='uploaded_image.jpg', error=None)
        elif 'delete' in request.form:
            file_path = os.path.join('static', 'uploaded_image.jpg')
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template('index.html', prediction=None, image_path=None, error=None)
    return render_template('index.html', prediction=None, image_path=None, error=error)

if __name__ == '__main__':
    app.run(debug=True)
