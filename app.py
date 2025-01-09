import os
import imghdr
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

#Uncomment the following two lines to disable GPU support
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#tf.config.set_visible_devices([], 'GPU')

supported_image_exts = ['jpeg', 'jpg', 'png']

model = load_model('trained_model.keras')  

def predict_emotion(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

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
                file_type = imghdr.what(file)
                if file_type in supported_image_exts:
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
