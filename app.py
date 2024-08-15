import os
import imghdr
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

app = Flask(__name__)

#For Mac support, uncomment these two lines below. This will disable the GPU and only use the CPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#tf.config.set_visible_devices([], 'GPU')

# Supported image extensions
supported_image_exts = ['jpeg', 'jpg', 'png']

# Function to create the model architecture
def create_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax')) # Output layer with 7 classes for emotions

    return model

# Load the trained model
model = create_model()
model.load_weights('trained_model.h5')

# Function to predict the emotion from an image
def predict_emotion(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make a prediction
    predictions = model.predict(img_array)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[np.argmax(predictions)]

    print(f'Predicted Emotion: {predicted_emotion}')
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
