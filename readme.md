# Emotion Classification with CNN
![Screenshot 2024-08-15 at 13-00-19 Image Emotion Classifier(3)](https://github.com/user-attachments/assets/753c6ed2-8673-4448-a04e-198169bacd0e)

## Project Overview

In this project, I built a Convolutional Neural Network (CNN) to classify emotions from images. The model was trained on the FER-2013 dataset. The model can recognize seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. Additionally, I created a Flask web application where you can upload your own images and receive emotion predictions based on the trained model.

### Key Features

- **CNN Architecture:** I designed a network with multiple convolutional, batch normalization, max-pooling, and dropout layers to effectively learn from the FER-2013 dataset.
- **Data Augmentation:** I employed techniques such as random shifts, horizontal flips, and rescaling to improve the model's generalization.
- **Model Checkpointing:** I implemented model checkpointing to save the best-performing model weights based on validation accuracy.
- **Visualization:** I generated plots to visualize training and validation loss and accuracy.
- **Web Application:** I developed a Flask app that allows users to upload images and get real-time emotion predictions.

### Project Files

- **train_model.py:** This file contains the code for training the CNN model, including data preprocessing, model architecture, and training steps.
- **app.py:** This file includes the Flask web application for image upload and emotion prediction using the trained model.

### Running and Testing

To run and test this project yourself:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jonathancolasante/Neural-Network-Emotion-Classifier.git
   cd Neural-Network-Emotion-Classifier
   ```
   
2. **Install the required packages:**

   ```bash
   pip install ...
   ```
   
3. **Run the Flask web application:**

   ```bash
   python app.py
   ```
   
   This will start the web app, and you can access it in your web browser. Upload your own images to test the emotion prediction.

4. **Train your own model:**

   If you want to train the model yourself, run:

   ```bash
   python train_model.py
   ```
   
   This will preprocess the data, train the CNN model, and save the best-performing model weights based on validation accuracy.

### Notes

1. **Mac Users:**

   Users on Mac may face issues trying to run the web app or train the model. In that case, try uncommenting these two lines in the files:

   ```python
   # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
   # tf.config.set_visible_devices([], 'GPU')
   ```

   Mac users may also need to set up a custom environment for TensorFlow

2. **Data Set:**

   Due to the size of the dataset, I was not able to provide it in the repository. Users can modify the code to use their own data or simply download the FER-2013 dataset online.
