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

2. **Build the Docker image**:
   
   - Make sure Docker is properly installed on your system and the Docker service is running.

   ```bash
   docker build -t emotion-classifier .
   ```

2. **Run the Docker container**:

   ```bash
   docker run -p 8000:8000 --name emotion-classifier-container emotion-classifier
   ```

Once the container is running, you can access the web application in your browser at http://localhost:8000. Upload your own images to test the emotion prediction.
   



