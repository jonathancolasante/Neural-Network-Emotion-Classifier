import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# For Mac support, uncomment these two lines below. This will disable the GPU and only use the CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.set_visible_devices

# Check if directories exist
train_dir = "data/train"  # Directory containing the training data
test_dir = "data/test"    # Directory containing the validation data

if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise ValueError("Train or test directory not found. Please check the paths.")

# Directory for saving training graphs
output_dir = 'training_graphs'
os.makedirs(output_dir, exist_ok=True)

# Image data generators for training and validation data
training_data_augmentation = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.2
)

validation_data_augmentation = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Generate batches of augmented data for training and validation
training_data_generator = training_data_augmentation.flow_from_directory(
    directory=train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

validation_data_generator = validation_data_augmentation.flow_from_directory(
    directory=test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# Define the CNN model architecture
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
model.add(Dense(7, activation='softmax'))  # Output layer with 7 classes for emotions

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Callback to save the best model weights based on validation accuracy
model_checkpoint = ModelCheckpoint(
    filepath='trained_model.h5',
    monitor='value_accuracyuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

# Train the model
training_history = model.fit(
    training_data_generator,
    steps_per_epoch=len(training_data_generator),
    epochs=50,
    validation_data=validation_data_generator,
    validation_steps=len(validation_data_generator),
    callbacks=[model_checkpoint]
)

# Plot training and validation loss
training_loss = training_history.history['loss']
value_loss = training_history.history['value_loss']
epochs = range(1, len(training_loss) + 1)
plt.plot(epochs, training_loss, 'bo', label='Training loss')
plt.plot(epochs, value_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
loss_plot_path = os.path.join(output_dir, 'loss_graph.jpg')
plt.savefig(loss_plot_path)

# Plot training and validation accuracy
training_accuracy = training_history.history['accuracy']
value_accuracy = training_history.history['val_accuracy']
plt.plot(epochs, training_accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, value_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
accuracy_plot_path = os.path.join(output_dir, 'accuracy_graph.jpg')
plt.savefig(accuracy_plot_path)