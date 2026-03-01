import cv2
import numpy as np
import os
from keras import layers, models

imgSize = 64 # Resize all the frames equally to 64 x 64 pixels

def load_data(folder, label):
    """ Load frames from all the videos and normalise them """
    data = []
    labels = []

    # Traverse throught each video in the folder
    for v in os.listdir(folder):
        cap = cv2.VideoCapture(os.path.join(folder, v))
        cnt = 0

        fbool, frame = cap.read() # Read first frame
        while fbool and cnt < 10: # Maximum frames is 10
            frame = cv2.resize(frame, (imgSize, imgSize)) # Resizing 
            frame = frame / 255.0 # Normalising
            # Appending into a list
            data.append(frame)
            labels.append(label)
            cnt += 1
            #Read next frame
            fbool, frame = cap.read()

        cap.release()

    return data, labels

# Load video data and label from violent and Non violent videos
violence_data, violence_labels = load_data("Violence Dataset/Violence", 1)
nonviolence_data, nonviolence_labels = load_data("Violence Dataset/NonViolence", 0)

# Combine the data and its label into an array
X = np.array(violence_data + nonviolence_data)
y = np.array(violence_labels + nonviolence_labels)

# Build CNN Model for prediction
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)), # Convolution layer
    layers.MaxPooling2D(2,2), # Reduce image size
    layers.Flatten(), # Converts 2D features to 1D
    layers.Dense(16, activation='relu'), # Combines all features to learn patterns
    layers.Dense(1, activation='sigmoid')]) # Outputs probability of violence


# Compile it with with optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Trains the model using the data
model.fit(X, y, epochs=3)

# Test trained model on a Video
cap = cv2.VideoCapture("Violence Dataset/Violence/V_3.mp4")
frames = []

fbool, frame = cap.read() 
while fbool:
    frame = cv2.resize(frame, (imgSize, imgSize))
    frame = frame / 255.0
    frames.append(frame)
    fbool, frame = cap.read()

cap.release()

frames = np.array(frames)
prediction = model.predict(frames)

# Detect violence if probability > 50%
if np.max(prediction) > 0.5:
    print("Violence Detected")
else:
    print("No Violence Detected")