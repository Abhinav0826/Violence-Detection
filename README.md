# Violence Detection using CNN (Keras + OpenCV)

This project implements a Violence Detection System using a Convolutional Neural Network (CNN) built with Keras and OpenCV. The system analyzes video footage and automatically detects whether violence is present or not.

The model works by extracting frames from input videos, preprocessing them (resizing and normalising), and training a CNN to perform classification. During prediction, the model evaluates video frames and determines whether the video contains violent activity based on the predicted probability.

## Methodology

### Data Preprocessing

- Extracts the first **10 frames** from each video  
- Resizes each frame to **64 × 64 pixels**  
- Normalizes pixel values to the range **[0, 1]**  
- Assigns labels: Violence or No Violence

---

### Model Architecture

The Convolutional Neural Network (CNN) consists of:

- `Conv2D` (16 filters, 3×3 kernel, ReLU activation)  
- `MaxPooling2D`  
- `Flatten`  
- `Dense` (16 units, ReLU activation)  
- `Dense` (1 unit, Sigmoid activation) → Outputs probability of violence  

---

### Training

- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Evaluation Metric:** Accuracy  
- **Epochs:** 3  

---

### Prediction

For a test video:

1. Extract frames  
2. Preprocess frames (resize and normalize)  
3. Predict violence probability  
4. If `max(prediction) > 0.5` → **Violence Detected**  
   Otherwise → **No Violence Detected**

## Block Output
<img width="577" height="131" alt="Screenshot 2026-03-01 at 3 07 27 pm" src="https://github.com/user-attachments/assets/7a9a016a-c199-4ea3-addf-aa1a7f58bd01" />


## Technologies Used

- Python
- Keras (TensorFlow backend) 
- OpenCV  
- NumPy  

### Possible Improvements

- CNN + LSTM (to capture temporal patterns)  
- 3D CNN for spatiotemporal learning   
- Larger and more diverse dataset 

