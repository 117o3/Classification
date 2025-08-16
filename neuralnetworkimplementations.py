import numpy as np
import random
import time
import matplotlib.pyplot as plt
from loaddata import loadLabelsFile, loadImagesFile

# Constants for digit and face images
DIGIT_IMAGE_WIDTH = 28
DIGIT_IMAGE_HEIGHT = 28
FACE_IMAGE_WIDTH = 60
FACE_IMAGE_HEIGHT = 70
DIGIT_INPUT_SIZE = DIGIT_IMAGE_WIDTH * DIGIT_IMAGE_HEIGHT
FACE_INPUT_SIZE = FACE_IMAGE_WIDTH * FACE_IMAGE_HEIGHT
HIDDEN1_SIZE = 64
HIDDEN2_SIZE = 32
OUTPUT_SIZE = 1

# Epochs, learning rate, and regularization
EPOCHS = 20
LEARNING_RATE = 0.001  # Adjust learning rate. Smaller = better
REG_LAMBDA = 0.001

# Activation functions and derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(a):
    return a * (1 - a)
def relu(z):
    return np.maximum(0, z) #Relu helps neural network learn non-linearly, prevents network from memorizing
def relu_derivative(a):
    return (a > 0).astype(float)


# Dropout function. Randomly drops units during training to reduce overfitting
def dropout(a, rate):
    mask = (np.random.rand(*a.shape) > rate) / (1 - rate)  
    return a * mask

# Forward pass with dropout, had errors with overfitting here
def forward(x, w1, b1, w2, b2, w3, b3, dropout_rate=0.5):
    x = np.array(x)
    
    a0 = x.reshape(-1, 1) 
    
    z1 = np.dot(w1, a0) + b1
    a1 = relu(z1)
    a1 = dropout(a1, dropout_rate)  # Dropout to a1
    
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    a2 = dropout(a2, dropout_rate)  # Dropout to a2
    
    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)  # Sigmoid for output layer, gives predicted probability
    
    return a0, a1, a2, a3

# Backward pass 
def backward(x, y_true, a0, a1, a2, a3, w1, w2, w3):
    y = np.array([[y_true]])

    dz3 = a3 - y
    dw3 = np.dot(dz3, a2.T) + REG_LAMBDA * w3
    db3 = dz3

    dz2 = np.dot(w3.T, dz3) * relu_derivative(a2)
    dw2 = np.dot(dz2, a1.T) + REG_LAMBDA * w2
    db2 = dz2

    dz1 = np.dot(w2.T, dz2) * relu_derivative(a1)
    dw1 = np.dot(dz1, a0.T) + REG_LAMBDA * w1
    db1 = dz1

    return dw1, db1, dw2, db2, dw3, db3

# Weight update, had to brute force because of errors
def update_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    return w1, b1, w2, b2, w3, b3

# Prediction function
def predict(x, w1, b1, w2, b2, w3, b3):
    a0, a1, a2, a3 = forward(x, w1, b1, w2, b2, w3, b3)
    return 1 if a3 >= 0.5 else 0

# Load Digit Data
def load_digit_data(image_file, label_file):
    with open(image_file, 'r') as f_img:
        raw = f_img.read().splitlines()

    with open(label_file, 'r') as f_lbl:
        labels = list(map(int, f_lbl.read().splitlines()))

    data = []
    for i in range(0, len(raw), DIGIT_IMAGE_HEIGHT):
        block = raw[i:i+DIGIT_IMAGE_HEIGHT]
        vector = []
        for row in block:
            for c in row:
                if c != ' ':
                    vector.append(1)
                else:
                    vector.append(0)
        data.append(np.array(vector))
        
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = 1
        else:
            labels[i] = 0
            
    return np.array(data), np.array(labels)



# Load Face Data
def load_face_data(image_file, label_file):
    with open(image_file, 'r') as f_img:
        raw = f_img.read().splitlines()

    with open(label_file, 'r') as f_lbl:
        labels = list(map(int, f_lbl.read().splitlines()))

    data = []
    for i in range(0, len(raw), FACE_IMAGE_HEIGHT):
        block = raw[i:i+FACE_IMAGE_HEIGHT]
        vector = []
        for row in block:
            for c in row:
                if c != ' ':
                    vector.append(1)
                else:
                    vector.append(0)
        data.append(np.array(vector))

    for i in range(len(labels)):
        if labels[i] == 1:
            labels[i] = 1
        else:
            labels[i] = 0  
    return np.array(data), np.array(labels)






# Neural Network Training, errors with overfitting and propogation
def train_model(x_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE, input_size=DIGIT_INPUT_SIZE):
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    w1 = np.random.randn(HIDDEN1_SIZE, input_size) * np.sqrt(1. / input_size)
    b1 = np.zeros((HIDDEN1_SIZE, 1))
    
    w2 = np.random.randn(HIDDEN2_SIZE, HIDDEN1_SIZE) * np.sqrt(1. / HIDDEN1_SIZE)
    b2 = np.zeros((HIDDEN2_SIZE, 1))
    
    w3 = np.random.randn(OUTPUT_SIZE, HIDDEN2_SIZE) * np.sqrt(1. / HIDDEN2_SIZE)
    b3 = np.zeros((OUTPUT_SIZE, 1))

    for epoch in range(epochs):
        perm = np.random.permutation(len(x_train))
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        for xi, yi in zip(x_shuffled, y_shuffled):
            yi = float(yi)

            a0, a1, a2, a3 = forward(xi, w1, b1, w2, b2, w3, b3)
            dw1, db1, dw2, db2, dw3, db3 = backward(xi, yi, a0, a1, a2, a3, w1, w2, w3)
            w1, b1, w2, b2, w3, b3 = update_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate)

        # Print progress at every epoch
        if epoch % 1 == 0:  
            predictions = np.array([predict(x, w1, b1, w2, b2, w3, b3) for x in x_train])
            accuracy = np.mean(predictions == np.array(y_train))
            print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {accuracy * 100:.2f}%")

    return w1, b1, w2, b2, w3, b3
    

# Main function
def main():
    
    dataset_choice = 'face'  # Changable

    if dataset_choice == 'digit':
        x_train, y_train = load_digit_data('data/digitdata/trainingimages', 'data/digitdata/traininglabels')
        x_val, y_val = load_digit_data('data/digitdata/validationimages', 'data/digitdata/validationlabels')
        input_size = DIGIT_INPUT_SIZE
        
    elif dataset_choice == 'face':
        x_train, y_train = load_face_data('data/facedata/facedatatrain', 'data/facedata/facedatatrainlabels')
        x_val, y_val = load_face_data('data/facedata/facedatavalidation', 'data/facedata/facedatavalidationlabels')
        input_size = FACE_INPUT_SIZE
        
    else:
        raise ValueError("Invalid dataset choice. Use 'digit' or 'face'.")

    # Training fxn
    w1, b1, w2, b2, w3, b3 = train_model(x_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE, input_size=input_size)

    # Evaluation
    predictions = np.array([predict(x, w1, b1, w2, b2, w3, b3) for x in x_train])
    accuracy = np.mean(predictions == np.array(y_train))
    print(f"Training accuracy: {accuracy:.3f}")

if __name__ == '__main__':
    main()
