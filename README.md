# Machine Learning Algorithm Comparison: Perceptron vs Neural Networks

A comprehensive implementation and comparison of different machine learning algorithms for image classification tasks, including digit recognition and face detection.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Analysis](#results--analysis)
- [Technical Details](#technical-details)


## Overview

This project implements and compares three different machine learning approaches for image classification:

1. **Multi-class Perceptron** - Classical linear classifier for digit recognition (0-9)
2. **Custom Neural Network** - From-scratch implementation with backpropagation
3. **PyTorch Neural Network** - Modern deep learning framework implementation

The project evaluates these algorithms on two distinct tasks:
- **Digit Classification**: Recognizing handwritten digits (0-9)
- **Face Detection**: Binary classification (face vs non-face)

## Project Structure

```
├── loaddata.py                     # Data loading and preprocessing utilities
├── perceptron.py                   # Perceptron implementations
├── neuralnetworkimplementations.py # Custom neural network from scratch
├── pytorch_implementation.py       # PyTorch-based neural network
├── data/
│   ├── digitdata/
│   │   ├── trainingimages
│   │   ├── traininglabels
│   │   ├── testimages
│   │   ├── testlabels
│   │   ├── validationimages
│   │   └── validationlabels
│   └── facedata/
│       ├── facedatatrain
│       ├── facedatatrainlabels
│       ├── facedatatest
│       ├── facedatatestlabels
│       ├── facedatavalidation
│       └── facedatavalidationlabels
└── README.md
```

## Algorithms Implemented

### 1. Multi-Class Perceptron
- **Architecture**: One-vs-all classification with 10 binary perceptrons
- **Features**: Linear decision boundaries, simple weight updates
- **Use Case**: Digit classification (0-9)

### 2. Face Classifier (Binary Perceptron)
- **Architecture**: Single perceptron for binary classification
- **Features**: Face vs non-face detection
- **Use Case**: Face detection

### 3. Custom Neural Network
- **Architecture**: 3-layer network (784/4200 → 64 → 32 → 1)
- **Features**: 
  - ReLU activation for hidden layers
  - Sigmoid activation for output layer
  - Dropout regularization (rate: 0.5)
  - L2 regularization (λ = 0.001)
  - Xavier weight initialization

### 4. PyTorch Neural Network
- **Architecture**: 3-layer network with configurable hidden sizes
- **Features**:
  - Adam optimizer
  - Cross-entropy loss (multiclass) / BCE loss (binary)
  - GPU acceleration support
  - Professional deep learning practices

## Datasets

### Digit Dataset
- **Format**: 28×28 pixel images
- **Classes**: 10 (digits 0-9)
- **Encoding**: ' ' = 0, '+' = 1, '#' = 2
- **Task**: Multi-class classification

### Face Dataset  
- **Format**: 70×60 pixel images
- **Classes**: 2 (face = 1, non-face = 0)
- **Encoding**: ' ' = background, other characters = foreground
- **Task**: Binary classification

## Requirements

### Core Dependencies
```
python >= 3.7
numpy >= 1.19.0
matplotlib >= 3.3.0
torch >= 1.9.0
torchvision >= 0.10.0
```

### Optional Dependencies
```
jupyter >= 1.0.0  # For notebook analysis
scikit-learn >= 0.24.0  # For comparison metrics
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd machine-learning-comparison
   ```

2. **Create virtual environment**
   ```bash
   python -m venv ml_env
   source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install numpy matplotlib torch torchvision
   ```

4. **Verify data structure**
   Ensure your `data/` folder contains the required datasets as shown in the project structure.

## 💻 Usage

### Run Individual Algorithms

**Perceptron Implementation:**
```bash
python perceptron.py
```
- Trains multi-class perceptron on digits
- Trains binary perceptron on faces
- Generates accuracy, timing, and standard deviation plots

**Custom Neural Network:**
```bash
python neuralnetworkimplementations.py
```
- Configurable dataset selection (`dataset_choice = 'digit'` or `'face'`)
- Implements dropout and regularization
- Tracks training progress per epoch

**PyTorch Implementation:**
```bash
python pytorch_implementation.py
```
- Runs both digit and face classification
- Provides prediction demonstrations
- Generates comprehensive performance plots

### Configuration Options

**Neural Network Hyperparameters:**
```python
# neuralnetworkimplementations.py
EPOCHS = 20
LEARNING_RATE = 0.001
REG_LAMBDA = 0.001
HIDDEN1_SIZE = 64
HIDDEN2_SIZE = 32
```

**PyTorch Model Configuration:**
```python
# pytorch_implementation.py
model = ThreeLayerNN(
    input_size=784,     # 28*28 for digits, 70*60 for faces
    hidden1=256,        # First hidden layer size
    hidden2=128,        # Second hidden layer size
    output_size=10      # 10 for digits, 1 for faces
)
```

## Results & Analysis

### Performance Metrics
Each algorithm is evaluated on:
- **Training Time**: Speed of convergence
- **Test Accuracy**: Final classification performance  
- **Standard Deviation**: Consistency across runs
- **Learning Curves**: Performance vs training data percentage

### Expected Performance Ranges
- **Perceptron (Digits)**: 70-85% accuracy
- **Perceptron (Faces)**: 80-90% accuracy
- **Neural Networks**: 85-95% accuracy (both tasks)

### Visualization
All implementations generate matplotlib plots showing:
- Training time vs data percentage
- Accuracy vs data percentage  
- Standard deviation across multiple runs

## Technical Details

### Data Preprocessing
- **Pixel Normalization**: Characters converted to binary (0/1) or normalized (0-1)
- **Feature Extraction**: Images flattened to 1D vectors
- **Label Processing**: Multi-class (digits) vs binary (faces) encoding

### Algorithm Specifics

**Perceptron Learning Rule:**
```
if prediction ≠ actual:
    weights += learning_rate × actual_label × features
    bias += learning_rate × actual_label
```

**Neural Network Backpropagation:**
- Forward pass with ReLU/Sigmoid activations
- Backward pass with gradient computation
- Weight updates with momentum and regularization

**Regularization Techniques:**
- L2 weight decay
- Dropout during training
- Xavier/He weight initialization

### Performance Optimization
- Batch processing for efficiency
- Random data shuffling per epoch
- Multiple trial averaging (3 runs per configuration)

---

**Note**: This project was developed for educational purposes to demonstrate the implementation and comparison of classical and modern machine learning algorithms on computer vision tasks.
