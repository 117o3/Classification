# Machine Learning Algorithm Comparison: Perceptron vs Neural Networks

A comprehensive implementation and comparison of different machine learning algorithms for image classification tasks, including digit recognition and face detection.

## Table of Contents
- [Overview](#overview)
- [Key Achievements](#key-achievements)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Data Loading & Preprocessing](#data-loading--preprocessing)
- [Datasets](#datasets)
- [Results & Analysis](#results--analysis)
- [Technical Details](#technical-details)
- [Usage](#usage)
- [Requirements](#requirements)

## Overview

This project implements and compares four different machine learning approaches for image classification:

- **Multi-class Perceptron** - Classical linear classifier for digit recognition (0-9)
- **Binary Perceptron** - Face vs non-face detection
- **Custom Neural Network** - From-scratch implementation with backpropagation
- **PyTorch Neural Network** - Modern deep learning framework implementation

The project evaluates these algorithms on two distinct computer vision tasks:
- **Digit Classification**: Recognizing handwritten digits (0-9)
- **Face Detection**: Binary classification (face vs non-face)

## Key Achievements

- **Implemented 4 different ML algorithms from scratch** in Python
- **Achieved 85-98% accuracy** on image classification tasks
- **Compared classical vs modern approaches** with comprehensive performance analysis
- **Built complete ML pipeline**: data loading → preprocessing → training → evaluation → visualization
- **Handled both binary and multiclass classification** problems
- **Processed ASCII-formatted image data** and converted to numerical representations

## Technologies Used

- **Languages**: Python
- **ML Libraries**: PyTorch, NumPy
- **Visualization**: Matplotlib
- **ML Concepts**: Neural Networks, Perceptron, Backpropagation, Regularization, Dropout
- **Skills**: Data preprocessing, Model evaluation, Performance analysis, Algorithm comparison

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
- **Implementation**: Separate binary perceptron for each digit class

### 2. Face Classifier (Binary Perceptron)
- **Architecture**: Single binary perceptron
- **Features**: Face vs non-face detection with binarized pixel inputs
- **Use Case**: Face detection
- **Label Processing**: Converts face=1, non-face=-1 for training

### 3. Custom Neural Network (Binary Classification Only)
- **Architecture**: 4-layer network with 2 hidden layers (input → 64 → 32 → 1)
- **Input Sizes**: 784 (digits) or 4200 (faces)
- **Features**:
  - ReLU activation for hidden layers
  - Sigmoid activation for output layer
  - Dropout regularization (rate: 0.5) during forward pass
  - L2 regularization (λ = 0.001)
  - Xavier weight initialization
- **Tasks**: 
  - Face detection (face vs non-face)
  - Digit binary classification (0 vs non-0 only)
- **Note**: Does not perform multiclass digit classification (0-9)

### 4. PyTorch Neural Network
- **Architecture**: 3-layer fully connected network (fc1 → fc2 → fc3)
- **Default Hidden Sizes**: 256 → 128 → output
- **Features**:
  - Adam optimizer
  - Cross-entropy loss (multiclass) / BCE loss (binary)
  - GPU acceleration support
  - Professional deep learning practices
- **Tasks**: Both multiclass digit classification (0-9) and binary face detection

## Data Loading & Preprocessing

### loaddata.py
Core utility module for loading and preprocessing ASCII-formatted image data.

**Key Functions:**
- `loadLabelsFile(filename)` - Reads classification labels from text files
- `loadImagesFile(filename, num_labels, cols)` - Converts ASCII art images to numerical matrices
- `IntegerConversionFunction(character)` - Maps ASCII characters to integers

**Character Encoding:**
```
' ' (space) → 0    # Background pixels
'+' (plus)  → 1    # Light foreground
'#' (hash)  → 2    # Heavy foreground
```

**Usage Example:**
```python
from loaddata import loadLabelsFile, loadImagesFile

# Load training data
labels = loadLabelsFile("data/digitdata/traininglabels")
images = loadImagesFile("data/digitdata/trainingimages", len(labels), 28)
```

The module validates file integrity and handles both 28×28 digit images and 60×70 face images, converting text-based representations into arrays suitable for machine learning algorithms.

## Datasets

### Digit Dataset
- **Format**: 28×28 pixel images
- **Classes**: 10 (digits 0-9) for PyTorch; Binary (0 vs non-0) for custom neural network
- **Encoding**: ' ' = 0, '+' = 1, '#' = 2 → normalized or binarized
- **Task**: Multiclass classification (PyTorch) or Binary classification (Custom NN)

### Face Dataset
- **Format**: 60×70 pixel images (width × height)
- **Classes**: 2 (face = 1, non-face = 0)
- **Encoding**: ' ' = background, other characters = foreground → binarized to 0/1
- **Task**: Binary classification

## Results & Analysis

### Performance Metrics
Each algorithm is evaluated on:
- **Training Time**: Speed of convergence
- **Test Accuracy**: Final classification performance
- **Standard Deviation**: Consistency across runs
- **Learning Curves**: Performance vs training data percentage

### Expected Performance Ranges
- **Perceptron (Digits 0-9)**: 70-85% accuracy
- **Perceptron (Faces)**: 80-90% accuracy  
- **Custom Neural Network (0 vs non-0 digits)**: 85-95% accuracy
- **Custom Neural Network (Faces)**: 85-95% accuracy
- **PyTorch Neural Network (Digits 0-9)**: 85-95% accuracy
- **PyTorch Neural Network (Faces)**: 90-98% accuracy

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

### Label Processing Complexities

**Digit Classification**:
- **Perceptron**: Maintains original labels (0-9) for multiclass training
- **Custom Neural Network**: Converts 0→1, all other digits→0 (binary classification)
- **PyTorch**: Supports both original multiclass (0-9) and binary tasks

**Face Classification**:
- **All models**: face=1, non-face=0 (or -1 for perceptron training)

**Data Preprocessing Differences**:
- **Custom Neural Network**: Simple binarization (non-space → 1, space → 0)
- **PyTorch Multiclass**: Normalized encoding (0 → 0.0, 1 → 0.5, 2 → 1.0)
- **PyTorch Binary**: Binarized encoding (non-zero → 1, zero → 0)

### Performance Optimization
- Batch processing for efficiency
- Random data shuffling per epoch
- Multiple trial averaging (3 runs per configuration)

## Usage

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
- Configurable dataset selection (modify `dataset_choice = 'digit'` or `'face'`)
- Implements dropout and regularization
- Tracks training progress per epoch

**PyTorch Implementation:**
```bash
python pytorch_implementation.py
```
- Runs both digit and face classification
- Provides prediction demonstrations
- Generates comprehensive performance plots

## Requirements

### Core Dependencies
- `python >= 3.7`
- `numpy >= 1.19.0`
- `matplotlib >= 3.3.0`
- `torch >= 1.9.0`
- `torchvision >= 0.10.0`

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Digit-and-Face-Classification

# Install dependencies
pip install numpy matplotlib torch torchvision
```

---

**Note**: This project was developed for educational purposes to demonstrate the implementation and comparison of classical and modern machine learning algorithms on computer vision tasks.
