import random
import numpy as np
import matplotlib.pyplot as plt
import time
from loaddata import loadLabelsFile, loadImagesFile

class MultiClassPerceptron:
    def __init__(self, input_size, num_classes=10, learning_rate=1.0):
        self.num_classes = num_classes
        self.models = [Perceptron(input_size, learning_rate) for _ in range(num_classes)]
    
    def train(self, training_data, training_labels, max_iterations=20):
        for i in range(self.num_classes):
            # For each classifier, we train for the respective class as the positive class
            binary_labels = [1 if label == i else -1 for label in training_labels]
            self.models[i].train(training_data, binary_labels, max_iterations)
    
    def predict(self, x):
        predictions = [model.predict(x) for model in self.models]
        # with open('_predictions.txt', 'a') as f:
        #     f.write(' '.join(str(p) for p in predictions) + '\n')
        return np.argmax(predictions)  # The model with the highest activation score is the predicted class
    
class FaceClassifier:
    def __init__(self, input_size, learning_rate=1.0):
        self.model = Perceptron(input_size, learning_rate)
    
    def train(self, training_data, training_labels, max_iterations=20):
        # Face vs Non-Face, where 1 = face, -1 = non-face
        binary_labels = [1 if label == 1 else -1 for label in training_labels] # redundant
        self.model.train(training_data, binary_labels, max_iterations)
    
    def predict(self, x):
        return self.model.predict(x)

class Perceptron:
    def __init__(self, input_size, learning_rate=1.0):
        self.weights = np.zeros(input_size) # intialize weights to 0
        self.bias = 0
        self.learning_rate = learning_rate

    def predict(self, x):
        activation = np.dot(self.weights, x) + self.bias # dot product of weights & inputs + bias
        return 1 if activation >= 0 else -1 # returns activation or -1

    def train(self, training_data, training_labels, max_iterations=20):
        for _ in range(max_iterations):
            updated = False
            for x, y in zip(training_data, training_labels): # process each training example
                prediction = self.predict(x)
                if prediction != y: # if misclassified
                    update = self.learning_rate * y
                    self.weights += update * x # update weights
                    self.bias += update # update bias
                    updated = True
            if not updated:
                break

    def evaluate(self, data, labels): # calculate accuracy
        correct = sum(self.predict(x) == y for x, y in zip(data, labels))
        return correct / len(labels)

def load_data(image_file, label_file, image_height, image_width, positive_class=None):
    # Load original labels without converting to binary for multiclass
    original_labels = list(map(int, loadLabelsFile(label_file)))
    num_labels = len(original_labels)
    image_sections = loadImagesFile(image_file, num_labels, image_width)
    
    # convert to vectors and binarize pixels
    data = []
    for section in image_sections:
        vector = [1 if pixel != 0 else 0 for row in section for pixel in row]
        data.append(np.array(vector))
    
    # If positive_class is provided, convert to binary labels (face vs non-face)
    if positive_class is not None:
        labels = [1 if lbl == positive_class else -1 for lbl in original_labels]
    else:
        # For multiclass, keep original labels
        labels = original_labels
        
    return np.array(data), np.array(labels)

def accuracy(predictions, labels):
    return np.mean(predictions == labels)

def plot_metric(values, title, ylabel, color):
    percentages = np.arange(0.1, 1.1, 0.1)
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, values, marker='o', linestyle='-', color=color)
    plt.xlabel('Percentage of Training Data')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_experiment(data_name, image_height, image_width, positive_class,
                  train_image_path, train_label_path,
                  test_image_path, test_label_path):
    
    print(f"\nRunning {data_name} classification...")
    
    # load data (x-images, y-labels)
    x_train, y_train = load_data(train_image_path, train_label_path, 
                               image_height, image_width, positive_class)
    x_test, y_test = load_data(test_image_path, test_label_path,
                             image_height, image_width, positive_class)

    # For digit classification, convert binary labels back to multiclass
    if data_name == "Digits":
        # We need to load the original non-binary labels for digit classification
        original_train_labels = list(map(int, loadLabelsFile(train_label_path)))
        original_test_labels = list(map(int, loadLabelsFile(test_label_path)))
        y_train_multiclass = np.array(original_train_labels)
        y_test_multiclass = np.array(original_test_labels)

    total_samples = len(x_train)
    percentages = [i/10 for i in range(1, 11)] # training percentages from 10%-100%
    
    accuracies, stds, times = [], [], []
    print("Training on randomly selected data for each percentage")
    
    for pct in percentages:
        size = int(total_samples * pct)
        acc_trials = []
        time_total = 0
        print(f"\nTraining on {int(pct*100)}% of data ({size} samples)")
        
        for _ in range(3): # from 5 to 3 for developing purposes (speed)
            indices = random.sample(range(total_samples), size)
            x_sub = x_train[indices]
            
            # Choose the appropriate model based on data_name
            if data_name == "Digits":
                y_sub = y_train_multiclass[indices]
                # For digits, use MultiClassPerceptron
                model = MultiClassPerceptron(input_size=image_height*image_width)
            else:  # Face classification
                y_sub = y_train[indices]
                # For faces, use regular Perceptron through FaceClassifier
                model = FaceClassifier(input_size=image_height*image_width)
            
            # calculate how long train() takes to run & train model
            start = time.time()
            model.train(x_sub, y_sub)
            end = time.time()
            
            # test trained model
            if data_name == "Digits":
                predictions = np.array([model.predict(x) for x in x_test])
                acc = accuracy(predictions, y_test_multiclass)
            else:  # Face classification
                predictions = np.array([model.predict(x) for x in x_test])
                acc = accuracy(predictions, y_test)
                
            # with open(f'{data_name.lower()}_predictions.txt', 'w') as f:
            #     for prediction in predictions:
            #         f.write(str(prediction) + '\n')
            
            acc_trials.append(acc)
            time_total += (end - start)
        
        avg_acc = np.mean(acc_trials)
        std_dev = np.std(acc_trials)
        avg_time = round(time_total / 3, 3)
        
        accuracies.append(avg_acc)
        stds.append(std_dev)
        times.append(avg_time)
        
        print(f"Test accuracy: {round(avg_acc, 3)}")
        print(f"Time taken: {avg_time}s")
        print(f"Standard deviation: {std_dev:.4f}")
    
    # plot results
    plot_metric(times, f'{"MultiClassPerceptron" if data_name == "Digits" else "Perceptron"} for {data_name} - Training Time', 
               'Time (s)', 'green')
    plot_metric(accuracies, f'{"MultiClassPerceptron" if data_name == "Digits" else "Perceptron"} for {data_name} - Accuracy',
               'Accuracy', 'blue')
    plot_metric(stds, f'{"MultiClassPerceptron" if data_name == "Digits" else "Perceptron"} for {data_name} - Std Dev',
               'Standard Deviation', 'red')

def main():
    # digit classification (multiclass 0-9)
    run_experiment(
        "Digits", 28, 28, None,  # Set positive_class to None for multiclass
        "data/digitdata/trainingimages",
        "data/digitdata/traininglabels",
        "data/digitdata/testimages",
        "data/digitdata/testlabels"
    )
    
    # face classification (face vs non-face)
    run_experiment(
        "Faces", 70, 60, 1,  # Keep positive_class=1 for face detection
        "data/facedata/facedatatrain",
        "data/facedata/facedatatrainlabels",
        "data/facedata/facedatatest",
        "data/facedata/facedatatestlabels"
    )

if __name__ == "__main__":
    main()