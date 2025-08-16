import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from loaddata import loadLabelsFile, loadImagesFile


"""
define 3 layer model class
"""
class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden1=256, hidden2=128, output_size=10):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



"""
use loaddata.py functions to create torch tensors
"""
def prepare_data(image_file, label_file, image_width, task="binary", positive_class=1):
    labels = list(map(int, loadLabelsFile(label_file)))
    num_labels = len(labels)
    image_sections = loadImagesFile(image_file, num_labels, image_width)

    if task == "multiclass":
        # Normalize pixel values: 0 → 0.0, 1 → 0.5, 2 → 1.0
        data = [
            [pixel / 2.0 for row in section for pixel in row]
            for section in image_sections
        ]
    else:
        # For binary classification (faces), keep binarized input
        data = [
            [1 if pixel != 0 else 0 for row in section for pixel in row]
            for section in image_sections
        ]

    if task == "binary":
        labels = [1 if label == positive_class else 0 for label in labels]
        y_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    else:  # multiclass
        y_tensor = torch.tensor(labels, dtype=torch.long)

    x_tensor = torch.tensor(data, dtype=torch.float32)
    return x_tensor, y_tensor



"""
define training function
"""
def train_model(model, x_train, y_train, x_test, y_test, task="binary", epochs=10, lr=0.001):
    if task == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    start_time = time.time()

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    end_time = time.time()

    model.eval()
    with torch.no_grad():
        outputs = model(x_test)

        if task == "binary":
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            acc = (preds == y_test).float().mean().item()
        else:
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_test).float().mean().item()

    return acc, round(end_time - start_time, 3)



"""
define function to run demonstration of prediction vs actual 
"""
def demo_predictions(model, x_test, y_test, task="binary", num_samples=10):
    model.eval()
    indices = random.sample(range(len(x_test)), num_samples)
    x_sample = x_test[indices]
    y_sample = y_test[indices]

    with torch.no_grad():
        outputs = model(x_sample)

        if task == "binary":
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float().view(-1)
            labels = y_sample.view(-1)
        else:
            preds = torch.argmax(outputs, dim=1)
            labels = y_sample 

    print("\nPredictions:\n")
    for i in range(num_samples):
        true_label = int(labels[i].item())
        predicted_label = int(preds[i].item())
        print(f"Sample {i+1}: Predicted = {predicted_label} | Actual = {true_label}")



"""
define plotting function
"""
def plot_all_metrics(percentages, times, accuracies, stds, label):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(percentages, times, marker='o', color='green')
    axs[0].set_title(f'{label} - Training Time')
    axs[0].set_xlabel('Training %')
    axs[0].set_ylabel('Time (s)')
    axs[0].grid(True)

    axs[1].plot(percentages, accuracies, marker='o', color='blue')
    axs[1].set_title(f'{label} - Accuracy')
    axs[1].set_xlabel('Training %')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(True)

    axs[2].plot(percentages, stds, marker='o', color='red')
    axs[2].set_title(f'{label} - Std Deviation')
    axs[2].set_xlabel('Training %')
    axs[2].set_ylabel('Standard Deviation')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()



"""
loop to run pytorch training 
"""
def run_pytorch_experiment(x_train, y_train, x_test, y_test, input_size, task, label=""):
    percentages = [i/10 for i in range(1, 11)]
    accuracies, stds, times = [], [], []

    for pct in percentages:
        size = int(pct * len(x_train))
        acc_trials = []
        time_total = 0

        for trial in range(3):
            indices = random.sample(range(len(x_train)), size)
            x_sub = x_train[indices]
            y_sub = y_train[indices]
            output_size = 1 if task == "binary" else 10
            model = ThreeLayerNN(input_size, output_size=output_size)

            acc, elapsed = train_model(model, x_sub, y_sub, x_test, y_test, task=task)
            acc_trials.append(acc)
            time_total += elapsed

            # Run demo at 100% training set
            if trial == 0 and pct == 1.0:
                demo_predictions(model, x_test, y_test, task=task)

        avg_acc = np.mean(acc_trials)
        std_acc = np.std(acc_trials)
        avg_time = round(time_total / 3, 3)

        accuracies.append(avg_acc)
        stds.append(std_acc)
        times.append(avg_time)

        print(f"\n{label} - {int(pct*100)}% | Accuracy: {avg_acc:.4f}, Std: {std_acc:.4f}, Time: {avg_time}s")

    # Plot results
    # After training loop ends
    percentages_float = [i/10 for i in range(1, 11)]
    plot_all_metrics(percentages_float, times, accuracies, stds, label)

if __name__ == "__main__":
    # run digits training and testing
    print("\nRunning Digit Classification (0–9)...")
    x_train, y_train = prepare_data(
        "data/digitdata/trainingimages",
        "data/digitdata/traininglabels",
        image_width=28,
        task="multiclass"
    )
    x_test, y_test = prepare_data(
        "data/digitdata/testimages",
        "data/digitdata/testlabels",
        image_width=28,
        task="multiclass"
    )
    run_pytorch_experiment(x_train, y_train, x_test, y_test, input_size=784, task="multiclass", label="Digit Classification")


    # run faces training and testing
    print("\nRunning Face Detection (binary)...")
    x_train, y_train = prepare_data(
        "data/facedata/facedatatrain",
        "data/facedata/facedatatrainlabels",
        image_width=60,
        task="binary",
        positive_class=1
    )
    x_test, y_test = prepare_data(
        "data/facedata/facedatatest",
        "data/facedata/facedatatestlabels",
        image_width=60,
        task="binary",
        positive_class=1
    )
    run_pytorch_experiment(x_train, y_train, x_test, y_test, input_size=4200, task="binary", label="Face Detection")


