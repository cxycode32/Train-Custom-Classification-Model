import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from dataset import ScreenshotDataset
from utils import load_model, save_model, get_transform, check_accuracy, plot_metrics, plot_confusion_matrix, visualize_augmentations
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 1024
CLASS_NUM = 3
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
EPOCH_NUM = 80
EARLY_STOPPING_PATIENCE = 8
WEIGHT_DECAY = 1e-5

# Change root directory to the directory where your datasets (images) are
ROOT_DIR = "your_datasets"

# You need to create your own CSV file containing data labels
# Currently data_labels_sample.csv already contains sample format for your reference
# You can edit it to tailor to your need
CSV_FILE = "data_labels_sample.csv"

# Change to your own model file name if any
MODEL_FILE = "model.pth.tar"


def load_data(dataset):
    """Set training, validation, and testing data size, then load the dataset."""
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """Train the model and print loss for each epoch."""
    best_val_accuracy = 0
    patience_counter = 0
    train_result = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy calculation
            _, preds = outputs.max(1)
            correct_train += (preds == targets).sum()
            total_train += targets.size(0)

        train_loss /= len(train_loader)
        train_accuracy = (correct_train / total_train) * 100

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)

                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Accuracy calculation
                _, preds = outputs.max(1)
                correct_val += (preds == targets).sum()
                total_val += targets.size(0)

        val_loss /= len(val_loader)
        val_accuracy = (correct_val / total_val) * 100

        # Update learning rate scheduler
        if scheduler:
            scheduler.step(val_loss)

        # Record metrics
        train_result["train_loss"].append(train_loss)
        train_result["val_loss"].append(val_loss)
        train_result["train_accuracy"].append(train_accuracy.cpu().numpy())
        train_result["val_accuracy"].append(val_accuracy.cpu().numpy())

        # Print metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Accuracy: {val_accuracy:.2f}%")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, optimizer, MODEL_FILE)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping to prevent overfitting
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    return train_result


def visualize_result(training_result, train_loader, val_loader, test_loader, model):
    """Visualize the training result with loss and accuracy plot, and confusion matrix."""
    plot_metrics(training_result)

    train_accuracy = check_accuracy(train_loader, model, device)
    print(f"Training Set Accuracy: {train_accuracy:.2f}")

    val_accuracy = check_accuracy(val_loader, model, device)
    print(f"Validation Set Accuracy: {val_accuracy:.2f}")

    test_accuracy = check_accuracy(test_loader, model, device)
    print(f"Test Set Accuracy: {test_accuracy:.2f}")

    plot_confusion_matrix(test_loader, model, device)


def main():
    model = torchvision.models.googlenet(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=INPUT_SIZE, out_features=CLASS_NUM)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, verbose=True)

    if os.path.exists(MODEL_FILE):
        user_input = input(f"Model file '{MODEL_FILE}' detected. Do you want to load and train this model? (yes/no): ").strip().lower()

        if user_input == "yes":
            load_model(model, optimizer, MODEL_FILE)

    print("Model Summary:")
    summary(model, input_size=(3, INPUT_SIZE, INPUT_SIZE))

    transform = get_transform()

    dataset = ScreenshotDataset(
        csv_file=CSV_FILE,
        root_dir=ROOT_DIR,
        transform=transform,
    )

    train_loader, val_loader, test_loader = load_data(dataset)

    visualize_augmentations(dataset)

    training_result = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCH_NUM, device)

    visualize_result(training_result, train_loader, val_loader, test_loader, model)


if __name__ == "__main__":
    main()