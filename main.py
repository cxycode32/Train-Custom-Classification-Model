import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from dataset import ScreenshotDataset
from utils import get_transform, check_accuracy, plot_metrics, plot_confusion_matrix, visualize_augmentations
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 1024
CLASS_NUM = 3
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
EPOCH_NUM = 10

ROOT_DIR = "your_datasets"
CSV_FILE = "data_labels.csv"


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """Train the model and print loss for each epoch."""
    history = {"loss": [], "accuracy": []}
    best_accuracy = 0
    best_model_path = "best_model.pth"

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Accuracy after epoch
        accuracy = check_accuracy(train_loader, model, device)
        history["loss"].append(epoch_loss / len(train_loader))
        history["accuracy"].append(accuracy)

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
    return history


def main():
    # Data Augmentation
    transform = get_transform()

    # Dataset and DataLoader
    dataset = ScreenshotDataset(
        csv_file=CSV_FILE,
        root_dir=ROOT_DIR,
        transform=transform, #.ToTensor(),
    )
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

    # Freeze all layers except the last one, then update the last layer
    model = torchvision.models.googlenet(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=INPUT_SIZE, out_features=CLASS_NUM)
    model.to(device)

    # Print model summary
    print("Model Summary:")
    summary(model, input_size=(3, 224, 224))

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Visualize Augmented Samples
    visualize_augmentations(dataset)

    # Train the model
    history = train_model(model, train_loader, criterion, optimizer, EPOCH_NUM, device)

    # Plot training metrics
    plot_metrics(history)

    # Check model's accuracy
    print("Training Set Accuracy:")
    train_accuracy = check_accuracy(train_loader, model, device)
    print("Test Set Accuracy:")
    test_accuracy = check_accuracy(test_loader, model, device)

    # Confusion matrix for the test set
    plot_confusion_matrix(test_loader, model, device)


if __name__ == "__main__":
    main()