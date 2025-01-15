import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from dataset import ScreenshotDataset
from utils import load_model, save_model, get_transform


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Here are hyperparameters
# You can experiment with different values to find what work best for you
INPUT_SIZE = 1024
CLASSES = ["0", "1", "2"]
CLASS_NUM = 3
CLASS_1_NUM = 50
CLASS_2_NUM = 325
CLASS_3_NUM = 264
LEARNING_RATES = [1e-2, 1e-3, 1e-4, 1e-5]
BATCH_SIZES = [1, 32, 64, 128, 256, 1024]
EPOCH_NUM = 10
EARLY_STOPPING_PATIENCE = 8
WEIGHT_DECAY = 0.0
LOAD_MODEL = False

# Change root directory to the directory where your datasets (images) are
ROOT_DIR = "your_datasets"

# You need to create your own CSV file containing data labels
# Currently data_labels_sample.csv already contains sample format for your reference
# You can edit it to tailor to your need
CSV_FILE = "data_labels.csv"

# Change to your own model file name if any
MODEL_FILE = "model.pth.tar"


def load_data(dataset, batch_size):
    """Set training, validation, and testing data size, then load the dataset."""
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def train_model(device, learning_rate, batch_size, model, criterion, optimizer, writer, train_loader, val_loader):
    """Train the model and print loss for each epoch."""
    step = 0
    best_val_acc = 0
    patience_counter = 0

    # Visualize model using TensorBoard
    images, _ = next(iter(train_loader))
    images = images.to(torch.float32).to(DEVICE)
    writer.add_graph(model, images)
    writer.close()

    for epoch in range(EPOCH_NUM):
        # Training phase
        model.train()
        train_loss_epoch = []
        train_acc_epoch = []
        correct_train = 0
        total_train = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets) # Compute loss
            train_loss_batch = loss.item()
            train_loss_epoch.append(train_loss_batch)

            # Backward pass
            optimizer.zero_grad() # Clear previous gradients
            loss.backward() # Compute gradients
            optimizer.step() # Update weights

            # Calculate accuracy
            _, preds = outputs.max(1)
            correct_train = (preds == targets).sum()
            total_train += targets.size(0)
            train_acc_batch = float(correct_train) / float(data.shape[0])
            train_acc_epoch.append(train_acc_batch)

            # Visualization
            img_grid = torchvision.utils.make_grid(data)
            writer.add_image("Training Image", img_grid)
            writer.add_scalar("Batch Training Loss", train_loss_batch, step)
            writer.add_scalar("Batch Training Accuracy", train_acc_batch, step)

            if batch_idx == len(train_loader) - 1:
                features = data.reshape(data.shape[0], -1)
                class_labels = [CLASSES[label] for label in preds]
                writer.add_embedding(
                    features,
                    metadata=class_labels,
                    label_img=data,
                    global_step=step
                )

            step += 1

        train_acc = sum(train_acc_epoch) / len(train_acc_epoch)
        train_loss = sum(train_loss_epoch) / len(train_loss_epoch)

        writer.add_hparams(
            {"lr": learning_rate, "bsize": batch_size},
            {
                "accuracy": train_acc,
                "loss": train_loss,
            }
        )

        # Validation phase
        model.eval()
        val_loss_epoch = []
        val_acc_epoch = []
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                data, targets = data.to(device), targets.to(device)

                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, targets) # Compute validation loss
                val_loss_batch = loss.item()
                val_loss_epoch.append(val_loss_batch)

                # Calculate accuracy
                _, preds = outputs.max(1)
                correct_val = (preds == targets).sum()
                total_val += targets.size(0)
                val_acc_batch = float(correct_val) / float(data.shape[0])
                val_acc_epoch.append(val_acc_batch)

                # Visualization
                img_grid = torchvision.utils.make_grid(data)
                writer.add_image("Validation Image", img_grid)
                writer.add_scalar("Batch Validation Loss", val_loss_batch, step)
                writer.add_scalar("Batch Validation Accuracy", val_acc_batch, step)

        val_acc = sum(val_acc_epoch) / len(val_acc_epoch)
        val_loss =  sum(val_loss_epoch) / len(val_loss_epoch)

        # Print metrics
        print(f"Epoch [{epoch+1}/{EPOCH_NUM}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"Train Accuracy: {train_acc * 100:.2f}%, "
            f"Validation Accuracy: {val_acc * 100:.2f}%"
        )

        # Save the best model based on validation accuracy
        avg_val_acc = (val_acc * 100) if val_acc else 0.0
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_model(model, optimizer, MODEL_FILE)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if validation accuracy doesn't improve (to prevent overfitting)
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break


def test_model(device, model, loader):
    correct_predictions = 0
    total_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            scores = model(x)
            _, preds = scores.max(1)
            correct_predictions += (preds == y).sum()
            total_samples += preds.size(0)

    accuracy = correct_predictions / total_samples * 100
    print(f"Test Set Accuracy: {accuracy:.2f}")


def main():
    # Auto-clean previous runs/ folder
    if os.path.exists("runs"):
        print("Cleaning up previous runs folder......")
        shutil.rmtree("runs")
        print("Previous runs folder removed!")

    transform = get_transform()
    dataset = ScreenshotDataset(
        csv_file=CSV_FILE,
        root_dir=ROOT_DIR,
        transform=transform,
    )

    for batch_size in BATCH_SIZES:
        for learning_rate in LEARNING_RATES:
            print(f"BATCH SIZE: {batch_size} | LEARNING RATE: {learning_rate}")

            # Initialization
            model = torchvision.models.googlenet(weights="DEFAULT")
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(in_features=INPUT_SIZE, out_features=CLASS_NUM)
            model.to(DEVICE)

            if LOAD_MODEL:
                load_model(model, optimizer, MODEL_FILE)

            # Adjust weights for imbalanced datasets
            # Remove weights if you have balanced datasets
            class_counts = [CLASS_1_NUM, CLASS_2_NUM, CLASS_3_NUM]
            total_samples = sum(class_counts)
            weights = [total_samples / class_count for class_count in class_counts]
            weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weights)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
            writer = SummaryWriter(f"runs/googlenet/MiniBatchSize_{batch_size}_LR_{learning_rate}")

            train_loader, val_loader, test_loader = load_data(dataset, batch_size)

            train_model(DEVICE, learning_rate, batch_size, model, criterion, optimizer, writer, train_loader, val_loader)

            test_model(DEVICE, model, test_loader)


if __name__ == "__main__":
    main()