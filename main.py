import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

import config
from dataset import ScreenshotDataset, get_dataloader
from utils import (
    clear_directories,
    get_transform,
    save_checkpoint,
    load_checkpoint
)


def train_model(model, criterion, optimizer, writer, scaler, train_loader, batch_size, learning_rate, device=config.DEVICE):
    step = 0

    # Visualize model using TensorBoard
    images, _ = next(iter(train_loader))
    images = images.to(torch.float32).to(device)
    writer.add_graph(model, images)
    writer.close()

    for epoch in range(config.EPOCH_NUM):
        model.train()
        train_loss_epoch = []
        train_acc_epoch = []
        correct_train = 0
        total_train = 0
        
        loop = tqdm(
            train_loader,
            total=len(train_loader),
            leave=False,
            desc=f"EPOCH [{epoch+1}/{config.EPOCH_NUM}]"
        )

        for batch_idx, (data, targets) in enumerate(loop):
            data, targets = data.to(device), targets.to(device)

            with autocast():
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, targets)
                train_loss_batch = loss.item()
                train_loss_epoch.append(train_loss_batch)

            # Backward pass
            optimizer.zero_grad() # Clear previous gradients
            scaler.scale(loss).backward() # Compute gradients
            scaler.step(optimizer) # Update weights
            scaler.update()

            # Calculate accuracy
            _, preds = outputs.max(1)
            correct_train = (preds == targets).sum()
            total_train += targets.size(0)
            train_acc_batch = float(correct_train) / float(data.shape[0])
            train_acc_epoch.append(train_acc_batch)
            
            loop.set_postfix(loss=train_loss_batch, acc=train_acc_batch)

            # Visualization
            img_grid = torchvision.utils.make_grid(data)
            writer.add_image("Training Image", img_grid)
            writer.add_scalar("Batch Training Loss", train_loss_batch, step)
            writer.add_scalar("Batch Training Accuracy", train_acc_batch, step)

            if batch_idx + 1 == len(train_loader):
                features = data.reshape(data.shape[0], -1)
                class_labels = [config.CLASS_LABELS[label] for label in preds]
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

        print(f"Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_acc * 100:.2f}%, "
        )
        
        if config.SAVE_MODEL:
            save_checkpoint(epoch+1, batch_size, learning_rate, model, optimizer)

    
def valid_model(model, criterion, writer, val_loader, device=config.DEVICE):
    model.eval()
    val_loss_epoch = []
    val_acc_epoch = []
    correct_val = 0
    total_val = 0
    
    step=0
    loop = tqdm(
        val_loader,
        total=len(val_loader),
        leave=False
    )

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
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
            
            loop.set_description(f"BATCH [{batch_idx+1}/{len(val_loader)}]")
            loop.set_postfix(loss=val_loss_batch, acc=val_acc_batch)

            # Visualization
            img_grid = torchvision.utils.make_grid(data)
            writer.add_image("Validation Image", img_grid)
            writer.add_scalar("Batch Validation Loss", val_loss_batch, step)
            writer.add_scalar("Batch Validation Accuracy", val_acc_batch, step)

            step += 1

    val_acc = sum(val_acc_epoch) / len(val_acc_epoch)
    val_loss =  sum(val_loss_epoch) / len(val_loss_epoch)
    
    print(f"Validation Loss: {val_loss:.4f}, "
        f"Validation Accuracy: {val_acc * 100:.2f}%, "
    )


def test_model(model, loader, device=config.DEVICE):
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
    clear_directories()

    transform = get_transform()
    dataset = ScreenshotDataset(
        dataset_dir=config.DATASET_DIR,
        csv_file=config.CSV_FILE,
        transform=transform,
    )

    for batch_size in config.BATCH_SIZES:
        for learning_rate in config.LEARNING_RATES:
            print(f"BATCH SIZE: {batch_size} | LEARNING RATE: {learning_rate}")

            model = torchvision.models.googlenet(weights="DEFAULT")
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(in_features=config.INPUT_SIZE, out_features=config.NUM_CLASSES)
            model.to(config.DEVICE)

            if config.LOAD_MODEL:
                load_checkpoint(model, optimizer)

            # Adjust weights for imbalanced datasets
            # Remove weights if you have balanced datasets
            weights = [config.TOTAL_NUM_SAMPLES / count for count in config.SAMPLES_PER_CLASS]
            weights = torch.tensor(weights, dtype=torch.float).to(config.DEVICE)
            criterion = nn.CrossEntropyLoss(weight=weights)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
            writer = SummaryWriter(f"{config.LOG_DIR}/BS_{batch_size}_LR_{learning_rate}")
            scaler = GradScaler()

            train_loader, val_loader, test_loader = get_dataloader(dataset, batch_size)

            if config.TRAIN_MODEL:
                train_model(model, criterion, optimizer, writer, scaler, train_loader, batch_size, learning_rate)

            if config.VALID_MODEL:
                valid_model(model, criterion, writer, val_loader)

            if config.TEST_MODEL:
                test_model(model, test_loader)


if __name__ == "__main__":
    main()