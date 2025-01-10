import os
import torch
import albumentations as A
from matplotlib import pyplot as plt
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_model(model, optimizer, model_path):
    """Load model."""
    if not os.path.exists(model_path):
        print(f"[ERROR] (utils.py) Error loading model file: {model_path} not found.")
        return

    print("Loading model......")
    state = torch.load(model_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print("Model loaded!")


def save_model(model, optimizer, model_path):
    """Save model."""
    print("Saving model......")
    state = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, model_path)
    print("Model saved!")


def get_transform():
    """Return the data augmentation pipeline."""
    return A.Compose(
        [
            A.Resize(width=224, height=224),
            A.RandomCrop(width=200, height=200),
            A.Rotate(limit=30, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.8),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=0.3),
                    A.ColorJitter(p=0.3),
                ],
                p=1.0,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def check_accuracy(loader, model, device):
    """Check the accuracy of the model."""
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            _, preds = outputs.max(1)
            num_correct += (preds == targets).sum().item()
            num_samples += targets.size(0)

    accuracy = 100 * num_correct / num_samples

    return accuracy


def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize an image tensor."""
    mean = torch.tensor(mean).view(1, 1, 3)
    std = torch.tensor(std).view(1, 1, 3)
    return (image * std) + mean


def visualize_augmentations(dataset, num_samples=5):
    for i in range(num_samples):
        image, label = dataset[i]
        image = denormalize(image.permute(1, 2, 0)).cpu().numpy()  # Denormalize and CHW -> HWC

        plt.figure()
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()


def plot_metrics(history):
    """Plot loss and accuracy trends."""
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss Trends")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Trends")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(loader, model, device, title="Confusion Matrix"):
    """Plot confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(cm)))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()
