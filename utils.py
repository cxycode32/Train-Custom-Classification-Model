import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_model(model, optimizer, model_path):
    """Load model."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Error loading model file: {model_path} not found.")
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
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
            A.Rotate(limit=40, p=0.8, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            ToTensorV2(),
        ]
    )
