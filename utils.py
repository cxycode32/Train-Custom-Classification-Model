import os
import shutil
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


def clear_directories(directories=config.DIRECTORIES):
    """
    Deletes all directories specified in the configuration file.
    
    This is useful for clearing previous training outputs, ensuring
    that new experiments start fresh without leftover data.
    """
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"{directory}/ deleted successfully!")


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


def get_checkpoint_filename(dir, epoch, batch_size, learning_rate):
    """
    Constructs the checkpoint filename based on the epoch, batch size, and learning rate.

    Args:
        dir (str): The directory where the model checkpoints are stored.
        epoch (int): The epoch number of the model checkpoint.
        batch_size (int): The batch size used during training.
        learning_rate (float): The learning rate used during training.

    Returns:
        str: The full file path of the checkpoint.
    """
    filename = f"{batch_size}_LR_{learning_rate}_EPOCH_{epoch}_model.pth"
    return os.path.join(dir, filename)


def save_checkpoint(epoch, batch_size, learning_rate, model, optimizer, dir=config.MODELS_DIR):
    """
    Saves the model and optimizer states as a checkpoint.

    Args:
        epoch (int): Epoch number.
        batch_size (int): Batch size.
        learning_rate: Learning rate.
        model (torch.nn.Module): The model whose state needs to be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state needs to be saved.
        dir (str, optional): Directory to store the checkpoint. Defaults to config.MODELS_DIR.
    """
    print("Saving checkpoint......")
    os.makedirs(dir, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    filepath = get_checkpoint_filename(dir, epoch, batch_size, learning_rate)
    torch.save(checkpoint, filepath)
    print("Checkpoint saved successfully.")
    
    
def get_user_choice(prompt, valid_choices, value_type):
    """
    Prompts the user to select a valid value from a given list.

    Args:
        prompt (str): The message displayed to the user.
        valid_choices (list): A list of valid values the user can choose from.
        value_type (type): The expected type of the user input (e.g., int or float).

    Returns:
        value_type: The user's selected valid value.
    """
    while True:
        print(f"Available options: {valid_choices}")
        user_input = input(prompt).strip()

        try:
            choice = value_type(user_input)
            if choice in valid_choices:
                return choice
            else:
                print(f"Invalid selection. Please choose from: {valid_choices}")
        except ValueError:
            print(f"Invalid input. Please enter a valid {value_type.__name__} value.")


def load_checkpoint(model, optimizer, dir=config.MODELS_DIR, device=config.DEVICE):
    """
    Loads a saved model checkpoint.

    Args:
        model (torch.nn.Module): The model where the checkpoint is loaded.
        optimizer (torch.optim.Optimizer): The optimizer where the checkpoint is loaded.
        dir (str, optional): Directory where the checkpoint is stored. Defaults to config.MODELS_DIR.

    Warning:
        If the checkpoint file does not exist, the function prints a warning and does not modify the model.
    """
    # Ask the user to select batch size and learning rate
    batch_size = get_user_choice("Enter batch size: ", config.BATCH_SIZES, int)
    learning_rate = get_user_choice("Enter learning rate: ", config.LEARNING_RATES, float)
    epoch = get_user_choice("Enter epoch number: ", range(1, 1000), int)
    
    checkpoint_path = get_checkpoint_filename(dir, epoch, batch_size, learning_rate)

    if not os.path.isfile(checkpoint_path):
        print(f"Warning: Checkpoint file '{checkpoint_path}' not found. Falling back without loading checkpoint.")
        return

    print("Loading checkpoint......")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Checkpoint loaded successfully.")