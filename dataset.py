import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split


class ScreenshotDataset(Dataset):
    def __init__(self, dataset_dir, csv_file, transform=None):
        super(ScreenshotDataset, self).__init__()
        self.data = []
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        self.annotations = pd.read_csv(csv_file)
        self.class_names = os.listdir(dataset_dir)

        for index, name in enumerate(self.class_names):
            folder_path = os.path.join(dataset_dir, name)
            files = os.listdir(folder_path)
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image, label = self.data[index]
        folder_path = os.path.join(self.dataset_dir, self.class_names[label])
        image = np.array(Image.open(os.path.join(folder_path, image)))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label


def get_dataloader(dataset, batch_size):
    """Set training, validation, and testing data size, then load the dataset."""
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader