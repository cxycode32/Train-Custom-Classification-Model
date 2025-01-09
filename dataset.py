import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset


class ScreenshotDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label