# Training with Custom Datasets

This repository is a deep learning pipeline for classifying custom screenshots with data augmentation using PyTorch and Albumentations. The model is based on Google's GoogLeNet architecture and includes functionalities for data preprocessing, augmentation, training, validation, testing, and result visualization.


## Features

- **Custom Dataset Loader:** Handles loading and preprocessing of screenshots from a directory structure and CSV file.
- **Data Augmentation:** Includes rotation, flipping, and padding to enhance model generalization.
- **Training and Validation:** Implements training with early stopping and learning rate scheduling.
- **Visualization:** Provides confusion matrix, metric plots, and augmented image samples for analysis.
- **Model Persistence:** Saves and loads models for reuse and evaluation.


## Installation

Clone this repository to your local machine:
```bash
git clone https://github.com/cxycode32/Training-With-Custom-Datasets.git
cd Training-With-Custom-Datasets
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```


### File Structure
```
├── main.py                # Training script
├── utils.py               # Utility functions
├── dataset.py             # Custom dataset class
├── your_datasets/         # Your images
├── data_labels.csv        # CSV file with image labels
├── model.pth.tar          # Your model
├── requirements.txt       # Project dependencies
└── .gitignore             # Ignored files for Git
```


### Dataset Structure
```
your_datasets/
  ├── class1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── class2/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── ...

```
A CSV file named **data_labels.csv** should contain the mapping of filenames to their corresponding labels.


## Usage

Run the training script with default parameters:
```bash
python main.py
```


## Visualization

### Training Loss and Accuracy

The training loss and accuracy.

![Training Loss And Accuracy](./assets/training-loss-and-accuracy.png)


## Contribution

Feel free to fork this repository and submit pull requests to improve the project or add new features.


## License

This project is licensed under the MIT License.