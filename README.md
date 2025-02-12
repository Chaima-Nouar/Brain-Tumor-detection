# Image Region Features - Machine Learning and Deep Learning Models for Tumor Detection

This project utilizes multiple models (Statistical, Machine Learning, and Deep Learning) to classify images into 'tumor' and 'no tumor' categories. The models use extracted features from image regions to perform the classification task. The project includes a mix of feature extraction techniques, data preprocessing, and model training for accurate tumor detection.

## Folders Overview:

- **`data/`**: Contains raw image datasets and CSV files with extracted features.
- The `tumor` and `notumor` files contain resized and preprocessed images used for training the models. Images are normalized, resized to 128x128 pixels, and converted to a uniform format to ensure consistency and better performance during model training.
- **`models/`**: Holds the trained models, including KNN, SVM, ANN, and CNN, along with scaler files for feature standardization.
- **`results/`**: Stores results such as figures, loss curves, and other output data.
- **`src/`**: Contains the source code for feature extraction, model training, testing, and other utilities.
- **`requirements.txt`**: Python dependencies for the project.

To get started, follow the instructions below.

## Requirements

- Python 3.x
- Required libraries listed in `requirements.txt`

## How to Run

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
