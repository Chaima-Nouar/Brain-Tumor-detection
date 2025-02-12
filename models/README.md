# Models

This folder contains the trained models for the tumor classification task. The models are based on different algorithms: KNN, SVM, ANN, and CNN. Scaler files for feature normalization are also included.

## Files:

- **`ann_model50.pth`**: Trained ANN model (50 epochs).
- **`cnn_model20.pth`**: Trained CNN model (20 epochs).
- **`knn_model.model`**: Trained KNN model.
- **`svm_model.model`**: Trained SVM model.
- **`scaler_knn.pkl`**: Scaler used for KNN model feature normalization.
- **`scaler_svm.pkl`**: Scaler used for SVM model feature normalization.
- **`X_test.csv`**, **`X_train.csv`**, **`y_test.csv`**, **`y_train.csv`**: Datasets used for training and testing the models.

The models are used to predict whether an image contains a tumor or not based on features extracted from regions of the images.
