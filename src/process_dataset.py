# 3. Function to extract features from a dataset
import os
import csv
import cv2
import numpy as np
from src.split import split
from src.compute_hist import ComputeHist

def extract_features(dataset_path, regions=4, bins=16, save_to="features.csv"):

    features = []

    for filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error reading image: {image_path}")
            continue

        regions_list = split(image, regions=regions, plot=False)
        hists = ComputeHist(regions_list, bins=bins, plot=False)

        # Combine all histograms into a single feature vector
        feature_vector = np.concatenate(hists).tolist()
        features.append(feature_vector)

    # Save features to CSV
    with open(save_to, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [f"feature_{i+1}" for i in range(len(features[0]))]
        writer.writerow(header)
        writer.writerows(features)

    print(f"Features saved to {save_to}")


tumor_dataset_path = r"C:\Users\HP\OneDrive\Bureau\ImageRegionFeatures\data\tumor"
notumor_dataset_path = r"C:\Users\HP\OneDrive\Bureau\ImageRegionFeatures\data\notumor"

# Extract features from tumor and notumor datasets and save to separate CSV files
extract_features(tumor_dataset_path, regions=4, bins=16, save_to="tumor_features.csv")
extract_features(notumor_dataset_path, regions=4, bins=16, save_to="notumor_features.csv")