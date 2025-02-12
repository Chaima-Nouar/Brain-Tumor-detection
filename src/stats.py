import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

tumor_path="C:/Users/HP/OneDrive/Bureau/ImageRegionFeatures/data/tumor"
notumor_path="C:/Users/HP/OneDrive/Bureau/ImageRegionFeatures/data/notumor"

# Function to calculate mean intensity for each image
def calculate_mean_intensity(image_path, regions=(4, 4)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Divide the image into regions
    h, w = image.shape
    region_h, region_w = h // regions[0], w // regions[1]
    mean_intensities = []

    for i in range(0, h, region_h):
        for j in range(0, w, region_w):
            region = image[i:i + region_h, j:j + region_w]
            mean_intensities.append(np.mean(region))

    return np.mean(mean_intensities)


# Function to classify based on mean intensity
def classify_image(mean_intensity, threshold):
    return "tumor" if mean_intensity <= threshold else "notumor"


# Load data and calculate mean intensities
def load_data_and_labels():
    data = [] #data: Contains the mean intensities of the images.
    labels = [] # labels: Contains the corresponding labels ("tumor" or "notumor").


    for label, folder_path in [("tumor", tumor_path), ("notumor", notumor_path)]:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                mean_intensity = calculate_mean_intensity(file_path)
                data.append(mean_intensity)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(data), np.array(labels)


# Main logic
data, labels = load_data_and_labels()

# Set threshold based on histogram analysis
threshold = np.mean(data)  # threshold = 44 : Accuracy =74

# Classify and calculate accuracy
predicted_labels = [classify_image(mi, threshold) for mi in data]
accuracy = accuracy_score(labels, predicted_labels)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(data[labels == "tumor"], bins=20, alpha=0.7, color="red", label="Tumor")
plt.hist(data[labels == "notumor"], bins=20, alpha=0.7, color="blue", label="No Tumor")
plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold: {threshold:.2f}")
plt.title("Threshold for Classification")
plt.xlabel("Mean Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# Set threshold based on histogram analysis
threshold = threshold - 10  #  44 - 10 =34 ==> Accuracy :78%

# Classify and calculate accuracy
predicted_labels = [classify_image(mi, threshold) for mi in data]
accuracy = accuracy_score(labels, predicted_labels)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(data[labels == "tumor"], bins=20, alpha=0.7, color="red", label="Tumor")
plt.hist(data[labels == "notumor"], bins=20, alpha=0.7, color="blue", label="No Tumor")
plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold: {threshold:.2f}")
plt.title("Threshold for Classification")
plt.xlabel("Mean Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.show()



# Function to test a new image and display classification percentages
def test_image(image_path, threshold):
    try:
        mean_intensity = calculate_mean_intensity(image_path)
        # This function likely divides the image into regions, calculates the mean intensity of each region, and returns the overall mean intensity.

        # Calculate the probability of each class based on the mean intensity
        distance_from_threshold = abs(mean_intensity - threshold)
        tumor_probability = max(0, 100 - (distance_from_threshold * 100 / threshold))  # Example formula
        notumor_probability = 100 - tumor_probability

        print(f"Mean Intensity: {mean_intensity:.2f}")
        print(f"Probability of Tumor: {tumor_probability:.2f}%")
        print(f"Probability of No Tumor: {notumor_probability:.2f}%")

        # Display image with the probabilities
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            plt.imshow(image)
            plt.title(f"Tumor: {tumor_probability:.2f}% | No Tumor: {notumor_probability:.2f}%")
            plt.axis('off')  # Hide axis
            plt.show()
        else:
            print("Could not read the test image.")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# Example usage to test a new image
test_image_path = "C:/Users/HP/OneDrive/Bureau/ImageRegionFeatures/data/tumor/11.jpg"  # Change this to the path of the image you want to test
test_image(test_image_path, threshold)
