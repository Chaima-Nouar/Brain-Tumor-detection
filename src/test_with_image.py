import cv2
import matplotlib.pyplot as plt
from src.stats import calculate_mean_intensity,threshold


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