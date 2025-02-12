import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Function to split an image into regions
def split(image, regions=4, plot=True):
    h, w = image.shape[:2]
    rows = cols = int(np.sqrt(regions))  # Ensure regions is a perfect square
    region_height = h // rows
    region_width = w // cols
    regions_list = []

    for i in range(rows):
        for j in range(cols):
            region = image[i * region_height:(i + 1) * region_height,
                           j * region_width:(j + 1) * region_width]
            regions_list.append(region)

            if plot:
                plt.subplot(rows, cols, i * cols + j + 1)
                plt.imshow(region, cmap='gray')
                plt.axis('off')

    if plot:
        plt.show()

    return regions_list
