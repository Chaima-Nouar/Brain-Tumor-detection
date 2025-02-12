import cv2
import numpy as np
import matplotlib.pyplot as plt

# 2. Function to compute histograms for regions
def ComputeHist(regions_list, bins=16, plot=True):
    hists = []
    for i, region in enumerate(regions_list):
        hist = cv2.calcHist([region], [0], None, [bins], [0, 256])  # Histogram for grayscale
        hist = hist.flatten()  # Flatten the histogram
        hists.append(hist)

        if plot:
            plt.subplot(1, len(regions_list), i + 1)
            plt.plot(hist)
            plt.title(f"Region {i+1}")
            plt.xlim([0, bins])

    if plot:
        plt.show()

    return hists
