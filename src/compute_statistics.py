import pandas as pd
import numpy as np
from skimage.measure import shannon_entropy

def compute_statistics_for_region(region_features):
    mean = np.mean(region_features)
    std = np.std(region_features)
    entropy = shannon_entropy(region_features)
    return mean, std, entropy

tumor_csv = "C:/Users/HP/OneDrive/Bureau/ImageRegionFeatures/results/tumor_features.csv"
notumor_csv = "C:/Users/HP/OneDrive/Bureau/ImageRegionFeatures/results/notumor_features.csv"

tumor_features = pd.read_csv(tumor_csv)
notumor_features = pd.read_csv(notumor_csv)

# Add label to the features: 1 for tumor, 0 for notumor
tumor_features['label'] = 1
notumor_features['label'] = 0

# Combine the tumor and notumor datasets
all_features = pd.concat([tumor_features, notumor_features], ignore_index=True)

processed_data = []

num_regions = 4
bins_per_region = 16

for index, row in all_features.iterrows():
    feature_values = row[:-1].to_numpy()  # Convert to NumPy array, excluding the label

    # Split the feature values into regions
    region_features = np.array_split(feature_values, num_regions)

    #compute for each region
    region_stats = []
    for region in region_features:
        mean, std, entropy = compute_statistics_for_region(region)
        region_stats.extend([mean, std, entropy])

    processed_data.append(region_stats + [row['label']])

columns = []
for i in range(1, num_regions + 1):
    columns.append(f"region_{i}_mean")
    columns.append(f"region_{i}_std")
    columns.append(f"region_{i}_entropy")
columns.append("label")

#new DataFrame
processed_df = pd.DataFrame(processed_data, columns=columns)

processed_csv = "processed_statistics_by_region.csv"
processed_df.to_csv(processed_csv, index=False)

print(f"Processed statistics saved to {processed_csv}")