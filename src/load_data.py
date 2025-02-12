import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("/data/processed_statistics_by_region.csv")

# Features (mean, std, entropy) and Labels (tumor=1, notumor=0)
X = data.drop(columns=['label'])
y = data['label']  # Labels (tumor or notumor)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

X_train.to_csv('../results/X_train.csv', index=False)
X_test.to_csv('../results/X_test.csv', index=False)
y_train.to_csv('../results/y_train.csv', index=False)
y_test.to_csv('../results/y_test.csv', index=False)

print("Train and test datasets have been saved as CSV files.")


