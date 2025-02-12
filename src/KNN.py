import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('/data/X_train.csv')
X_test = pd.read_csv('/data/X_test.csv')
y_train = pd.read_csv('/data/y_train.csv')
y_test = pd.read_csv('/data/y_test.csv')

# Ensure labels are Series
y_train = y_train.squeeze()
y_test = y_test.squeeze()

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train
knn_model = KNeighborsClassifier(n_neighbors=3)
""" 5 neighbors: KNN Model Accuracy: 88.65%
7 neighbors: KNN Model Accuracy: 87.85%  """
knn_model.fit(X_train_scaled, y_train)

#predict
knn_predictions = knn_model.predict(X_test_scaled)

#evaluate
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"KNN Model Accuracy: {knn_accuracy * 100:.2f}%")
print("KNN Classification Report:\n", classification_report(y_test, knn_predictions))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_predictions))

joblib.dump(knn_model, '../models/knn_model.model')
joblib.dump(scaler, '../models/scaler_knn.pkl')

print("Model and scaler saved successfully.")

"""
 KNN Model Accuracy: 88.70%
KNN Classification Report:
               precision    recall  f1-score   support

         0.0       0.86      0.91      0.89       988
         1.0       0.91      0.86      0.89      1012

    accuracy                           0.89      2000
   macro avg       0.89      0.89      0.89      2000
weighted avg       0.89      0.89      0.89      2000

KNN Confusion Matrix:
 [[904  84]
 [142 870]]
 """