import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd

X_train = pd.read_csv('/data/X_train.csv')
X_test = pd.read_csv('/data/X_test.csv')
y_train = pd.read_csv('/data/y_train.csv')
y_test = pd.read_csv('/data/y_test.csv')

# Convert labels to Series
y_train = y_train.squeeze()
y_test = y_test.squeeze()

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train
svm_model = SVC(kernel='rbf', random_state=42)
""" i used kernel= linear and i got bad results : Accuracy: 77.45% 
 SVM Confusion Matrix:
 [[756 232]
 [219 793]] 
 """
svm_model.fit(X_train_scaled, y_train)

#predict
svm_predictions = svm_model.predict(X_test_scaled)

#Evaluate
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Model Accuracy: {svm_accuracy * 100:.2f}%")
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))

joblib.dump(svm_model, '/models/svm_model.model')
joblib.dump(scaler, '/models/scaler_svm.pkl')

print("SVM Model and scaler saved successfully.")

"""
SVM Model Accuracy: 84.35%
SVM Classification Report:
               precision    recall  f1-score   support

         0.0       0.79      0.93      0.85       988
         1.0       0.92      0.76      0.83      1012

    accuracy                           0.84      2000
   macro avg       0.85      0.84      0.84      2000
weighted avg       0.85      0.84      0.84      2000

SVM Confusion Matrix:
 [[920  68]
 [245 767]]
"""