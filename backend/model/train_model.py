import os
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Fetching MNIST data...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print("Saving model...")
# ✅ Create directory if it doesn't exist
os.makedirs("backend/model", exist_ok=True)

# ✅ Save the model
joblib.dump(clf, "digit_model.pkl")

print("Model saved successfully!")
