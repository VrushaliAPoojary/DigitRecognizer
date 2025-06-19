from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

print("Fetching MNIST digits...")
digits = load_digits()
X, y = digits.data, digits.target

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Save model
model_path = "backend/model"
os.makedirs(model_path, exist_ok=True)
joblib.dump(clf, os.path.join(model_path, "digit_model.pkl"))

print("✅ Model trained and saved successfully!")
