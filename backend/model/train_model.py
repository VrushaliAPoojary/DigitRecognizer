from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib
import os
param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [10, 20, 30, None]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

joblib.dump(grid.best_estimator_, os.path.join(model_path, "digit_model.pkl"))
print("Fetching MNIST digits...")
digits = load_digits()
X, y = digits.data, digits.target

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")
clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# Save model
model_path = "backend/model"
os.makedirs(model_path, exist_ok=True)
joblib.dump(clf, os.path.join(model_path, "digit_model.pkl"))

print("✅ Model trained and saved successfully!")
