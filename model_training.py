import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

# Load the dataset
df = pd.read_csv("C:\Users\ASUS\cropifytcs\crop_prediction_dataset.csv")  # Make sure this matches your file name

# Fix target column name
X = df.drop("Crop", axis=1)
y = df["Crop"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestClassifier().fit(X_train, y_train)
mlp = MLPClassifier(max_iter=500).fit(X_train, y_train)
dt = DecisionTreeClassifier().fit(X_train, y_train)
nb = GaussianNB().fit(X_train, y_train)

# Save models
joblib.dump(rf, "C:\Users\ASUS\cropifytcs\random_forest.pkl)
joblib.dump(mlp, "C:\Users\ASUS\cropifytcs\MLP.pkl.py")
joblib.dump(dt,"C:\Users\ASUS\cropifytcs\random_tree.pkl.py")
joblib.dump(nb, "C:\Users\ASUS\cropifytcs\naive_bayes.pkl.py")

print("✅ Models trained and saved successfully.")