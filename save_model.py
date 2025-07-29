import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset.csv")

# Keep only the selected 5 features
selected_features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "diagnosis"]
data = data[selected_features]

# Convert diagnosis to numerical values
data["diagnosis"] = data["diagnosis"].map({'M': 1, 'B': 0})

# Split into features (X) and target variable (y)
X = data.drop(columns=["diagnosis"])
y = data["diagnosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model on only 5 features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the new model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print(" Model trained and saved successfully with only 5 features!")
