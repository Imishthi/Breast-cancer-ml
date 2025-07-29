import pickle
import numpy as np

# Load the saved model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Feature order:
# [Family_History, Genetic_Mutation, BMI, Age, Previous_Breast_Biopsy]
# Example input: Has family history, no genetic mutation, BMI 23.5, age 45, had previous biopsy
new_data = np.array([[1, 0, 23.5, 45, 1]])

# Make prediction
prediction = model.predict(new_data)

# Output result
if prediction[0] == 1:
    print("Prediction: ⚠ High Risk of Developing Cancer")
else:
    print("Prediction: ✅ Low Risk of Developing Cancer")

