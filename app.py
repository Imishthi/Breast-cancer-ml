import os
import numpy as np
import pickle
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your_secret_key'

# Load models
logreg_model = pickle.load(open("models/best_model.pkl", "rb"))       # Risk prediction model
detection_model = pickle.load(open("models/model.pkl", "rb"))         # Numeric detection model
cnn_model = load_model("model_cnn.h5")                                # CNN image classifier

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

# New About Section Routes
@app.route("/about/what-is-breast-cancer")
def what_is_breast_cancer():
    return render_template("what-is-breast-cancer.html")

@app.route("/about/early-detection")
def early_detection():
    return render_template("early-detection.html")

@app.route("/about/diagnosis")
def diagnosis():
    return render_template("diagnosis.html")

@app.route("/about/stages")
def stages():
    return render_template("stages.html")

@app.route("/about/types")
def types():
    return render_template("types.html")

@app.route("/about/treatment")
def treatment():
    return render_template("treatment.html")

@app.route("/choose", methods=["GET", "POST"])
def choose():
    if request.method == "POST":
        choice = request.form.get("choice")
        if choice == "predict":
            return redirect(url_for("predict"))
        elif choice == "detect":
            return redirect(url_for("detect"))
        elif choice == "image":
            return redirect(url_for("detect_image"))
    return render_template("choose.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/detect_image")
def detect_image():
    return render_template("detect_image.html")

@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    try:
        features = [
            int(request.form["Family_History"]),
            int(request.form["Genetic_Mutation"]),
            float(request.form["BMI"]),
            int(request.form["Age"]),
            int(request.form["Previous_Breast_Biopsy"]),
        ]
        features_array = np.array([features])
        prediction = logreg_model.predict(features_array)[0]
        result = "⚠ High Risk of Developing Cancer" if prediction == 1 else "✅ Low Risk of Developing Cancer"
        return render_template("predict.html", prediction_text=result)
    except Exception as e:
        error_message = f"⚠ Error: {str(e)}"
        return render_template("predict.html", prediction_text=error_message)

@app.route("/detect_cancer", methods=["POST"])
def detect_cancer():
    detection_text = None
    try:
        features = [float(request.form.get(f)) for f in [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"
        ]]
        features_array = np.array([features])
        prediction = detection_model.predict(features_array)[0]
        detection_text = "⚠ Malignant (Cancerous)" if prediction == 1 else "✅ Benign (Non-Cancerous)"
    except Exception as e:
        detection_text = f"⚠ Error: {str(e)}"

    return render_template("detect.html", detection_text=detection_text)

@app.route("/classify_image", methods=["POST"])
def classify_image():
    if "image_file" not in request.files:
        flash("⚠ No image uploaded!", "error")
        return redirect(url_for("detect_image"))

    file = request.files["image_file"]
    if file.filename == "":
        flash("⚠ No selected file!", "error")
        return redirect(url_for("detect_image"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        image_cv = cv2.imread(filepath)
        if image_cv is None:
            raise ValueError("Invalid image format.")

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # 1. Check grayscale level (color variance)
        b, g, r = cv2.split(image_cv)
        color_diff = np.std(b - g) + np.std(b - r)
        is_grayscale_like = color_diff < 15  # Allow small variation

        # 2. Check for texture (Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_textured_like_ultrasound = laplacian_var > 10  # Adjusted threshold

        if not (is_grayscale_like and is_textured_like_ultrasound):
            raise ValueError("The image does not appear to be an ultrasound scan.")

    except Exception as e:
        flash(f"⚠ Rejected: {str(e)}", "error")
        os.remove(filepath)
        return redirect(url_for("detect_image"))

    try:
        # Prepare image for CNN
        img = load_img(filepath, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = cnn_model.predict(img_array)
        class_index = np.argmax(prediction[0])
        labels = ["benign", "malignant", "normal"]
        result = f"🧠 Predicted Class: {labels[class_index]}"
        return render_template("detect_image.html", prediction=result, image_path="/" + filepath)

    except Exception as e:
        flash(f"⚠ Error in image processing: {str(e)}", "error")
        os.remove(filepath)
        return redirect(url_for("detect_image"))



# Utility function
def secure_filename(filename):
    return filename.replace(" ", "_").replace("..", "").lower()

if __name__ == "__main__":
    app.run(debug=True)
