from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("model/house_price_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["OverallQual"]),
            float(request.form["GrLivArea"]),
            float(request.form["TotalBsmtSF"]),
            float(request.form["GarageCars"]),
            float(request.form["FullBath"]),
            float(request.form["YearBuilt"])
        ]

        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
