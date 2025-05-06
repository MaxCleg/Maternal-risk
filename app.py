from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
import os

model = joblib.load("maternal-risk-model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        sbp = float(request.form["sbp"])
        dbp = float(request.form["dbp"])
        bs = float(request.form["bs"])
        body_temp = float(request.form["body_temp"])
        heart_rate = float(request.form["heart_rate"])

        data = np.array([[age, sbp, dbp, bs, body_temp, heart_rate]])
        prediction = model.predict(data)[0]

        labels = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
        result = labels.get(prediction, "Unknown")

        return render_template("index.html", result=result)
    except:
        return render_template("index.html", result="Error: Invalid input")

if __name__ == "__main__":
    app.run(debug=True)
