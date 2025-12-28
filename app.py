from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("student.pkl", "rb"))

@app.route("/")
def home():
    return render_template("student.html")

@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    subject = request.form["subject"]

    hours = float(request.form["hours"])
    prev_score = float(request.form["prev_score"])
    activity = 1 if request.form["activity"] == "Yes" else 0
    sleep = float(request.form["sleep"])
    papers = int(request.form["papers"])

    input_data = np.array([[hours, prev_score, activity, sleep, papers]])
    prediction = round(model.predict(input_data)[0], 2)

    return render_template(
        "result.html",
        name=name,
        subject=subject,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
