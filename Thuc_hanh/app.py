from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
mo_hinh = joblib.load("random_forest_model.pkl")

dac_trung_quan_trong = [
    "concave points_mean",
    "concave points_worst",
    "area_worst",
    "concavity_mean",
    "radius_worst"
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Lấy giá trị từ form
    gia_tri = [
        float(request.form["concave_points_mean"]),
        float(request.form["concave_points_worst"]),
        float(request.form["area_worst"]),
        float(request.form["concavity_mean"]),
        float(request.form["radius_worst"])
    ]

    # Dự đoán
    du_lieu_moi = pd.DataFrame([gia_tri], columns=dac_trung_quan_trong)
    du_doan = mo_hinh.predict(du_lieu_moi)
    ket_qua = "Ác tính" if du_doan[0] == 0 else "Lành tính"

    return render_template("index.html", ket_qua=ket_qua)

if __name__ == "__main__":
    app.run(debug=True)
