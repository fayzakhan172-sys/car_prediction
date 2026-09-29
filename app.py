import joblib, numpy as np, pandas as pd
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
bundle = joblib.load("car_model.pkl")
model, encoders, columns = bundle["model"], bundle["encoders"], bundle["columns"]
RENAME = {"EngineType": "Engine Type"}   # UI field -> CSV column


@app.get("/")
def home():
    return send_from_directory(".", "index.html")


@app.post("/predict")
def predict():
    try:
        d = request.get_json()
        df = pd.DataFrame([{RENAME.get(k, k): v for k, v in d.items()}])
        for col, enc in encoders.items():
            df[col] = enc.transform(df[col])
        price = float(np.exp(model.predict(df[columns])[0]))
        return jsonify(price=price)
    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
