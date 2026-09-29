# Run once locally:  python train.py
# Needs "1.04. Real-life example.csv" in the same folder.
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("1.04. Real-life example.csv")
df = df.dropna(subset=["Price", "EngineV"])
df = df[df["EngineV"] <= 10]
df["Log_price"] = np.log(df["Price"])
df = df.drop(columns=["Model", "Price"])

encoders = {}                      # ONE encoder per column (fixes the notebook bug)
for col in df.select_dtypes(include="object").columns:
    encoders[col] = LabelEncoder().fit(df[col])
    df[col] = encoders[col].transform(df[col])

X, y = df.drop(columns="Log_price"), df["Log_price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(alpha=0.001),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}
best, best_r2 = None, -1
for name, m in models.items():
    m.fit(X_tr, y_tr)
    r2 = r2_score(y_te, m.predict(X_te))
    print(f"{name:13s} R2 = {r2:.4f}")
    if r2 > best_r2:
        best, best_name, best_r2 = m, name, r2

print("Best model:", best_name)
joblib.dump({"model": best, "encoders": encoders, "columns": list(X.columns)},
            "car_model.pkl", compress=3)
print("Saved car_model.pkl")
