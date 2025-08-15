import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from mlflow.models import infer_signature

# =========================
# 1. Init DagsHub Tracking
# =========================
dagshub.init(repo_owner='sagargahlyan738', repo_name='mlflowexperiments', mlflow=True)

# =========================
# 2. Load Sample Data
# =========================
URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
data = pd.read_csv(URL, sep=";")

# Features & Target
X = data.drop(["quality"], axis=1)
y = data["quality"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# =========================
# 3. Training Parameters
# =========================
alpha = 0.5
l1_ratio = 0.5

# =========================
# 4. MLflow Experiment
# =========================
with mlflow.start_run():
    # Model Training
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(X_train, y_train)

    # Predictions
    preds = lr.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Log Params & Metrics
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")

    # Model Signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # =========================
    # Save model locally
    # =========================
    model_path = "model.pkl"
    joblib.dump(lr, model_path)

    # Log model file as artifact (safe for DagsHub)
    mlflow.log_artifact(model_path)

    # Optional: log input/output schema as artifact
    with open("model_signature.txt", "w") as f:
        f.write(str(signature))
    mlflow.log_artifact("model_signature.txt")

print("✅ Training complete and model logged to DagsHub artifacts.")
