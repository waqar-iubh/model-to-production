import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib

# Generate simulated data
np.random.seed(42)
data = pd.DataFrame({
    "temperature": np.random.normal(loc=50, scale=10, size=1000),
    "humidity": np.random.normal(loc=70, scale=20, size=1000),
    "sound_volume": np.random.normal(loc=90, scale=30, size=1000)
})

# Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['temperature', 'humidity', 'sound_volume']])
scaled_df = pd.DataFrame(scaled_data, columns=['temperature', 'humidity', 'sound_volume'])

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(scaled_df)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved to scaler.pkl")

# Define input example
input_example = data.head(1)
signature = infer_signature(data[['temperature', 'humidity', 'sound_volume']], model.predict(scaled_df))

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ensure this matches your MLflow server
mlflow.set_experiment("AnomalyDetection")

with mlflow.start_run(run_name="IsolationForest_AnomalyDetection") as run:
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("contamination", 0.05)

    # Log the model with input example and signature
    model_artifact_path = "model"
    #mlflow.sklearn.log_model(model, artifact_path=model_artifact_path, input_example=input_example, signature=signature)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_artifact_path,
        input_example=input_example,
        signature=signature
    )
    print(f"Model logged at: {model_artifact_path}")

    # Register the model
    artifact_uri = f"runs:/{run.info.run_id}/{model_artifact_path}"
    registered_model = mlflow.register_model(model_uri=artifact_uri, name="AnomalyDetectionModel")
