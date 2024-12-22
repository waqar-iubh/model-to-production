from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import time  # For timestamp
import threading

app = Flask(__name__)

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ensure this matches your MLflow server

# Load model from MLflow using sklearn flavor
MODEL_URI = "models:/AnomalyDetectionModel/1"
model = mlflow.sklearn.load_model(MODEL_URI)

# Load pre-fitted scaler
scaler = joblib.load("scaler.pkl")

# Feature names expected by the model
FEATURE_NAMES = ['temperature', 'humidity', 'sound_volume']

# Start a single MLflow run for monitoring
mlflow.set_experiment("IntegratedMonitoring")
monitoring_run = mlflow.start_run(run_name="IntegratedMonitoring", nested=False)
run_id = monitoring_run.info.run_id

# Initialize aggregate metrics
aggregate_metrics = {
    "total_predictions": 0,
    "total_normal": 0,
    "total_anomalies": 0,
}
step_counter = 0  # Sequential step counter
lock = threading.Lock()  # Prevent race conditions in multithreading


@app.route('/predict', methods=['POST'])
def predict():
    try:
        global monitoring_run, run_id, step_counter

        # Parse the input JSON
        data = request.get_json()

        # Convert to a DataFrame with feature names
        input_df = pd.DataFrame([data], columns=FEATURE_NAMES)

        # Standardize the input features
        standardized_input = scaler.transform(input_df)
        standardized_df = pd.DataFrame(standardized_input, columns=FEATURE_NAMES)  # Restore column names

        # Predict using sklearn model methods
        prediction = model.predict(standardized_df)

        # Update aggregate metrics
        with lock:
            aggregate_metrics["total_predictions"] += 1
            if prediction[0] == 1:
                aggregate_metrics["total_normal"] += 1
            else:
                aggregate_metrics["total_anomalies"] += 1

            step_counter += 1  # Increment step counter

        # Log metrics to the ongoing MLflow run
        with mlflow.start_run(run_id=run_id):
            # Option 1: Use timestamp as step
            timestamp_step = int(time.time())  # Unix timestamp
            mlflow.log_metric("total_predictions", aggregate_metrics["total_predictions"], step=timestamp_step)
            mlflow.log_metric("total_normal", aggregate_metrics["total_normal"], step=timestamp_step)
            mlflow.log_metric("total_anomalies", aggregate_metrics["total_anomalies"], step=timestamp_step)

            # Option 2: Use sequential step counter
            mlflow.log_metric("total_predictions_sequential", aggregate_metrics["total_predictions"], step=step_counter)
            mlflow.log_metric("total_normal_sequential", aggregate_metrics["total_normal"], step=step_counter)
            mlflow.log_metric("total_anomalies_sequential", aggregate_metrics["total_anomalies"], step=step_counter)

        return jsonify({
            "prediction": int(prediction[0])  # Convert to Python int
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

