# Anomaly Detection System for Wind Turbine Component Production

## 📋 Project Overview
Development of an anomaly detection system for a modern factory producing wind turbine components.

## 🎯 Objective
Identify and flag anomalies in sensor data (temperature, humidity, sound volume) to detect faults in the production cycle.

## ✨ Key Features
- Real-time data processing
- Integration of a predictive model (**Isolation Forest**) into a scalable service
- Continuous monitoring of system performance using **MLflow**

## 💡 Motivation
Provide a reliable decision-support system to:
- Minimize faults and improve manufacturing efficiency
- Support factory managers with actionable insights

# 🛠 Installation Guide

## 1. Install RabbitMQ Server
Run the following commands to install and start RabbitMQ:
```bash
sudo apt install -y rabbitmq-server
sudo systemctl enable rabbitmq-server
sudo systemctl start rabbitmq-server
```


## 2. Install Required Dependencies
Install all necessary Python packages:
```bash
pip install -r requirements.txt
```

## 3. Start MLflow Server
Launch the MLflow server for tracking experiments:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## 4. Start Prediction Service
Start the prediction service by navigating to the relevant directory:
```bash
cd prediction-service
python prediction_service.py
```

## 5. Start RabbitMQ Message Broker
Run the RabbitMQ consumer to handle messages:
```bash
cd consumer
python consumer.py
```

## 6. Start Sensor Data Generator

Run the producer script to simulate sensor data:
```bash
cd producer
python producer.py
```

## 7. Monitor Model Performance

Open the following link in your browser to monitor model performance:
```bash
http://127.0.0.1:5000
```
