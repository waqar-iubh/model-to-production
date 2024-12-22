import pika
import json
import requests

# RabbitMQ connection setup
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare a queue
channel.queue_declare(queue='sensor_data')

# Flask API URL
API_URL = "http://127.0.0.1:5001/predict"

# Callback to process messages
def callback(ch, method, properties, body):
    sensor_data = json.loads(body)
    print(f"Received: {sensor_data}")  # Log raw data
    try:
        response = requests.post(API_URL, json=sensor_data)
        print(f"Response: {response.json()}")  # Log API response
    except Exception as e:
        print(f"Exception occurred: {e}")

# Set up consumer
channel.basic_consume(queue='sensor_data', on_message_callback=callback, auto_ack=True)

print("Waiting for messages. To exit, press CTRL+C")
channel.start_consuming()
