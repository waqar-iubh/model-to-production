import pika
import json
import random
import time

# RabbitMQ connection setup
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare a queue
channel.queue_declare(queue='sensor_data')

# Simulate sensor data with similar distribution as training data
def simulate_sensor_data():
    while True:
        data = {
            "temperature": random.normalvariate(50, 10),  # Mean 50, std 10
            "humidity": random.normalvariate(70, 20),     # Mean 70, std 20
            "sound_volume": random.normalvariate(90, 30)  # Mean 90, std 30
        }
        yield data
        time.sleep(1)

# Publish data to RabbitMQ
for sensor_data in simulate_sensor_data():
    channel.basic_publish(exchange='', routing_key='sensor_data', body=json.dumps(sensor_data))
    print(f"Sent: {sensor_data}")

# Close the connection
connection.close()
