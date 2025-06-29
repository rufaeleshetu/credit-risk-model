import requests

url = "http://127.0.0.1:8000/predict"

# Sample payload matching pydantic schema
data = {
    "feature1": 35,
    "feature2": "M",
    "feature3": 3.5,
    "feature4": "yes"
}

print("Sending data:", data)
response = requests.post(url, json=data)

print("Response:", response.json())
