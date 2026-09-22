import requests
import os

api_key = os.environ["GROQ_API_KEY"]
r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)
data = r.json()

for model in data.get("data", []):
    print(model["id"])
