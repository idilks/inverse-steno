import os
import requests
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

api_key = os.getenv("DARTMOUTH_CHAT_API_KEY")
if not api_key:
    raise ValueError("DARTMOUTH_CHAT_API_KEY not found in .env")

resp = requests.get(
    "https://chat.dartmouth.edu/api/models",
    headers={"Authorization": f"bearer {api_key}"},
    timeout=60,
)

print("status:", resp.status_code)
resp.raise_for_status()

data = resp.json()
models = [m["id"] for m in data["data"]]
pprint(models)