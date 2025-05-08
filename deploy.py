from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from datetime import datetime

# ==== USER CONFIGURATION ====
MODEL_PATH = "./models/fold_1"
REPO_NAME = "emotion-classifier-meld"
HF_USERNAME = "Nn-n"  # ✅ Make sure this is your real HF username
HF_TOKEN = os.getenv("HF_TOKEN")

README_PATH = "./README.md"  # Save API usage info here
# ============================

# Safety check
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Make sure it's in your .env file and correctly loaded.")

API_DEMO = f"""```python
import requests

API_URL = "https://api-inference.huggingface.co/models/{HF_USERNAME}/{REPO_NAME}"
headers = {{"Authorization": "Bearer YOUR_HF_TOKEN"}}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({{"inputs": "I'm feeling excited!"}})
print(output)
```"""

# Generate README.md file with API instructions
def write_readme():
    with open(README_PATH, "w") as f:
        f.write(f"""
---
language: en
tags:
- text-classification
- emotion-recognition
pipeline_tag: text-classification
---

# Emotion Classifier (MELD)

This model classifies text input into emotional categories, trained on MELD.

## API Usage

{API_DEMO}

## Unreal Engine Usage

1. Use HTTP Request Plugin or Blueprint to POST to the API URL.
2. Include a header with Bearer token authorization.
3. Send a JSON object like: `{{"inputs": "Your text here"}}`
4. Parse the returned label and score.
""")

def deploy():
    api = HfApi()

    # Create repo on the Hub
    print(f"Creating or updating repo: {HF_USERNAME}/{REPO_NAME}")
    create_repo(
        repo_id=f"{HF_USERNAME}/{REPO_NAME}",
        exist_ok=True,
        repo_type="model",
        token=HF_TOKEN
    )

    # Upload model files (include bin, json, tokenizer files)
    print("Uploading model files...")
    upload_folder(
        folder_path=MODEL_PATH,
        repo_id=f"{HF_USERNAME}/{REPO_NAME}",
        token=HF_TOKEN,
        allow_patterns=["*.bin", "*.json", "*.txt", "*.model", "*.safetensors"]
    )

    # Upload README
    print("Uploading README...")
    api.upload_file(
        path_or_fileobj=README_PATH,
        path_in_repo="README.md",
        repo_id=f"{HF_USERNAME}/{REPO_NAME}",
        token=HF_TOKEN
    )

    print(f"\n✅ Model deployed successfully!")
    print(f"🌍 Access it at: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
    print(f"📡 API Endpoint: https://api-inference.huggingface.co/models/{HF_USERNAME}/{REPO_NAME}")

if __name__ == "__main__":
    write_readme()
    deploy()
