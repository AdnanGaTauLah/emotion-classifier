
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

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/Nn-n/emotion-classifier-meld"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({"inputs": "I'm feeling excited!"})
print(output)
```

## Unreal Engine Usage

1. Use HTTP Request Plugin or Blueprint to POST to the API URL.
2. Include a header with Bearer token authorization.
3. Send a JSON object like: `{"inputs": "Your text here"}`
4. Parse the returned label and score.
