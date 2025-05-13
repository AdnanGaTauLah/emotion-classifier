# Deploying Emotion Classifier to Hugging Face

This guide explains how to deploy your emotion classifier to Hugging Face.

## Prerequisites

1. A trained model in the `models/fold_1` directory
2. A Hugging Face account
3. A Hugging Face API token with write access

## Step 1: Install Required Packages

Make sure you have the necessary packages installed:

```bash
pip install -r requirements.txt
pip install huggingface_hub python-dotenv
```

## Step 2: Create a Hugging Face Token

1. Go to [Hugging Face](https://huggingface.co/) and create an account if you don't have one
2. Go to Settings → Access Tokens
3. Create a new token with "Write" permission
4. Copy the token

## Step 3: Set Up Environment File

Create a `.env` file in the root of your project with your Hugging Face token:

```
HF_TOKEN=your_hugging_face_token_here
```

## Step 4: Modify Configuration (Optional)

Open `deploy.py` and update the configuration if needed:

```python
# ==== USER CONFIGURATION ====
MODEL_PATH = "./models/fold_1"  # Path to your trained model
REPO_NAME = "emotion-classifier-meld"  # Name for your Hugging Face repository
HF_USERNAME = "YourUsername"  # Update this to your Hugging Face username
HF_TOKEN = os.getenv("HF_TOKEN")  # This will be loaded from the .env file
# ============================
```

## Step 5: Run Deployment Script

```bash
python deploy.py
```

This will:
1. Create a Hugging Face repository
2. Upload your model files
3. Upload a README with API usage instructions

## Step 6: Verify Deployment

Once deployed, your model will be available at:
- Repository: `https://huggingface.co/YourUsername/emotion-classifier-meld`
- API Endpoint: `https://api-inference.huggingface.co/models/YourUsername/emotion-classifier-meld`

## API Usage

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/YourUsername/emotion-classifier-meld"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({"inputs": "I'm feeling excited!"})
print(output)
```

## Troubleshooting

If you encounter the error "No model files found in the specified directory":
1. Make sure you've trained your model and the files exist in `models/fold_1`
2. Run `python train.py` to train the model if needed

If you encounter "Authorization error":
1. Check that your HF_TOKEN is valid and has write permissions
2. Make sure the token is correctly loaded from the .env file 