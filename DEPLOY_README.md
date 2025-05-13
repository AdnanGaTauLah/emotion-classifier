# Deploying the Emotion Classifier to Hugging Face

This guide provides a comprehensive walkthrough for deploying your trained emotion classifier model to Hugging Face Hub, making it easily accessible for inference via API.

## Prerequisites

Before deploying, ensure you have:

1. **A trained model**: Make sure you've trained the model and have files in the `models/fold_1` directory
2. **A Hugging Face account**: Create one at [huggingface.co](https://huggingface.co) if you don't have one
3. **Python environment**: Make sure all the required packages are installed

## Installation

First, install the required packages:

```bash
pip install -r requirements.txt
```

## Deployment Options

You have three ways to deploy your model:

### Option 1: Guided Deployment (Recommended)

Run the helper script for a step-by-step guided deployment process:

```bash
python deploy_helper.py
```

This script will:
1. Check if your model files exist
2. Prompt for your Hugging Face username and repository name
3. Help you set up your Hugging Face token
4. Update the configuration in deploy.py
5. Deploy your model to Hugging Face

### Option 2: Enhanced Deployment

For a more visual deployment experience with better error handling and feedback:

```bash
python enhanced_deploy.py
```

Make sure to update the configuration section in this file first:
- Set `HF_USERNAME` to your Hugging Face username
- Create a `.env` file with your Hugging Face token (`HF_TOKEN=your_token_here`)

### Option 3: Standard Deployment

Update the configuration in `deploy.py` and run:

```bash
python deploy.py
```

## Customizing Your Model Card

Before deployment, you can customize the model card that will appear on your Hugging Face model page:

1. Edit `model_card.md` to add specific details about your model
2. Include performance metrics, training details, and usage examples
3. The deployment scripts will automatically update your username in the model card

## Testing Your Deployed Model

After deployment, you can test your model using the included demo script:

### Local Testing:

```bash
python demo.py --mode local --model_path ./models/fold_1
```

### API Testing:

```bash
python demo.py --mode api --api_url https://api-inference.huggingface.co/models/YOUR_USERNAME/emotion-classifier-meld --token YOUR_HF_TOKEN
```

## Troubleshooting

### Common Issues:

1. **"No model files found"**
   - Make sure you've trained your model first
   - Check that files exist in the specified model directory

2. **"Authentication error"**
   - Verify your Hugging Face token has write permissions
   - Check that the token is correctly saved in the `.env` file

3. **"Repository creation failed"**
   - Make sure your Hugging Face username is correct
   - Check that the repository name doesn't already exist (or use `exist_ok=True`)

4. **API not working after deployment**
   - Allow some time for the model to be loaded on Hugging Face's infrastructure
   - Check that your model has the correct files and is properly formatted

## Using the API in Applications

Once deployed, you can use your model in any application by making API requests:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/YOUR_USERNAME/emotion-classifier-meld"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Test with sample text
output = query({"inputs": "I'm feeling excited!"})
print(output)
```

## Further Resources

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Inference API Documentation](https://huggingface.co/docs/api-inference/index)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index) 