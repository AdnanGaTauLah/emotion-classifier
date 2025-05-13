from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv
import os
import sys
import time
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

# Load environment variables from .env file
load_dotenv()

from datetime import datetime

# ==== USER CONFIGURATION ====
MODEL_PATH = "./models/fold_1"
REPO_NAME = "emotion-classifier-meld"
HF_USERNAME = "YourUsername"  # ✅ Make sure this is your real HF username
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_CARD_PATH = "./model_card.md"  # Better model card 
# ============================

# Safety check
if not HF_TOKEN:
    print(f"{Fore.RED}ERROR: HF_TOKEN is not set. Make sure it's in your .env file and correctly loaded.{Style.RESET_ALL}")
    sys.exit(1)

# Check if model files exist
model_dir = Path(MODEL_PATH)
if not model_dir.exists():
    print(f"{Fore.RED}ERROR: Model directory {MODEL_PATH} does not exist.{Style.RESET_ALL}")
    sys.exit(1)

model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.json")) + list(model_dir.glob("*.safetensors"))
if not model_files:
    print(f"{Fore.RED}ERROR: No model files found in {MODEL_PATH}.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Hint: Make sure you've trained your model and it saved files to {MODEL_PATH}{Style.RESET_ALL}")
    sys.exit(1)

# Check if model card exists
if not os.path.exists(MODEL_CARD_PATH):
    print(f"{Fore.YELLOW}Warning: Model card not found at {MODEL_CARD_PATH}. Using minimal README instead.{Style.RESET_ALL}")
    # Use the old README approach as fallback
    MODEL_CARD_PATH = "./README.md"
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
    with open(MODEL_CARD_PATH, "w") as f:
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
else:
    # Update the model card with the correct username
    with open(MODEL_CARD_PATH, "r") as f:
        model_card = f.read()
    
    # Replace placeholder with actual username
    model_card = model_card.replace("[YourUsername]", HF_USERNAME)
    
    with open(MODEL_CARD_PATH, "w") as f:
        f.write(model_card)
    
    print(f"{Fore.GREEN}Updated model card with your username.{Style.RESET_ALL}")

def deploy():
    api = HfApi()

    # Create repo on the Hub
    print(f"\n{Fore.CYAN}Step 1/3: Creating or updating repo: {HF_USERNAME}/{REPO_NAME}{Style.RESET_ALL}")
    try:
        create_repo(
            repo_id=f"{HF_USERNAME}/{REPO_NAME}",
            exist_ok=True,
            repo_type="model",
            token=HF_TOKEN
        )
        print(f"{Fore.GREEN}Repository created/updated successfully.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error creating repository: {str(e)}{Style.RESET_ALL}")
        return False

    # Upload model files (include bin, json, tokenizer files)
    print(f"\n{Fore.CYAN}Step 2/3: Uploading model files... {len(model_files)} files found{Style.RESET_ALL}")
    print(f"This may take some time depending on your internet connection speed and model size.")
    
    try:
        upload_folder(
            folder_path=MODEL_PATH,
            repo_id=f"{HF_USERNAME}/{REPO_NAME}",
            token=HF_TOKEN,
            allow_patterns=["*.bin", "*.json", "*.txt", "*.model", "*.safetensors", "*.tokenizer", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"],
            ignore_patterns=["*.md"]
        )
        print(f"{Fore.GREEN}Model files uploaded successfully.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error uploading model files: {str(e)}{Style.RESET_ALL}")
        return False

    # Upload model card
    print(f"\n{Fore.CYAN}Step 3/3: Uploading model card...{Style.RESET_ALL}")
    try:
        api.upload_file(
            path_or_fileobj=MODEL_CARD_PATH,
            path_in_repo="README.md",
            repo_id=f"{HF_USERNAME}/{REPO_NAME}",
            token=HF_TOKEN
        )
        print(f"{Fore.GREEN}Model card uploaded successfully.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error uploading model card: {str(e)}{Style.RESET_ALL}")
        return False

    return True

if __name__ == "__main__":
    print(f"{Fore.CYAN}Starting Hugging Face model deployment...{Style.RESET_ALL}")
    print(f"Username: {HF_USERNAME}")
    print(f"Repository: {REPO_NAME}")
    print(f"Model path: {MODEL_PATH}")
    
    start_time = time.time()
    success = deploy()
    elapsed_time = time.time() - start_time
    
    if success:
        print(f"\n{Fore.GREEN}✅ Model deployed successfully in {elapsed_time:.1f} seconds!{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Your model is now available at:{Style.RESET_ALL}")
        print(f"🌍 Repository: {Fore.YELLOW}https://huggingface.co/{HF_USERNAME}/{REPO_NAME}{Style.RESET_ALL}")
        print(f"📡 API Endpoint: {Fore.YELLOW}https://api-inference.huggingface.co/models/{HF_USERNAME}/{REPO_NAME}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Example API usage:{Style.RESET_ALL}")
        print(f"""
{Fore.WHITE}import requests

API_URL = "https://api-inference.huggingface.co/models/{HF_USERNAME}/{REPO_NAME}"
headers = {{"Authorization": "Bearer YOUR_HF_TOKEN"}}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({{"inputs": "I'm feeling excited!"}})
print(output){Style.RESET_ALL}
        """)
        
        print(f"{Fore.CYAN}Next steps:{Style.RESET_ALL}")
        print("1. Check your model on the Hugging Face Hub")
        print("2. Test the API with real inputs")
        print("3. Share your model with others!")
    else:
        print(f"\n{Fore.RED}❌ Deployment failed. Please check the errors above.{Style.RESET_ALL}") 