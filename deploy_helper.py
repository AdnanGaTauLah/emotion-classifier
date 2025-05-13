#!/usr/bin/env python
import os
import sys
import getpass
from pathlib import Path
import subprocess
import time

def check_model_files(model_path):
    """Check if model files exist in the specified directory."""
    model_dir = Path(model_path)
    
    if not model_dir.exists():
        return False, f"Model directory {model_path} does not exist."
    
    # Check for essential model files
    model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.json"))
    
    if not model_files:
        return False, f"No model files found in {model_path}."
    
    return True, f"Found {len(model_files)} model files in {model_path}."

def write_env_file(token):
    """Write the HF token to .env file."""
    env_path = Path(".env")
    
    # Add to existing .env if it exists
    if env_path.exists():
        env_content = env_path.read_text()
        if "HF_TOKEN" not in env_content:
            with open(env_path, "a") as f:
                f.write(f"\nHF_TOKEN={token}\n")
        else:
            # Replace existing token
            lines = []
            for line in env_content.splitlines():
                if line.startswith("HF_TOKEN="):
                    lines.append(f"HF_TOKEN={token}")
                else:
                    lines.append(line)
            
            env_path.write_text("\n".join(lines) + "\n")
    else:
        # Create new .env file
        with open(env_path, "w") as f:
            f.write(f"HF_TOKEN={token}\n")
    
    return True

def update_config(model_path, repo_name, username):
    """Update the configuration in deploy.py."""
    deploy_path = Path("deploy.py")
    
    if not deploy_path.exists():
        return False, "deploy.py not found."
    
    content = deploy_path.read_text()
    
    # Replace the configuration section
    config_start = content.find("# ==== USER CONFIGURATION ====")
    config_end = content.find("# ============================", config_start)
    
    if config_start == -1 or config_end == -1:
        return False, "Could not find configuration section in deploy.py."
    
    new_config = f'''# ==== USER CONFIGURATION ====
MODEL_PATH = "{model_path}"
REPO_NAME = "{repo_name}"
HF_USERNAME = "{username}"  # ✅ Make sure this is your real HF username
HF_TOKEN = os.getenv("HF_TOKEN")

README_PATH = "./README.md"  # Save API usage info here
# ============================'''
    
    new_content = content[:config_start] + new_config + content[config_end + len("# ============================"):]
    deploy_path.write_text(new_content)
    
    return True, "Updated configuration in deploy.py."

def main():
    print("=" * 60)
    print("Hugging Face Deployment Helper")
    print("=" * 60)
    
    # Step 1: Check for model files
    model_path = input("Enter the path to your model directory [models/fold_1]: ") or "models/fold_1"
    success, message = check_model_files(model_path)
    print(message)
    
    if not success:
        print("\nWould you like to train the model first? (y/n)")
        if input().lower() == 'y':
            print("\nRunning training...")
            subprocess.run(["python", "train.py"])
            # Check again
            success, message = check_model_files(model_path)
            print(message)
            if not success:
                print("Model training completed but files still not found. Please check your model path.")
                return
        else:
            print("Please train your model or specify a valid model path.")
            return
    
    # Step 2: Get HF credentials
    print("\nHugging Face Configuration:")
    username = input("Enter your Hugging Face username: ")
    while not username:
        print("Username cannot be empty.")
        username = input("Enter your Hugging Face username: ")
    
    repo_name = input("Enter repository name [emotion-classifier-meld]: ") or "emotion-classifier-meld"
    
    print("\nNow you need to provide your Hugging Face token with write permission.")
    print("You can create one at https://huggingface.co/settings/tokens")
    token = getpass.getpass("Enter your Hugging Face token: ")
    
    while not token:
        print("Token cannot be empty.")
        token = getpass.getpass("Enter your Hugging Face token: ")
    
    # Step 3: Write to .env
    write_env_file(token)
    print("\n✅ Token saved to .env file.")
    
    # Step 4: Update configuration
    success, message = update_config(model_path, repo_name, username)
    print(message)
    
    # Step 5: Deploy
    print("\nReady to deploy your model to Hugging Face!")
    print(f"The model will be deployed to: https://huggingface.co/{username}/{repo_name}")
    print("Proceed with deployment? (y/n)")
    
    if input().lower() == 'y':
        print("\nDeploying model...")
        subprocess.run(["python", "deploy.py"])
        
        print("\n🎉 Deployment process completed!")
        print(f"Check your model at: https://huggingface.co/{username}/{repo_name}")
        print(f"API endpoint: https://api-inference.huggingface.co/models/{username}/{repo_name}")
    else:
        print("\nDeployment cancelled.")

if __name__ == "__main__":
    main() 