import argparse
import requests
import json
from transformers import pipeline
import sys
import os
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

def local_inference(text, model_path="./models/fold_1"):
    """Run inference using the local model."""
    print(f"{Fore.CYAN}Running local inference using model at {model_path}{Style.RESET_ALL}")
    
    try:
        # Load the model pipeline
        classifier = pipeline("text-classification", model=model_path)
        
        # Get prediction
        result = classifier(text)[0]
        
        return result
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return None

def api_inference(text, api_url, token=None):
    """Run inference using the HuggingFace API."""
    print(f"{Fore.CYAN}Sending request to Hugging Face API at {api_url}{Style.RESET_ALL}")
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    payload = {"inputs": text}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Emotion Classifier Demo")
    parser.add_argument("--text", type=str, help="Text to classify", required=False)
    parser.add_argument("--mode", type=str, choices=["local", "api"], default="local", 
                       help="Inference mode: local or api")
    parser.add_argument("--model_path", type=str, default="./models/fold_1", 
                       help="Path to local model")
    parser.add_argument("--api_url", type=str, 
                       help="Hugging Face API URL (required for API mode)")
    parser.add_argument("--token", type=str, help="Hugging Face API token")
    
    args = parser.parse_args()
    
    # If text is not provided, use a list of sample texts
    sample_texts = [
        "I'm so excited about this!",
        "That makes me really angry.",
        "I feel sad about what happened.",
        "I'm so surprised by the results!",
        "I'm disgusted by this behavior.",
        "That's scary, I'm afraid.",
        "I don't feel anything special about it."
    ]
    
    texts_to_process = [args.text] if args.text else sample_texts
    
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Emotion Classifier Demo{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"Mode: {args.mode}")
    
    for text in texts_to_process:
        print(f"\n{Fore.YELLOW}Input: \"{text}\"{Style.RESET_ALL}")
        
        if args.mode == "local":
            result = local_inference(text, args.model_path)
            if result:
                label = result["label"]
                score = result["score"]
                print(f"{Fore.GREEN}Emotion: {label}{Style.RESET_ALL}")
                print(f"Confidence: {score:.4f}")
        else:  # api mode
            if not args.api_url:
                print(f"{Fore.RED}Error: API URL is required for API mode.{Style.RESET_ALL}")
                print(f"Example: python demo.py --mode api --api_url https://api-inference.huggingface.co/models/username/emotion-classifier-meld")
                sys.exit(1)
                
            result = api_inference(text, args.api_url, args.token)
            if result:
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                if "error" in result:
                    print(f"{Fore.RED}API Error: {result['error']}{Style.RESET_ALL}")
                else:
                    if isinstance(result, list):
                        # Handle array response format
                        for item in result:
                            print(f"{Fore.GREEN}Emotion: {item['label']}{Style.RESET_ALL}")
                            print(f"Confidence: {item['score']:.4f}")
                    else:
                        # Handle single result format
                        label = result.get("label", "unknown")
                        score = result.get("score", 0)
                        print(f"{Fore.GREEN}Emotion: {label}{Style.RESET_ALL}")
                        print(f"Confidence: {score:.4f}")
    
    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 