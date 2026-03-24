import os
from huggingface_hub import hf_hub_download

# Path to the persistent cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface_hub")

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Replace 'your_huggingface_token' with your actual Hugging Face token
HUGGINGFACE_TOKEN = 'your_huggingface_token'

# Function to download the Speech2Latex dataset

def download_speech2latex():
    dataset_path = hf_hub_download(repo_id='speech2latex', cache_dir=CACHE_DIR, use_auth_token=HUGGINGFACE_TOKEN)
    print(f"Dataset downloaded to: {dataset_path}")

# Example usage
if __name__ == '__main__':
    download_speech2latex()