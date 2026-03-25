import os
from datasets import load_dataset

# Set persistent cache directory for SSP Cloud DataLab
# /home/onyxia/work persists across sessions
cache_dir = os.path.join("/home/onyxia/work", ".cache", "huggingface/datasets")
os.makedirs(cache_dir, exist_ok=True)

# Get token from environment variable
token = os.getenv("HF_TOKEN")

# Load dataset with caching
ds = load_dataset(
    "marsianin500/Speech2Latex",
    cache_dir=cache_dir
)

print(f"✅ Dataset loaded successfully!")
print(f"📁 Cache directory: {cache_dir}")
print(f"📊 Dataset info: {ds}")