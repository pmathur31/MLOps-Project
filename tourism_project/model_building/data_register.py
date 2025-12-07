from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

repo_id = "pragmat/Tourism"
repo_type = "dataset"

# Load token from environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

print("Current working directory:", os.getcwd())
print("Uploading folder:", os.path.abspath("tourism_project/data"))

# Check for dataset repo
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

# Upload folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
