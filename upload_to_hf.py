from huggingface_hub import HfApi, create_repo
import os

# Initialize the API
api = HfApi()

# Your HuggingFace username
username = "vdhanraj"
repo_name = "neurosymbolic-llm"
repo_id = f"{username}/{repo_name}"

# Create the repository (if it doesn't exist)
try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"Repository {repo_id} created/exists")
except Exception as e:
    print(f"Repository might already exist: {e}")

# Define your model paths
encoder_path = "Programs/models/encoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_seed_42.pth"
decoder_path = "Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_seed_42.pth"
decoder_finetuned_path = "Programs/models/decoders_linear_encoder_3_digit_data_pre_generated_dataset_x_gt_y_seed_42_post_fine_tuning_bnzd6vw8_2025_05_19.pth"

# Upload the encoder
print("Uploading encoder...")
api.upload_file(
    path_or_fileobj=encoder_path,
    path_in_repo="encoders_seed_42.pth",
    repo_id=repo_id,
    repo_type="model"
)

# Upload the decoder
print("Uploading decoder...")
api.upload_file(
    path_or_fileobj=decoder_path,
    path_in_repo="decoders_seed_42.pth",
    repo_id=repo_id,
    repo_type="model"
)

# Upload the fine-tuned decoder
print("Uploading fine-tuned decoder...")
api.upload_file(
    path_or_fileobj=decoder_finetuned_path,
    path_in_repo="decoders_seed_42_finetuned.pth",
    repo_id=repo_id,
    repo_type="model"
)

print("All models uploaded successfully!")
