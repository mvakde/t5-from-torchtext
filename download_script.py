import os
import requests

# Define the URLs of the model and tokenizer files.
MODEL_URL = "https://download.pytorch.org/models/text/t5.base.v2.pt"
ENCODER_URL = "https://download.pytorch.org/models/text/t5.base.encoder.v2.pt"
GENERATION_URL = "https://download.pytorch.org/models/text/t5.base.generation.v2.pt"
TOKENIZER_URL = "https://download.pytorch.org/models/text/t5_tokenizer_base.model"
LOCAL_DIR = "t5_weights"
os.makedirs(LOCAL_DIR, exist_ok=True)


def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"File exists at {local_path}, skipping download")
        return
    print(f"Downloading {url} to {local_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we got an OK response

    with open(local_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("Download successful")


# Download both the model weights and the sentencepiece model
# download_file(MODEL_URL, os.path.join(LOCAL_DIR, os.path.basename(MODEL_URL)))
# download_file(ENCODER_URL, os.path.join(LOCAL_DIR, os.path.basename(ENCODER_URL)))
# download_file(GENERATION_URL, os.path.join(LOCAL_DIR, os.path.basename(GENERATION_URL)))
# download_file(TOKENIZER_URL, os.path.join(LOCAL_DIR, os.path.basename(TOKENIZER_URL)))

print("Finished download. Please ensure there are no errors above.")
