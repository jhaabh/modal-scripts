# test-keys.py

import os
import modal

# === CONFIGURATIONS ===
# VOLUME_NAME: The name of the pre-configured Modal volume used to cache downloaded HuggingFace models.
VOLUME_NAME = "huggingface-model-cache"  # Replace with your pre-setup volume name

# MODELS_DIR: Directory path on the Modal container where the models will be stored.
MODELS_DIR = "/models"  # Path on the Modal machine

# HUGGINGFACE_SECRET_NAME: The name of the Modal secret that contains your HuggingFace API token (HF_TOKEN).
HUGGINGFACE_SECRET_NAME = "huggingface-secret"  # Secret containing HF_TOKEN

# WANDB_SECRET_NAME: The name of the Modal secret that contains your WandB API key (WANDB_API_KEY).
WANDB_SECRET_NAME = "wandb-secret"  # Secret containing WANDB_API_KEY

# === MODAL SETUP ===
# Configure a shared Modal volume for model caching.
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Define a Modal image with dependencies installed.
image = (
    modal.Image.debian_slim()
    .pip_install(
        "huggingface_hub",  # For downloading HuggingFace models
        "hf-transfer",  # Speeds up HuggingFace model downloads
        "wandb",  # For tracking experiments in WandB
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable faster model downloads
            "HUGGINGFACE_HUB_CACHE": MODELS_DIR,  # Set the cache directory for models
        }
    )
)

# Define the Modal app.
app = modal.App(name="test-keys", image=image)


# === TEST FUNCTION ===
# This function tests:
# - Access to HuggingFace API token
# - Access to WandB API key and ability to log in
# - The ability to download a HuggingFace model
@app.function(
    volumes={MODELS_DIR: volume},  # Attach the shared volume for storing models
    secrets=[
        modal.Secret.from_name(HUGGINGFACE_SECRET_NAME),  # Add HuggingFace secret
        modal.Secret.from_name(WANDB_SECRET_NAME),  # Add WandB secret
    ],
)
def test_modal_setup(model_name: str, test_type: str):
    """
    Tests the Modal setup for HuggingFace and WandB integration.

    Args:
        model_name (str): The HuggingFace model repository name to download.
        test_type (str): The type of test to run. Options:
            - "huggingface": Test HuggingFace API token.
            - "wandb": Test WandB API key and login.
            - "model": Test HuggingFace model download.
            - "all": Run all the above tests.

    Returns:
        dict: A dictionary with the results of each test.
    """
    from huggingface_hub import snapshot_download
    import wandb

    results = {}  # Store results of each test

    # Test HuggingFace API Token
    if test_type == "huggingface" or test_type == "all":
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("HuggingFace token is accessible.")
            results["huggingface"] = "Token accessible"
        else:
            print("HuggingFace token is missing.")
            results["huggingface"] = "Token missing"

    # Test WandB API Key
    if test_type == "wandb" or test_type == "all":
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            print("WandB API key is accessible.")
            try:
                wandb.login()
                results["wandb"] = "Key accessible and verified"
            except Exception as e:
                print(f"Error during WandB login: {e}")
                results["wandb"] = f"Key accessible but login failed: {e}"
        else:
            print("WandB API key is missing.")
            results["wandb"] = "Key missing"

    # Test HuggingFace Model Download
    if test_type == "model" or test_type == "all":
        try:
            print(f"Downloading model: {model_name}")
            snapshot_download(
                repo_id=model_name,  # Model repository name
                local_dir=MODELS_DIR,  # Download to the models directory
                token=os.getenv("HF_TOKEN"),  # Use the HuggingFace API token
            )
            print(f"Model downloaded successfully to {MODELS_DIR}/{model_name}")
            volume.commit()  # Commit changes to the shared volume
            results["model"] = "Downloaded successfully"
        except Exception as e:
            print(f"Failed to download model: {e}")
            results["model"] = f"Error: {e}"

    return results


# === LOCAL ENTRYPOINT ===
# This is the main entrypoint when running the script locally.
@app.local_entrypoint()
def main(
    model: str = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM", test: str = "all"
):
    """
    Main entrypoint for testing Modal setup.

    Args:
        model (str): HuggingFace model to download. Defaults to a small test model.
        test (str): Type of test to run. Options:
            - "huggingface": Test HuggingFace API token.
            - "wandb": Test WandB API key and login.
            - "model": Test HuggingFace model download.
            - "all": Run all the above tests.

    Raises:
        ValueError: If an invalid test type is provided.
    """
    if test not in ["huggingface", "wandb", "model", "all"]:
        raise ValueError("test must be one of: huggingface, wandb, model, all")

    print(f"Testing Modal setup with model: {model} and test type: {test}")
    # Run the test function remotely in Modal
    results = test_modal_setup.remote(model, test)
    print("Test Results:", results)
