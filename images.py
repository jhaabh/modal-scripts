import modal

MODELS_DIR = "/models"

UNSLOTH_IMAGE = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    .apt_install("git")
    .run_commands(
        'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
    )
    .run_commands(
        "pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes",
    )
    .pip_install("huggingface_hub", "hf-transfer", "wandb")
    .env(
        {
            "HUGGINGFACE_HUB_CACHE": MODELS_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

UNSLOTH_IMAGE_OLD = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"
    )  # Using older PyTorch version
    .apt_install("git")
    .pip_install(
        "torch==2.1.0",  # Match base image torch version
        "transformers>=4.37.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.41.0",
    )
    .run_commands(
        'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"'
    )
    .run_commands(
        "pip install --no-deps packaging ninja einops flash-attn==2.3.3 xformers==0.0.23 trl peft"
    )
    .pip_install("huggingface_hub", "hf-transfer", "wandb")
    .env(
        {
            "HUGGINGFACE_HUB_CACHE": MODELS_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

# Doesn't work yet
UNSLOTH_IMAGE_CONDA = (
    modal.Image.micromamba("3.11")
    .apt_install("git")
    .micromamba_install(
        "pytorch-cuda=12.1",
        "pytorch",
        "cudatoolkit",
        "xformers",
        channels=["pytorch", "nvidia", "xformers"],
    )
    .run_commands(
        'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
        "pip install --no-deps trl peft accelerate bitsandbytes",
    )
    .micromamba_install(
        "huggingface_hub", "hf-transfer", "wandb", channels=["conda-forge"]
    )
    .env(
        {
            "HUGGINGFACE_HUB_CACHE": MODELS_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)
