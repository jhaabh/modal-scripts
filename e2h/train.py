from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List
import os
import json
from datetime import datetime

from .data import load_and_process_e2h_data
from .utils import calc_wandb_tags
from images import UNSLOTH_IMAGE
from .constants import (
    HF_DIR,
    HUGGINGFACE_SECRET_NAME,
    MODULES,
    RUNS_DIR,
    WANDB_SECRET_NAME,
    E2H_RAW_DATA_14k,
)
import modal

MODEL = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
RUN_FOLDER_BASE = Path("/runs/")

hf_volume = modal.Volume.from_name("e2h-hf-vol", create_if_missing=True)
runs_volume = modal.Volume.from_name("e2h-runs-3-vol", create_if_missing=True)

VOLUME_CONFIG = {
    HF_DIR: hf_volume,
    RUNS_DIR: runs_volume,
}

app = modal.App(name="e2h-train-test")

image = UNSLOTH_IMAGE

LORA_RANK = 64
LORA_ALPHA = 128


@dataclass
class ModelConfig:
    """Configuration for model loading and training"""

    model_name: str
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    lora_rank: int = LORA_RANK
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = 0
    learning_rate: float = 1e-5
    target_modules: Optional[List[str]] = None
    gradient_checkpoint: bool = False

    def to_dict(self):
        """Convert config to dictionary, handling None values and serializing sets to lists"""

        def serialize_value(v):
            if v is None:
                return "None"
            if isinstance(v, set):
                return sorted(list(v))
            return v

        return {k: serialize_value(v) for k, v in asdict(self).items()}

    def save_to_file(self, path: Path):
        """Save config to JSON file"""
        with open(path / "run_model_config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 10
    num_train_epochs: int = 1
    use_liger: bool = True
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "constant"
    seed: int = 3407
    evaluation_strategy: Optional[str] = None
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None

    def to_dict(self):
        """Convert config to dictionary, handling None values"""
        return {k: (v if v is not None else "None") for k, v in asdict(self).items()}

    def save_to_file(self, path: Path):
        """Save config to JSON file"""
        with open(path / "run_training_config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def save_run_metadata(
    run_folder: Path, dataset: str, cot_percentage: int, train_suffix: str
):
    """Save run metadata to JSON file"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "cot_percentage": cot_percentage,
        "train_suffix": train_suffix,
    }
    with open(run_folder / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


@app.function(
    volumes=VOLUME_CONFIG,
    image=image,
    secrets=[modal.Secret.from_name(HUGGINGFACE_SECRET_NAME)],
)
def download_data_and_model(
    model_name: str = MODEL,
    dataset: str = E2H_RAW_DATA_14k,
    cot_percentage: int = 0,
    train_suffix: str = "",
):
    from huggingface_hub import snapshot_download

    # Ensure the base model is downloaded
    try:
        download_path = snapshot_download(model_name, local_files_only=True)
        print(f"Checked {download_path}. Volume contains {model_name}.")
    except FileNotFoundError:
        print(f"Downloading {model_name} ...")
        download_path = snapshot_download(
            model_name,
            token=os.environ["HF_TOKEN"],
        )
        print("Committing HF directory (no progress bar) ...")
        print(f"Downloaded to {download_path}")
        VOLUME_CONFIG[HF_DIR].commit()

    # Write config and data into a training subfolder
    time_string = int(datetime.now().timestamp())
    run_name = f"e2h-{time_string}"
    if train_suffix := train_suffix.strip():
        run_name += f"-{train_suffix}"
    run_folder = RUN_FOLDER_BASE / run_name
    os.makedirs(run_folder)
    print(f"Preparing training run in {run_folder}.")

    # Save run metadata
    save_run_metadata(run_folder, dataset, cot_percentage, train_suffix)

    # Download and prepare data
    processed_dataset = load_and_process_e2h_data(
        dataset=dataset, test_split=0.05, cot_proportion=cot_percentage
    )
    processed_dataset.save_to_disk(run_folder / "data")
    VOLUME_CONFIG[RUNS_DIR].commit()

    return run_name, run_folder


@app.function(
    gpu=modal.gpu.A100(),
    image=image,
    volumes=VOLUME_CONFIG,
    timeout=3600 * 12,
    secrets=[
        modal.Secret.from_name(HUGGINGFACE_SECRET_NAME),
        modal.Secret.from_name(WANDB_SECRET_NAME),
    ],
)
def train(
    run_name: str,
    run_folder: Path,
    model_config: ModelConfig,
    tags: list[str],
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    epoch: int = 1,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTTrainer, SFTConfig
    import wandb
    from datasets import load_from_disk

    def load_data(run_folder: Path, use_small_subset: bool = False):
        dataset_all = load_from_disk(run_folder / "data")
        train_dataset = dataset_all["train"]
        eval_dataset = dataset_all.get("test", None)

        if use_small_subset:
            train_dataset = train_dataset.select(
                range(len(train_dataset) // 100)
            )  # Take first 1%

        # Print number of examples
        print(f"Number of training examples: {len(train_dataset)}")
        print(f"Number of eval examples: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def setup_tokenizer(model_name: str) -> AutoTokenizer:
        """
        Sets up and validates a tokenizer with proper pad and unk tokens.

        Args:
            model_name: Name/path of the model to load tokenizer for

        Returns:
            Configured tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad and unk tokens exist
        has_pad = bool(tokenizer.pad_token)
        has_unk = bool(tokenizer.unk_token)

        if not has_pad and has_unk:
            tokenizer.pad_token = tokenizer.unk_token
        elif not has_unk and has_pad:
            tokenizer.unk_token = tokenizer.pad_token
        elif not has_pad and not has_unk:
            raise ValueError(
                "Tokenizer must have either pad_token or unk_token defined"
            )

        return tokenizer

    def setup_model_for_lora(config: ModelConfig):
        """Sets up model with quantization and LoRA configs."""
        # TODO: Make it more generic; don't assume quant type and dtype
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Save quantization config
        with open(run_folder / "run_quantization_config.json", "w") as f:
            json.dump(bnb_config.to_dict(), f, indent=2)

        tokenizer = setup_tokenizer(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            use_cache=False,
            device_map="auto",
        )

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.unk_token_id = tokenizer.unk_token_id

        if config.gradient_checkpoint:
            model.gradient_checkpointing_enable()

        if not config.target_modules:
            config.target_modules = MODULES

        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.target_modules,
        )

        # Save LoRA config
        with open(run_folder / "run_lora_config.json", "w") as f:
            lora_dict = peft_config.to_dict()
            # Convert any sets in the config to lists for JSON serialization
            for k, v in lora_dict.items():
                if isinstance(v, set):
                    lora_dict[k] = list(v)
            json.dump(lora_dict, f, indent=2)

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model, tokenizer

    train_dataset, eval_dataset = load_data(
        run_folder=run_folder, use_small_subset=False
    )

    # Save model config to run folder
    model_config.save_to_file(run_folder)

    # Initialize wandb with all configs
    training_config = TrainingConfig(
        num_train_epochs=epoch,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps" if eval_dataset else None,
        eval_steps=100 if eval_dataset else None,
        save_steps=100 if eval_dataset else None,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
    )
    training_config.save_to_file(run_folder)

    wandb.init(
        project=app.name,
        name=run_name,
        tags=tags,
        config={
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "wandb_tags": tags,
        },
    )

    model, tokenizer = setup_model_for_lora(config=model_config)

    # Set up training arguments
    training_args = SFTConfig(
        dataset_text_field="text",
        learning_rate=model_config.learning_rate,
        max_seq_length=model_config.max_seq_length,
        **training_config.to_dict(),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir=run_folder,
        run_name=run_name,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(run_folder)
    tokenizer.save_pretrained(run_folder)

    VOLUME_CONFIG["/runs"].commit()
    wandb.finish()

    return True


@app.local_entrypoint()
def main(
    model: str = MODEL,
    rate: int = 8,
    cot: int = 0,
    rank: int = LORA_RANK,
    alpha: int = LORA_ALPHA,
    batch: int = 4,
    micro_batch: int = 4,
    epoch: int = 1,
):
    assert micro_batch <= batch, "Micro batch size can not be bigger than batch"
    assert (
        batch % micro_batch == 0
    ), "Batch size should be a multiple of micro batch size"
    print("Training model: ", model)

    lr = 2 ** (rate - 20)
    config = ModelConfig(
        model_name=model, lora_rank=rank, lora_alpha=alpha, learning_rate=lr
    )

    wandb_tags = calc_wandb_tags(
        "e2h",
        model,
        rate,
        config.lora_rank,
        config.lora_alpha,
        chat_template=None,
        rslora=False,
        cot=cot,
        batch=batch,
        micro_batch=micro_batch,
        epoch=epoch,
    )
    print("Wandb tags: ", wandb_tags)

    run_name, run_folder = download_data_and_model.remote(
        model_name=model, dataset=E2H_RAW_DATA_14k, cot_percentage=cot
    )
    print("Run Name: ", run_name)
    print("Run Directory: ", run_folder)

    status = train.remote(
        run_name=run_name,
        run_folder=run_folder,
        model_config=config,
        tags=wandb_tags,
        gradient_accumulation_steps=int(batch / micro_batch),
        per_device_train_batch_size=micro_batch,
    )
