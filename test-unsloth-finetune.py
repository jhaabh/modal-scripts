# test-unsloth-finetune.py
import os
import modal

# Reuse the same image from your images.py
from images import UNSLOTH_IMAGE

# Configuration
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
HUGGINGFACE_SECRET_NAME = "huggingface-secret"
WANDB_SECRET_NAME = "wandb-secret"
VOLUME_NAME = "unsloth-model-cache"
MODELS_DIR = "/models"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = UNSLOTH_IMAGE

app = modal.App(name="test-unsloth-finetune", image=image)


@app.function(
    gpu="T4",  # or "A10G", "L40S", etc.
    volumes={MODELS_DIR: volume},
    secrets=[
        modal.Secret.from_name(HUGGINGFACE_SECRET_NAME),
        modal.Secret.from_name(WANDB_SECRET_NAME),
    ],
    timeout=3600,
)
def train_and_save(
    max_steps: int = 50,
    lr: float = 2e-4,
    max_seq_len: int = 1024,
    use_small_subset: bool = True,
    wandb_project: str = "unsloth-finetune",  # WandB project name
):
    """
    Finetune a model using Unsloth and track with WandB.

    Args:
      max_steps: Number of steps to train for.
      lr: Learning rate.
      max_seq_len: Max sequence length for your model/training.
      use_small_subset: If True, use smaller portion of dataset for a quick demonstration.
      wandb_project: Name of the WandB project for logging.
    """
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from trl import SFTTrainer

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import (
        get_chat_template,
        train_on_responses_only,
        standardize_sharegpt,
    )
    import wandb

    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=f"finetune-{MODEL_NAME}",
        config={
            "model": MODEL_NAME,
            "learning_rate": lr,
            "max_seq_length": max_seq_len,
            "max_steps": max_steps,
            "batch_size": 2,  # per_device_train_batch_size
        },
    )

    # 1) Load your dataset
    dataset_name = "mlabonne/FineTome-100k"
    subset_str = "train[:1%]" if use_small_subset else "train"
    print(f"Loading dataset = {dataset_name}, subset = {subset_str}")
    dataset = load_dataset(dataset_name, split=subset_str)
    dataset = standardize_sharegpt(dataset)

    print(f"Dataset size: {len(dataset)}")

    # 2) Load base model with Unsloth
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        max_seq_length=max_seq_len,
        token=os.getenv("HF_TOKEN"),  # optional, if model is gated
    )

    # 3) Prepare LoRA PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # 4) (Optional) Chat template
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Format dataset
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 5) Create a DataCollator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    print("BF16 supported: ", is_bfloat16_supported())

    # 6) Setup training arguments
    training_args = TrainingArguments(
        output_dir=f"{MODELS_DIR}/finetune-lora-output",
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_steps=5,
        logging_steps=10,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="wandb",
        optim="adamw_8bit",
        save_total_limit=1,
        save_steps=20,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        data_collator=data_collator,
        args=training_args,
    )

    # Optional: Train only on assistant's responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # 7) Run the training
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # 8) Save final LoRA adapter
    save_dir = f"{MODELS_DIR}/finetune-lora-model"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Finetuned LoRA model saved to: {save_dir}")

    # 9) Commit changes to the volume so the files are persisted
    volume.commit()

    # Finish WandB run
    wandb.finish()


@app.local_entrypoint()
def main():
    """
    Local entrypoint to run finetuning. Adjust arguments as needed.
    """
    train_and_save.remote(
        max_steps=50, lr=2e-4, max_seq_len=1024, use_small_subset=True
    )
