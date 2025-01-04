from pathlib import Path
from typing import Optional
import modal

from images import UNSLOTH_IMAGE

from .constants import (
    HF_DIR,
    RUNS_DIR,
    SYSTEM_PROMPT_W_EXPLANATION,
    SYSTEM_PROMPT_WO_EXPLANATION,
)

MODEL = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
RUN = "e2h-1735975994"

RUN_FOLDER_BASE = Path("/runs/")

hf_volume = modal.Volume.from_name("e2h-hf-vol", create_if_missing=True)
runs_volume = modal.Volume.from_name("e2h-runs-3-vol", create_if_missing=True)

VOLUME_CONFIG = {
    HF_DIR: hf_volume,
    RUNS_DIR: runs_volume,
}

app = modal.App(name="e2h-train-test")

image = UNSLOTH_IMAGE


@app.function(
    gpu=modal.gpu.L4(),
    image=image,
    volumes=VOLUME_CONFIG,
    timeout=600,
)
def generate(
    model: str,
    peft_path: str,
    input: str,
    cot: bool,
    max_token: int,
    min_p: float,
    temp: float,
):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_path,
        device_map="auto",
        # torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_path)

    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT_W_EXPLANATION if cot else SYSTEM_PROMPT_WO_EXPLANATION
            ),
        },
        {"role": "user", "content": input},
    ]
    tokenized_chat = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    tokenized_chat = tokenized_chat.to(model.device)  # Move input to model's device
    outputs = model.generate(tokenized_chat, max_new_tokens=max_token)

    return tokenizer.decode(outputs[0])


@app.local_entrypoint()
def main(
    model: str = MODEL,
    run: str = RUN,
    cp: Optional[str] = None,
    input: str = "Translate to Hindi: How are you?",
    cot: bool = False,
    max_tok: int = 512,
    min_p: float = 0.1,
    temp: float = 0.1,
):
    base_path = f"{RUNS_DIR}/{run}"
    peft_path = f"{base_path}/checkpoint-{cp}" if cp else base_path
    print("Loading weights from: ", peft_path)

    response = generate.remote(
        model=model,
        peft_path=peft_path,
        input=input,
        cot=cot,
        max_token=max_tok,
        min_p=min_p,
        temp=temp,
    )
    print("Response/>\n", response)
