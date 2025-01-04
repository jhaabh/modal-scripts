from typing import Literal, NamedTuple, Optional

HUGGINGFACE_SECRET_NAME = "huggingface-secret"
WANDB_SECRET_NAME = "wandb-secret"

HF_DIR = "/models"
RUNS_DIR = "/runs"

E2H_RAW_DATA_14k = "jhaabhi/eng_2_hindi_w_cot_14k"

SYSTEM_PROMPT_W_EXPLANATION = "Translate given English text into Hindi. Carefully think step by step before coming up with appropriate translation."
SYSTEM_PROMPT_WO_EXPLANATION = "Translate given English text into Hindi."

MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class ModelTag(NamedTuple):
    family: str
    sub_family: Optional[str]
    size: float  # In Billion params
    type: Literal["instruct", "base"]
    tag: str


MODEL_TO_TAG_DATA = {
    "Qwen/Qwen2.5-0.5B-Instruct": ModelTag("qwen", "2.5", 0.5, "instruct", "q0.5b-nu"),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelTag("qwen", "2.5", 1.5, "instruct", "q1.5b-nu"),
    "Qwen/Qwen2.5-3B-Instruct": ModelTag("qwen", "2.5", 3, "instruct", "q3b-nu"),
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit": ModelTag(
        "llama", "3.2", 1, "instruct", "l1b3.2"
    ),
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit": ModelTag(
        "llama", "3.2", 3, "instruct", "l3b3.2"
    ),
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit": ModelTag(
        "llama", "3.3", 70, "instruct", "l70b3.3"
    ),
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": ModelTag(
        "llama", "3.1", 8, "instruct", "l8b3.1"
    ),
    "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit": ModelTag(
        "qwen", "2.5", 0.5, "instruct", "q0.5b2.5"
    ),
    "unsloth/Qwen2.5-0.5B-Instruct": ModelTag(
        "qwen", "2.5", 0.5, "instruct", "q0.5b2.5-nuu"
    ),
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit": ModelTag(
        "qwen", "2.5", 1.5, "instruct", "q1.5b2.5"
    ),
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit": ModelTag(
        "qwen", "2.5", 3, "instruct", "q3b2.5"
    ),
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit": ModelTag(
        "qwen", "2.5", 7, "instruct", "q7b2.5"
    ),
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit": ModelTag(
        "qwen", "2.5", 14, "instruct", "q14b2.5"
    ),
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit": ModelTag(
        "qwen", "2.5", 32, "instruct", "q32b2.5"
    ),
    "unsloth/gemma-2-9b-it-bnb-4bit": ModelTag("gemma", "2", 9, "instruct", "g9b2"),
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit": ModelTag(
        "mistral", "nemo", 12, "instruct", "m12bnemo"
    ),
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit": ModelTag(
        "mistral", "0.3", 7, "instruct", "m7b0.3"
    ),
    "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit": ModelTag(
        "smollm", None, 1.7, "instruct", "s1.7b"
    ),
    "unsloth/gemma-2-2b-bnb-4bit": ModelTag("gemma", "2", 2, "instruct", "g2b2"),
    "unsloth/Llama-3.2-1B-bnb-4bit": ModelTag("llama", "3.2", 1, "base", "l1b3.2base"),
    "unsloth/Llama-3.2-3B-bnb-4bit": ModelTag("llama", "3.2", 3, "base", "l3b3.2base"),
    "unsloth/gemma-2-2b-bnb-4bit": ModelTag("gemma", "2", 2, "base", "g2b2base"),
    "unsloth/Qwen2.5-0.5B-bnb-4bit": ModelTag(
        "qwen", "2.5", 0.5, "instruct", "q0.5b2.5base"
    ),
    "unsloth/Qwen2.5-1.5B-bnb-4bit": ModelTag(
        "qwen", "2.5", 1.5, "instruct", "q1.5b2.5base"
    ),
    "unsloth/Qwen2.5-3B-bnb-4bit": ModelTag("qwen", "2.5", 3, "instruct", "q3b2.5base"),
}
