from .constants import ModelTag, MODEL_TO_TAG_DATA


def calc_wandb_tags(
    project_tag,
    model_name,
    rate: int,
    lora_rank: int = None,
    lora_alpha: int = None,
    chat_template: str = None,
    rslora: bool = None,
    cot: int = None,
    batch: int = None,
    micro_batch: int = None,
    epoch: int = None,
):
    model_tag = MODEL_TO_TAG_DATA[model_name]
    wandb_tags = [
        project_tag,
        model_name,  # Existing tag
        f"family:{model_tag.family}",
        f"sub_family:{model_tag.sub_family or 'None'}",
        f"size:{model_tag.size}B",
        f"type:{model_tag.type}",
        f"rate:{rate}",
        f"rank: {lora_rank}",
        f"alpha: {lora_alpha}",
        f"rslora: {rslora}",
    ]
    if chat_template:
        wandb_tags.append(f"chat_template: {chat_template}")
    if cot:
        wandb_tags.append(f"cot: {cot}")
    if batch:
        wandb_tags.append(f"batch: {batch}")
    if micro_batch:
        wandb_tags.append(f"micro_batch: {micro_batch}")
    if epoch:
        wandb_tags.append(f"epoch: {epoch}")

    return wandb_tags


def calc_suffix(
    train_suffix,
    model_name,
    lora_rank,
    lora_alpha,
    cot_proportion,
    rate,
    chat_template: str,
    rslora: bool,
):
    model_tag_data = MODEL_TO_TAG_DATA.get(
        model_name,
        ModelTag(tag="unk", family=None, sub_family=None, size=None, type=None),
    )
    model_tag = model_tag_data.tag
    lora_tag = f"r{lora_rank}a{lora_alpha}"
    lora_tag = f"{lora_tag}-rslora" if rslora else lora_tag
    cot_proportion = fraction_to_percentage(cot_proportion)
    tag_suffix = f"{model_tag}-{lora_tag}-{cot_proportion}-r{rate}"
    train_suffix = f"{train_suffix}-{tag_suffix}" if train_suffix else tag_suffix
    train_suffix = f"{train_suffix}-{chat_template}" if chat_template else train_suffix
    return train_suffix


def fraction_to_percentage(fraction):
    # Convert fraction to percentage and round to the nearest integer
    percentage = round(fraction * 100)
    # Append 'p' to the result
    return f"{percentage}p"
