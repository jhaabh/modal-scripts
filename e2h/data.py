from datasets import load_dataset, DatasetDict, concatenate_datasets
from .constants import (
    E2H_RAW_DATA_14k,
    SYSTEM_PROMPT_W_EXPLANATION,
    SYSTEM_PROMPT_WO_EXPLANATION,
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_process_e2h_data(
    dataset=E2H_RAW_DATA_14k, cot_proportion=0.0, test_split=0.2, random_seed=42
):
    hf_dataset = load_dataset(dataset, split="train")

    processed_data = process_dataset(
        hf_dataset=hf_dataset,
        cot_proportion=cot_proportion,
        test_split=test_split,
        random_seed=random_seed,
    )

    return processed_data


def is_valide_row(row):
    """Validate if a row has all required fields with non-empty values."""
    if not row:
        return False
    if not isinstance(row.get("english"), str) or not row.get("english").strip():
        return False
    if not isinstance(row.get("hindi"), str) or not row.get("hindi").strip():
        return False
    if not isinstance(row.get("reasoning"), str) or not row.get("reasoning").strip():
        return False
    return True


def process_row_with_cot(row):
    try:
        original = row["english"].strip()
        translation = row["hindi"].strip()
        explanation = row["reasoning"].strip()

        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT_W_EXPLANATION,
        }
        assistant_content = (
            f"<thought>\n{explanation}\n</thought>\n\n<hindi>\n{translation}\n</hindi>"
        )

        user_message = {"role": "user", "content": original}
        assistant_message = {"role": "assistant", "content": assistant_content}

        return {"messages": [system_message, user_message, assistant_message]}
    except Exception as e:
        logger.error(f"Error processing row with CoT: {e}")
        return None


def process_row_without_cot(row):
    try:
        original = row["english"].strip()
        translation = row["hindi"].strip()

        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT_WO_EXPLANATION,
        }
        assistant_content = f"<hindi>\n{translation}\n</hindi>"

        user_message = {"role": "user", "content": original}
        assistant_message = {"role": "assistant", "content": assistant_content}

        return {"messages": [system_message, user_message, assistant_message]}
    except Exception as e:
        logger.error(f"Error processing row without CoT: {e}")
        return None


def process_dataset(hf_dataset, cot_proportion=0.0, test_split=0.2, random_seed=42):
    logger.info("Starting dataset processing...")

    hf_dataset = hf_dataset.filter(is_valide_row)

    # Split dataset using HF methods for train-test
    if not test_split:
        train_set = hf_dataset
        test_set = None
    else:
        dataset_split = hf_dataset.train_test_split(
            test_size=test_split, seed=random_seed
        )
        train_set = dataset_split["train"]
        test_set = dataset_split["test"]

    # Further split train set to handle CoT proportion
    if not cot_proportion:
        with_cot_set = None
        wo_cot_set = train_set
    elif cot_proportion < 1:
        cot_split = train_set.train_test_split(
            test_size=cot_proportion, seed=random_seed
        )
        with_cot_set = cot_split["test"]
        wo_cot_set = cot_split["train"]
    else:
        with_cot_set = train_set
        wo_cot_set = None

    logger.info("Processing train set with CoT...")
    processed_train_cot = (
        (
            with_cot_set.map(
                process_row_with_cot,
                remove_columns=hf_dataset.column_names,
                desc="Processing CoT data",
            ).filter(lambda x: x is not None)
        )
        if with_cot_set
        else None
    )

    logger.info("Processing train set without CoT...")
    processed_train_non_cot = (
        (
            wo_cot_set.map(
                process_row_without_cot,
                remove_columns=hf_dataset.column_names,
                desc="Processing non-CoT data",
            ).filter(lambda x: x is not None)
        )
        if wo_cot_set
        else None
    )

    # Combine CoT and non-CoT train splits
    if processed_train_cot and processed_train_non_cot:
        processed_train = concatenate_datasets(
            [processed_train_cot, processed_train_non_cot]
        )
    elif processed_train_cot:
        processed_train = processed_train_cot
    else:
        processed_train = processed_train_non_cot

    logger.info("Processing test set...")
    processed_test = (
        (
            test_set.map(
                process_row_without_cot,
                remove_columns=hf_dataset.column_names,
                desc="Processing test data",
            ).filter(lambda x: x is not None)
        )
        if test_set
        else None
    )

    return DatasetDict(
        {
            key: value
            for key, value in {
                "train": processed_train,
                "test": processed_test,
            }.items()
            if value is not None
        }
    )
