# test-unsloth-inference.py

import os
import modal

from images import UNSLOTH_IMAGE

# Configuration
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
HUGGINGFACE_SECRET_NAME = "huggingface-secret"
VOLUME_NAME = "unsloth-model-cache"
MODELS_DIR = "/models"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = UNSLOTH_IMAGE

app = modal.App(name="test-unsloth-inference", image=image)


@app.function(
    gpu="T4",
    volumes={MODELS_DIR: volume},
    secrets=[modal.Secret.from_name(HUGGINGFACE_SECRET_NAME)],
    timeout=1800,
)
def test_and_infer(test_type: str, prompt: str = None):
    import atexit
    import signal
    import threading
    import torch

    def cleanup():
        """Cleanup CUDA cache and threads"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Attempt to clean up non-daemon threads
        for thread in threading.enumerate():
            if thread != threading.current_thread() and not thread.daemon:
                try:
                    thread.join(timeout=0.5)
                except TimeoutError:
                    pass

    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda signo, frame: cleanup())

    try:
        """Combined test and inference function."""
        results = {}

        if test_type in ["setup", "all"]:
            try:
                from unsloth import FastLanguageModel

                # Test CUDA
                if torch.cuda.is_available():
                    device = torch.cuda.get_device_name(0)
                    print(f"✓ CUDA available: {device}")
                    results["setup"] = {"status": "success", "device": device}
                else:
                    raise RuntimeError("CUDA not available")

            except Exception as e:
                print(f"✗ Setup failed: {e}")
                results["setup"] = {"status": "failed", "error": str(e)}
                return results

        if test_type in ["inference", "all"] and prompt:
            try:
                response = generate_with_unsloth(MODEL_NAME, prompt)
                results["inference"] = {"status": "success", "response": response}

            except Exception as e:
                print(f"✗ Inference failed: {e}")
                results["inference"] = {"status": "failed", "error": str(e)}
    finally:
        cleanup()

    return results


def generate_with_unsloth(model_name, prompt):
    """
    Perform inference using Unsloth's FastLanguageModel with proper chat formatting.

    Args:
        model_name (str): The name of the pre-trained model to use.
        prompt (str): The user prompt to generate a response for.

    Returns:
        str: The generated response from the model.
    """
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print(f"Loading model: {model_name}...")

    # Load the model with Unsloth's FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        token=os.getenv(
            "HF_TOKEN"
        ),  # Ensure your HuggingFace token is set in the environment
        load_in_4bit=True,  # Utilize 4-bit quantization for memory efficiency
    )

    # Apply the appropriate chat template
    # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    chat_template = "llama-3.1"  # Specify the correct template for the model
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    print("\nPreparing for inference...")
    FastLanguageModel.for_inference(model)  # Enable Unsloth's optimized inference mode

    # Format the conversation with Unsloth's chat template
    messages = [{"role": "user", "content": prompt}]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to("cuda")

    attention_mask = inputs["attention_mask"]
    input_ids = inputs["input_ids"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )

    # Decode and return the response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nResponse: {response}")
    return response


@app.local_entrypoint()
def main(
    test: str = "all",
    prompt: str = "Explain the significance of the Fibonacci sequence.",
):
    """
    Run tests and inference.
    Args:
        test: Type of test ("setup", "inference", "all")
        prompt: Text prompt for inference
    """
    if test not in ["setup", "inference", "all"]:
        raise ValueError("test must be one of: setup, inference, all")

    results = test_and_infer.remote(test, prompt)

    # Pretty print results
    print("\nTest Results:")
    for test_name, result in results.items():
        print(f"\n{test_name.upper()}:")
        if result["status"] == "success":
            print("✓ Success")
            if test_name == "setup":
                print(f"  Device: {result['device']}")
            elif test_name == "inference":
                print(f"  Response: {result['response']}")
        else:
            print(f"✗ Failed: {result['error']}")
