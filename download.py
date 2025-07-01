import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model():
    """
    Downloads the Qwen3-0.6B model from Hugging Face to the specified directory.
    """
    repo_id = "Qwen/Qwen3-0.6B"
    # The bench.py script expects this specific path.
    local_dir = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    print(f"Downloading model '{repo_id}' to '{local_dir}'...")

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
    )

    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)

    print("Download complete.")
    print(f"Model files are saved in: {local_dir}")

if __name__ == "__main__":
    download_model()