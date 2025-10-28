from huggingface_hub import hf_hub_url, HfApi, RepositoryNotFoundError, hf_hub_download
import requests
import time

def model_accessible(model_name: str, token: str | None) -> bool:
    api = HfApi()
    try:
        api.model_info(model_name, token=token)
        return True
    except Exception as e:
        logging.error(f"Model access check failed for {model_name}: {e}")
        return False

def download_model(model_name: str) -> bool:
    logging.info(f"--- Checking for model: {model_name} ---")
    # preflight
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not model_accessible(model_name, token):
        logging.error(f"🔥 Access denied or model not found on Hugging Face: {model_name}")
        logging.error("   Make sure your account has accepted any gated model terms or set a valid HUGGINGFACE_HUB_TOKEN.")
        return False

    # robust loading with retry for transient errors
    for attempt in range(6):
        try:
            SentenceTransformer(model_name)
            logging.info(f"✅ Model '{model_name}' is available locally.")
            return True
        except Exception as e:
            if attempt < 5:
                sleep = 2 ** attempt
                logging.warning(f"Transient error loading {model_name}: {e}. Retrying in {sleep}s.")
                time.sleep(sleep)
                continue
            logging.error(f"🔥 Failed to download or load model '{model_name}'. Error: {e}")
            return False

