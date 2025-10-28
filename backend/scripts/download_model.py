#!/usr/bin/env python
"""
Standalone script to download and cache the required Sentence Transformer models.

This script should be run once before the first launch of the application to ensure
that the necessary AI models are available locally, preventing startup delays and
network-related errors.
"""
import os
import sys
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)

# Ensure the script can find the 'app' module
# This adds the parent directory ('backend') to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_model(model_name: str) -> bool:
    """
    Downloads a specified Sentence Transformer model to the local cache.

    Args:
        model_name: The name of the model from Hugging Face.

    Returns:
        True if the model was downloaded or already exists, False otherwise.
    """
    logging.info(f"--- Checking for model: {model_name} ---")
    try:
        # This line will download the model if it's not in the cache
        # and do nothing if it is already cached.
        SentenceTransformer(model_name)
        logging.info(f"✅ Model '{model_name}' is available locally.")
        return True
    except Exception as e:
        logging.error(f"🔥 Failed to download or load model '{model_name}'.")
        logging.error(f"   Error: {e}")
        logging.error("   Please check your internet connection and ensure you can connect to huggingface.co.")
        return False

def main():
    """
    Main function to download all required embedding models.
    """
    logging.info("Starting download of required AI embedding models...")
    logging.info("This is a one-time setup and may take a few minutes.")

    try:
        # Import settings from the application
        from app.config import settings
    except ImportError as e:
        logging.error("Failed to import application settings.")
        logging.error(f"Error: {e}")
        logging.error("Please ensure you are running this script from the 'backend' directory.")
        sys.exit(1)

    # List of models to download from the settings
    models_to_download = [
        settings.embedding_gemma_model,
        settings.embedding_model,
    ]

    successful_downloads = 0
    for model in models_to_download:
        if download_model(model):
            successful_downloads += 1

    logging.info("---")
    if successful_downloads == len(models_to_download):
        logging.info("🎉 All required models are downloaded and cached successfully!")
        logging.info("You can now start the backend application.")
        sys.exit(0)
    else:
        logging.error("🔥 Some models failed to download. The application may not function correctly.")
        logging.error("Please resolve the network issues and run this script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
