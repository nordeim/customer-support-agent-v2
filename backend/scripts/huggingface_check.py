from huggingface_hub import hf_api
api = hf_api.HfApi()
print(api.model_info("google/embeddinggemma-300m").private)  # will raise if unauthorized

