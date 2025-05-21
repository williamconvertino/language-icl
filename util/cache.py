import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "../cache")

def setup_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    os.environ["HF_HOME"] = f"{CACHE_DIR}"
    os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"
    os.environ["TMPDIR"] = f"{CACHE_DIR}"
    
setup_cache_dir()