"""Fixed settings for the study. Changing any of these changes the results."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
HANDCHECK = ROOT / "handcheck"

SEED = 20261019

# SQuAD v1.1 on the Hugging Face hub, pinned to one commit.
SQUAD_REPO = "rajpurkar/squad"
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
SQUAD_FILES = {
    "train": ("plain_text/train-00000-of-00001.parquet",
              "ea7f52bac024f6b1bdc7aaa2a4ee302cba8c2fdc8d4a235cf18a9a5196b6175b"),
    "validation": ("plain_text/validation-00000-of-00001.parquet",
                   "8c6646d36bd5a95061e076788cf3161d11f6f3e7d625dac7a83bbed0a49f69f7"),
}

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CORPUS_SIZE = 10_000
N_BROAD = 500
NEARBY_SAME_PASSAGE = 2
NEARBY_SAME_ARTICLE = 3
KS = (3, 5, 10)
MAIN_K = 5
DOSE_COPIES = 5
# Subsets of the broad set that keep wall time down (all 500 are used where calls are free).
QUOTE_BROAD = 200
REPEAT_BROAD = 150
FORCED_BROAD = 100
# Reduced profile for the slower, larger models: smaller broad set, k=5 only.
REDUCED_BROAD = 100
REDUCED_FORCED = 50

OLLAMA_URL = "http://localhost:11434"
LLM_OPTIONS = {"temperature": 0, "seed": 1234, "num_ctx": 8192, "num_predict": 160}
