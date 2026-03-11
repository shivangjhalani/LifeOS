import json
from pathlib import Path

import chromadb
import chromadb.utils.embedding_functions as ef
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARIES_DIR = PROJECT_ROOT / "private" / "summaries"

load_dotenv(PROJECT_ROOT / ".env")


def load_summaries() -> list[dict]:
    files = sorted(SUMMARIES_DIR.glob("*_summary.json"))
    summaries = []
    for f in files:
        with open(f) as fh:
            summaries.append(json.load(fh))
    return summaries


def get_chromadb_client(persist_dir: str | Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(persist_dir))


def get_embed_fn(task_type: str = "RETRIEVAL_DOCUMENT") -> ef.GoogleGeminiEmbeddingFunction:
    return ef.GoogleGeminiEmbeddingFunction(
        model_name="gemini-embedding-001",
        task_type=task_type,
    )
