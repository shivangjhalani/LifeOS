"""Index summaries as single text blobs using all available fields."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared import get_chromadb_client, get_embed_fn, load_summaries

PERSIST_DIR = Path(__file__).parent / ".chromadb"


def build_text(s: dict) -> str:
    parts = [
        f"date: {s.get('date', '')}",
        f"duration_minutes: {s.get('duration_minutes', '')}",
        f"title: {s.get('title', '')}",
        f"mood: {', '.join(s.get('mood', []))}",
        f"topics: {', '.join(s.get('topics', []))}",
        f"memorable_quotes: {' | '.join(s.get('memorable_quotes', []))}",
        f"key_learnings: {' | '.join(s.get('key_learnings', []))}",
        f"active_questions: {' | '.join(s.get('active_questions', []))}",
        f"transcript: {s.get('transcript', '')}",
    ]
    return "\n".join(parts)


def main():
    summaries = load_summaries()
    client = get_chromadb_client(PERSIST_DIR)
    embed_fn = get_embed_fn("RETRIEVAL_DOCUMENT")

    col = client.get_or_create_collection("full_blob", embedding_function=embed_fn)

    ids, docs, metas = [], [], []
    for i, s in enumerate(summaries):
        ids.append(f"journal_{i}")
        docs.append(build_text(s))
        metas.append({"date": s["date"], "title": s["title"]})

    BATCH = 80
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        col.upsert(ids=ids[start:end], documents=docs[start:end], metadatas=metas[start:end])
    print(f"Indexed {len(ids)} journals into full_blob collection.")


if __name__ == "__main__":
    main()
