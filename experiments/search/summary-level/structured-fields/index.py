"""Index summaries with one embedding per field per journal."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared import get_chromadb_client, get_embed_fn, load_summaries

PERSIST_DIR = Path(__file__).parent / ".chromadb"

FIELD_SPECS = {
    "title": lambda s: s["title"],
    "learnings": lambda s: "\n".join(s["key_learnings"]),
    "quotes": lambda s: "\n".join(s["memorable_quotes"]),
    "questions": lambda s: "\n".join(s["active_questions"]),
    "topics": lambda s: ", ".join(s["topics"]),
    "mood": lambda s: ", ".join(s["mood"]),
    "transcript": lambda s: s.get("transcript", ""),
}


def main():
    summaries = load_summaries()
    client = get_chromadb_client(PERSIST_DIR)
    embed_fn = get_embed_fn("RETRIEVAL_DOCUMENT")

    col = client.get_or_create_collection("structured_fields", embedding_function=embed_fn)

    ids, docs, metas = [], [], []
    for i, s in enumerate(summaries):
        for field_name, extractor in FIELD_SPECS.items():
            text = extractor(s)
            if not text.strip():
                continue
            ids.append(f"journal_{i}_{field_name}")
            docs.append(text)
            metas.append({
                "date": s["date"],
                "title": s["title"],
                "field_type": field_name,
            })

    BATCH = 80
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        col.upsert(ids=ids[start:end], documents=docs[start:end], metadatas=metas[start:end])
    print(f"Indexed {len(ids)} documents ({len(summaries)} journals x up to {len(FIELD_SPECS)} fields).")


if __name__ == "__main__":
    main()
