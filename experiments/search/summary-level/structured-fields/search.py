"""Interactive semantic search over structured-field summary embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared import get_chromadb_client, get_embed_fn

PERSIST_DIR = Path(__file__).parent / ".chromadb"


def _compact(text: str, max_chars: int = 220) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 3] + "..."


def main():
    client = get_chromadb_client(PERSIST_DIR)
    embed_fn = get_embed_fn("RETRIEVAL_QUERY")
    col = client.get_collection("structured_fields", embedding_function=embed_fn)

    print("Structured-fields summary search (type 'q' to quit)")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == "q":
            break

        results = col.query(query_texts=[query], n_results=10)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        if not docs:
            print("No results.")
            continue

        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            embedding_for = (meta or {}).get("embedding_for", "structured")
            print(f"{rank:>2}. dist={dist:.4f} | for={embedding_for} | {_compact(doc)}")


if __name__ == "__main__":
    main()
