"""Search full-blob summary embeddings. Usage: uv run search.py "query" """

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared import get_chromadb_client, get_embed_fn

PERSIST_DIR = Path(__file__).parent / ".chromadb"


def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: uv run search.py \"your query here\"")
        sys.exit(1)

    client = get_chromadb_client(PERSIST_DIR)
    embed_fn = get_embed_fn("RETRIEVAL_QUERY")
    col = client.get_collection("full_blob", embedding_function=embed_fn)

    results = col.query(query_texts=[query], n_results=5)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    if not docs:
        print("No results.")
        return

    print(f"Top {len(docs)} results for: {query}\n")
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        embedding_for = (meta or {}).get("embedding_for", "full_blob")
        print(f"[{rank:02d}] dist={dist:.4f}")
        print(f"for: {embedding_for}")
        print(f"content:\n{doc}\n")


if __name__ == "__main__":
    main()
