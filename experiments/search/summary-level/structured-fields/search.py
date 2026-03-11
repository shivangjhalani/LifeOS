"""Interactive semantic search over structured-field summary embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared import get_chromadb_client, get_embed_fn

PERSIST_DIR = Path(__file__).parent / ".chromadb"


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
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            print(f"\n--- #{i+1} (dist={dist:.4f}) [{meta['field_type']}] ---")
            print(f"  Title: {meta['title']}")
            print(f"  Date:  {meta['date']}")
            print(f"  Text:  {doc[:200]}...")


if __name__ == "__main__":
    main()
