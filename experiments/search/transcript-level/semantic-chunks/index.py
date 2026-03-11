"""Chunk transcripts with chonkie SemanticChunker, then index chunks in ChromaDB."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from chonkie import SemanticChunker

from shared import get_chromadb_client, get_embed_fn, load_summaries

PERSIST_DIR = Path(__file__).parent / ".chromadb"


def main():
    summaries = load_summaries()
    client = get_chromadb_client(PERSIST_DIR)
    embed_fn = get_embed_fn("RETRIEVAL_DOCUMENT")

    col = client.get_or_create_collection("semantic_chunks", embedding_function=embed_fn)

    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-32M",
        threshold=0.5,
        chunk_size=512,
    )

    ids, docs, metas = [], [], []
    for i, s in enumerate(summaries):
        transcript = s.get("transcript", "").strip()
        if not transcript:
            continue

        chunks = chunker.chunk(transcript)
        for ci, chunk in enumerate(chunks):
            ids.append(f"journal_{i}_chunk_{ci}")
            docs.append(chunk.text)
            metas.append({
                "date": s["date"],
                "title": s["title"],
                "chunk_index": ci,
            })

    BATCH = 80
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        col.upsert(ids=ids[start:end], documents=docs[start:end], metadatas=metas[start:end])
    print(f"Indexed {len(ids)} chunks from {len(summaries)} journals.")


if __name__ == "__main__":
    main()
