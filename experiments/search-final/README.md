# Search Final

## Layer 1: Summary Retrieval

- Goal: find the right journal entry.
- Indexing idea: split each summary into a few semantically tight field-level documents, instead of embedding one big blob.
- One embedded document/vector:
  - `title + topics + key_learnings`
  - `title + topics + active_questions`
  - `title + topics + memorable_quotes`
  - `title + topics + mood`
- Retrieval idea: search over these field docs, then group hits by journal and keep the best match per journal.

Why this is the best layer-1 choice:

- It was the overall best method in evaluation.
- It keeps field-level precision.
- Grouping removes slot bias from multiple fields of the same journal competing with each other.
- It gives better journal discovery than transcript chunks.

## Layer 2: Transcript Retrieval

- Goal: find the exact moment/passage inside a journal.
- Indexing idea: split the transcript into semantic chunks, then prepend light summary context.
- One embedded document/vector:
  - `[title | topics] + one semantic transcript chunk`
- Chunking method:
  - semantic chunking, not fixed-size slicing
  - chunk boundary follows meaning shifts in the transcript

Why this is the best layer-2 choice:

- It was the best transcript-level method in evaluation.
- It beats raw transcript chunks because `title/topics` helps disambiguate the journal.
- It is better for passage retrieval than summary-level methods, because it returns the actual relevant excerpt.

## Why Two Layers

- Layer 1 answers: "which journal?"
- Layer 2 answers: "where exactly inside that journal?"
- This is the strongest practical split from the evaluation:
  - `Grouped Structured` for journal discovery
  - `Enriched Chunks` for passage retrieval

## Hybrid Variants

- `layer-1-summary-hybrid`: dense + BM25 over structured summary docs
- `layer-2-transcript-hybrid`: dense + BM25 over enriched transcript chunks
- Hybrid score:
  - `0.6 * dense + 0.4 * bm25`
