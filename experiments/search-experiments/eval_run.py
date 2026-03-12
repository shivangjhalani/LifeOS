"""Run test queries against all search experiments and dump raw results to JSON."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SEARCH_DIR = Path(__file__).resolve().parent

EXPERIMENTS = {
    "full_blob": (SEARCH_DIR / "summary-level" / "full-blob", "search.py"),
    "structured_fields": (SEARCH_DIR / "summary-level" / "structured-fields", "search.py"),
    "semantic_chunks": (SEARCH_DIR / "transcript-level" / "semantic-chunks", "search.py"),
    "enriched_chunks": (SEARCH_DIR / "transcript-level" / "enriched-semantic-chunks", "search.py"),
    "grouped_structured": (SEARCH_DIR / "summary-level" / "structured-fields-grouped", "search.py"),
    "grouped_semantic_chunks": (SEARCH_DIR / "transcript-level" / "semantic-chunks", "search_grouped.py"),
    "grouped_enriched_chunks": (SEARCH_DIR / "transcript-level" / "enriched-semantic-chunks", "search_grouped.py"),
}

# ── Test Queries with Ground Truth ──────────────────────────────────────
# Each entry: (query, category, expected_journal_indices)
# Indices match sorted file order (0-27).
# Expected = journals that SHOULD appear in top-5.
QUERIES = [
    # EXACT MATCH: specific content that lives in one or two journals
    ("Dr. K video about communication and empathy", "exact",
     [25]),  # Communicating = QOthoughts
    ("Elon Musk brain chip power level", "exact",
     [0]),   # 43-min journal mentions Elon Musk
    ("Thorium AI internship interview", "exact",
     [18]),  # 2 intern opportunities
    ("AI for Good startup part time", "exact",
     [18]),  # same
    ("not able to sleep figuring things out", "exact",
     [10, 24]),  # both "Not able to sleeep" entries

    # THEMATIC: broad themes across multiple journals
    ("motivation to build my own app", "thematic",
     [0, 2, 7, 15, 17, 19]),  # app dev + commitment journals
    ("meaning of life and helping others", "thematic",
     [4, 22]),  # both meaning-of-life entries
    ("procrastination and turning a bad day around", "thematic",
     [6, 15, 19, 23]),  # bad-day-turnaround + procrastination
    ("career anxiety and job rejection", "thematic",
     [5, 18, 21, 27]),  # career/job related
    ("craft versus outcome satisfaction", "thematic",
     [8, 12]),  # both craft-vs-outcome entries

    # QUESTION-STYLE: how a real user would query their journal
    ("when did I feel most determined to build something?", "question",
     [2, 17]),  # I WILL MAKE THIS FKIN APPP entries
    ("what is my approach to problem solving?", "question",
     [11]),  # My strength of solving problems from fundamentals
    ("how can I improve my communication skills?", "question",
     [0, 3, 9, 13, 25]),  # communication-related journals

    # EMOTIONAL: mood-based retrieval
    ("feeling rejected and unwanted", "emotional",
     [21]),  # I need companies, they don't need me
    ("anxious and restless at night", "emotional",
     [10, 24]),  # Not able to sleep entries
    ("feeling excited about a new project idea", "emotional",
     [0, 2, 15, 17, 19]),  # app idea excitement

    # OBSCURE/INDIRECT: non-obvious connections
    ("walking as a mood booster and thinking tool", "indirect",
     [3, 6, 9, 15, 19, 23]),  # walking mentioned across many
    ("voice as a tool for self-expression", "indirect",
     [0, 3, 9, 25]),  # voice journaling concept
    ("inner drive versus discipline and habits", "indirect",
     [13, 26]),  # output + cultivate inner drive
    ("dating apps and self-identity confusion", "indirect",
     [20]),  # the idea of being like someone
]


def run_search(experiment_dir: Path, query: str, script: str = "search.py") -> str:
    """Run a search and return raw stdout."""
    result = subprocess.run(
        ["uv", "run", script, query],
        cwd=str(experiment_dir),
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout + result.stderr


def parse_results(raw: str) -> list[dict]:
    """Parse search output into structured results."""
    results = []
    lines = raw.strip().split("\n")
    current = {}
    content_lines = []
    in_content = False
    for line in lines:
        if line.startswith("[") and "] dist=" in line:
            if current:
                current["content"] = "\n".join(content_lines).strip()
                results.append(current)
                content_lines = []
                in_content = False
            rank_str = line.split("]")[0].strip("[")
            dist_str = line.split("dist=")[1]
            current = {"rank": int(rank_str), "distance": float(dist_str)}
        elif line.startswith("title: ") and not in_content:
            current["title"] = line[7:].strip()
        elif line.startswith("for: "):
            current["embedding_for"] = line[5:]
        elif line.startswith("content:"):
            in_content = True
            rest = line[8:].strip()
            if rest:
                content_lines.append(rest)
        elif in_content and current:
            content_lines.append(line)
    if current:
        current["content"] = "\n".join(content_lines).strip()
        results.append(current)
    return results


def main():
    all_results = {}
    total = len(QUERIES) * len(EXPERIMENTS)
    done = 0

    for q_idx, (query, category, expected) in enumerate(QUERIES):
        q_key = f"q{q_idx:02d}"
        all_results[q_key] = {
            "query": query,
            "category": category,
            "expected_indices": expected,
            "experiments": {},
        }
        for exp_name, (exp_dir, exp_script) in EXPERIMENTS.items():
            print(f"[{done+1}/{total}] {exp_name}: {query[:50]}...", flush=True)
            raw = run_search(exp_dir, query, exp_script)
            parsed = parse_results(raw)
            all_results[q_key]["experiments"][exp_name] = {
                "raw": raw[:2000],  # truncate for size
                "parsed": parsed,
            }
            done += 1
            time.sleep(0.5)  # rate limit buffer

    out_path = SEARCH_DIR / "eval_raw_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone. Results saved to {out_path}")


if __name__ == "__main__":
    main()
