#!/usr/bin/env python3
"""Summarize transcript JSON files using Gemini API."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_SUMMARIES = PROJECT_ROOT / "private" / "summaries"

SUMMARY_PROMPT_TEMPLATE = """You are analyzing a personal voice journal transcript. This is a {duration:.1f}-minute reflective session where the person thinks out loud about their life, career, and personal growth.

Your task is to create a summary that helps the person:
1. Remember what they were feeling and thinking
2. Recall specific learnings and insights they want to internalize
3. Revisit this period of their life without listening to the full audio

TRANSCRIPT:
{text}

---

Generate a structured summary with the following sections:

## EMOTIONAL LANDSCAPE
Describe the emotional tone and mental state during this session. What was the person feeling? Were they anxious, hopeful, confused, energized, reflective? Capture the emotional texture, not just facts.

## KEY LEARNINGS & INSIGHTS
Extract the most important insights, realizations, or lessons the person articulated. These are things they explicitly or implicitly want to remember and internalize. Format as bullet points, each starting with the core insight.

## TOPICS & THEMES EXPLORED
List the main subjects discussed (e.g., career decisions, relationships, technical topics, philosophical questions). For each topic, write 1-2 sentences about what was said.

## MEMORABLE QUOTES
Pull 3-5 direct quotes from the transcript that best capture the person's thinking, voice, or emotional state. Choose quotes that are specific, personal, and meaningful - not generic statements.

## ASPIRATIONS & GOALS MENTIONED
What does the person want to achieve, become, or work toward? Include both explicit goals and implicit desires/directions they're considering.

## QUESTIONS & TENSIONS
What questions were they wrestling with? What tensions or contradictions were they exploring? What seemed unresolved?

## PATTERNS & RECURRING THEMES
If this topic/concern/question has appeared in previous journals, note that. If this seems like a new direction in their thinking, note that too.

---

IMPORTANT GUIDELINES:
- Use the person's own language and phrasing where possible
- Be specific, not generic. "Worried about AI replacing junior developers" not "concerned about career"
- Capture nuance. If they were both excited AND anxious, say that
- Don't editorialize or add wisdom that wasn't in the transcript
- If they mentioned specific people, companies, technologies, or events, include those details"""


def load_transcript(path: Path) -> dict:
    """Load and validate transcript JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "text" not in data:
        raise ValueError(f"Transcript missing 'text' field: {path}")
    return data


def summarize_with_gemini(transcript: dict, api_key: str) -> str:
    """Call Gemini API to generate summary."""
    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        language=transcript.get("language", "Unknown"),
        duration=float(transcript.get("duration", 0)) / 60,
        source_name=transcript.get("source", "Unknown"),
        text=transcript["text"].strip(),
    )

    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a transcript JSON file using Gemini API."
    )
    parser.add_argument(
        "transcript",
        type=Path,
        help="Path to transcript JSON (e.g. private/transcripts/2025-03-14T10:41:33Z_Outcome vs craft.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for summary (default: private/summaries/<basename>.md)",
    )
    args = parser.parse_args()

    transcript_path = args.transcript
    if not transcript_path.is_absolute():
        transcript_path = (PROJECT_ROOT / transcript_path).resolve()

    if not transcript_path.exists():
        parser.error(f"Transcript not found: {transcript_path}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        parser.error(
            "GEMINI_API_KEY not set. Add it to .env (copy from .example.env)."
        )

    transcript = load_transcript(transcript_path)
    transcript["source"] = transcript_path.stem

    print("Generating summary...")
    summary = summarize_with_gemini(transcript, api_key)

    PRIVATE_SUMMARIES.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()
    else:
        out_path = PRIVATE_SUMMARIES / f"{transcript_path.stem}_summary.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8")
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
