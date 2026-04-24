# backend/qa/generate_slow_hq.py
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from backend.qa.ollama_client import ollama_generate
from backend.qa.generate_common import load_chunks_from_jsonl, write_json, write_jsonl, jaccard


def qa_prompt(rule_text: str, degree_level: str, program: str) -> str:
    return f"""
Du bist ein Assistent, der aus Prüfungsordnungen sehr hochwertige FAQ-Frage-Antwort-Paare erzeugt.

Kontext:
- Studienniveau: {degree_level}
- Studiengang/Programm: {program}

Einzelner Regel-/Textauszug:
\"\"\"{rule_text}\"\"\"

Aufgabe:
1) Formuliere eine konkrete, nützliche Frage (Studierendenperspektive).
2) Beantworte sie ausschließlich anhand des Auszugs (keine Spekulation).
3) Antwort 1–4 Sätze, klar und präzise.

Gib NUR gültiges JSON zurück:
{{"question": "...", "answer": "..."}}
""".strip()


def parse_one_json(raw: str) -> Dict[str, str] | None:
    # (Same JSON parser as before)
    raw = (raw or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except Exception:
        return None
    q = (obj.get("question") or "").strip()
    a = (obj.get("answer") or "").strip()
    if not q or not a:
        return None
    return {"question": q, "answer": a}


def generate_slow_hq(
    model: str,
    chunks_path: Path,
    out_path: Path,
    seed_limit: int = 10,
    max_context_words: int = 1400,
    max_qas_per_seed: int = 6,
    temperature: float = 0.05, 
    num_predict: int = 260,
    dup_question_jaccard_threshold: float = 0.85,
) -> Dict[str, Any]:
    chunks = load_chunks_from_jsonl(chunks_path)
    if not chunks:
        raise RuntimeError(f"No chunks loaded from {chunks_path}")

    rng = random.Random(1337)
    seeds = rng.sample(chunks, k=min(seed_limit, len(chunks)))

    out_rows: List[Dict[str, Any]] = []
    seen_questions: List[str] = []

    t0 = time.time()

    for si, seed in enumerate(seeds, start=1):
        
        rule_text = " ".join(seed.text.split()[:max_context_words])

        added_for_seed = 0
        
        
        for attempt in range(max_qas_per_seed * 3):
            if added_for_seed >= max_qas_per_seed:
                break

            raw = ollama_generate(
                model=model,
                prompt=qa_prompt(rule_text, seed.degree_level, seed.program),
                temperature=temperature,
                num_predict=num_predict,
            )
            parsed = parse_one_json(raw)
            if not parsed:
                continue

            q = parsed["question"]
            if any(jaccard(q, prev) >= dup_question_jaccard_threshold for prev in seen_questions):
                continue

            seen_questions.append(q)
            out_rows.append(
                {
                    "question": parsed["question"],
                    "answer": parsed["answer"],
                    "degree_level": seed.degree_level,
                    "program": seed.program,
                    "source_chunk_id": seed.chunk_id,
                }
            )
            added_for_seed += 1

    runtime = round(time.time() - t0, 2)

    write_jsonl(out_path, out_rows)
    meta = {
        "model": model,
        "strategy": "single_chunk_baseline",
        "runtime_seconds": runtime,
        "written_qas": len(out_rows),
    }
    write_json(out_path.with_suffix(out_path.suffix + ".meta.json"), meta)
    return meta

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed_limit", type=int, default=10) 
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    out_path = Path(args.out) if args.out else Path(f"data/generated/faqs_slow_hq__{args.model.replace(':', '_')}.jsonl")

    generate_slow_hq(
        model=args.model, 
        chunks_path=chunks_path, 
        out_path=out_path,
        seed_limit=args.seed_limit
    )

if __name__ == "__main__":
    main()