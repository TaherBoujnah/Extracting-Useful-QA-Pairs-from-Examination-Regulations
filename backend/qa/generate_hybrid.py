# backend/qa/generate_hybrid.py
from __future__ import annotations

import argparse
import json
import random
import time
import dataclasses
from pathlib import Path
from typing import Any, Dict, List

from backend.qa.ollama_client import ollama_generate
from backend.qa.generate_common import Chunk, load_chunks_from_jsonl, write_json, write_jsonl, jaccard
from backend.retrieval.bundle import bundle_chunk_ids  # <-- IMPORTED BUNDLE.PY


def qa_prompt(rule_text: str, degree_level: str, program: str) -> str:
    return f"""
Du bist ein Assistent, der aus Prüfungsordnungen FAQ-Frage-Antwort-Paare erzeugt.
Kontext:
- Studienniveau: {degree_level}
- Studiengang/Programm: {program}

Zusammenhängender Regel-Auszug (inklusive Kontext und Abhängigkeiten):
\"\"\"{rule_text}\"\"\"

Erzeuge genau EIN hilfreiches FAQ-Paar (Frage + Antwort), das vollständig aus dem obigen Auszug ableitbar ist.
- Keine Spekulation, keine neuen Fakten.
- Antwort kurz und präzise (1–4 Sätze).
- Frage soll realistisch sein (Studierendenperspektive).
- Kein Verweis auf "Auszug" oder "oben".

Gib NUR gültiges JSON zurück, exakt mit diesen Schlüsseln:
{{"question": "...", "answer": "..."}}
""".strip()


def parse_one_json(raw: str) -> Dict[str, str] | None:
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


def generate_hybrid(
    model: str,
    chunks_path: Path,
    out_path: Path,
    seed_limit: int = 10,
    qas_per_seed: int = 6,
    max_context_words: int = 2500, 
    temperature: float = 0.7,      
    num_predict: int = 240,
    dup_question_jaccard_threshold: float = 0.85,
) -> Dict[str, Any]:
    chunks = load_chunks_from_jsonl(chunks_path)
    if not chunks:
        raise RuntimeError(f"No chunks loaded from {chunks_path}")

    # Build the dictionary map required by bundle.py
    chunks_map = {}
    for c in chunks:
        if dataclasses.is_dataclass(c):
            chunks_map[c.chunk_id] = dataclasses.asdict(c)
        else:
            chunks_map[c.chunk_id] = c.__dict__.copy()

    rng = random.Random(1337)
    seeds = rng.sample(chunks, k=min(seed_limit, len(chunks)))

    out_rows: List[Dict[str, Any]] = []
    seen_questions: List[str] = []

    t0 = time.time()

    for seed in seeds:
        if len(out_rows) >= seed_limit * qas_per_seed:
            break

        # 1. BUNDLE THE CHUNKS
        bundled_dicts = bundle_chunk_ids(
            seed_chunk_ids=[seed.chunk_id],
            chunks_map=chunks_map,
            add_neighbors=True,
            add_deps=True
        )

        # 2. GLUE TEXT TOGETHER (Chronological Mega-Prompt)
        bundle_text = "\n\n---\n\n".join([d.get("text", "") for d in bundled_dicts])
        rule_words = bundle_text.split()
        rule_text = " ".join(rule_words[:max_context_words])

        added_for_seed = 0
        # 3. ASK THE LLM MULTIPLE TIMES USING THE SAME BUNDLE
        # We loop more times than needed (qas_per_seed * 3) because some will be duplicates/fail
        for attempt in range(qas_per_seed * 3):
            if added_for_seed >= qas_per_seed:
                break
            if len(out_rows) >= seed_limit * qas_per_seed:
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
        "strategy": "true_bundle",
        "runtime_seconds": runtime,
        "written_qas": len(out_rows),
        "settings": {
            "seed_limit": seed_limit,
            "qas_per_seed": qas_per_seed,
            "max_context_words": max_context_words,
        },
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
    out_path = Path(args.out) if args.out else Path(f"data/generated/faqs_hybrid__{args.model.replace(':', '_')}.jsonl")

    generate_hybrid(
        model=args.model, 
        chunks_path=chunks_path, 
        out_path=out_path,
        seed_limit=args.seed_limit,
    )

if __name__ == "__main__":
    main()