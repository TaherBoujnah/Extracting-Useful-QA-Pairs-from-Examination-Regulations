import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FAQS = PROJECT_ROOT / "data" / "synthetic_faqs.jsonl"
OUTPUT_FILTERED = PROJECT_ROOT / "data" / "filtered_faqs.jsonl"

# Configuration
SIMILARITY_THRESHOLD = 0.92  
MODEL_NAME = 'intfloat/multilingual-e5-large' 

def filter_dataset():
    if not INPUT_FAQS.exists():
        print(f"❌ Input file missing: {INPUT_FAQS}")
        return

    print("Loading data...")
    records = []
    with open(INPUT_FAQS, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # --- TIER 1: HEURISTIC FILTERING ---
    print("🧹 Running Tier 1: Heuristic Filtering...")
    seen_exact_questions = set()
    tier1_passed_faqs = []  
    
    stats = {"original": 0, "no_q_mark": 0, "too_short": 0, "exact_duplicate": 0, "semantic_duplicate": 0}

    for record_idx, record in enumerate(records):
        for faq in record.get("faqs", []):
            stats["original"] += 1
            q = faq.get("question", "").strip()
            a = faq.get("answer", "").strip()

            # Must be a question
            if not q.endswith("?"):
                stats["no_q_mark"] += 1
                continue
            
            # Substantial answer
            if len(a.split()) < 5:
                stats["too_short"] += 1
                continue
                
            # Exact String Deduplication
            if q.lower() in seen_exact_questions:
                stats["exact_duplicate"] += 1
                continue
                
            seen_exact_questions.add(q.lower())
            
            # Save mapping so we can rebuild the JSONL later
            tier1_passed_faqs.append({
                "record_idx": record_idx,
                "question": q,
                "answer": a
            })

    print(f"   Passed Tier 1: {len(tier1_passed_faqs)} / {stats['original']} FAQs")

    # --- TIER 2: SEMANTIC DEDUPLICATION ---
    print(f"\n🧠 Running Tier 2: Semantic Deduplication...")
    print(f"   Downloading/Loading Heavyweight Model: {MODEL_NAME}")
    print(f"   (This may take a minute on the first run as it downloads ~2.2GB)...")
    
    # Determine device (Use GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # Prefix required by the e5 models for optimal performance
    questions = [f"query: {item['question']}" for item in tier1_passed_faqs]
    
    print("   Computing embeddings for all questions...")
    embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True)
    
    print("   Calculating cosine similarity matrix...")
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    # Find Semantic Duplicates
    indices_to_drop = set()
    for i in range(len(tier1_passed_faqs)):
        if i in indices_to_drop:
            continue
        for j in range(i + 1, len(tier1_passed_faqs)):
            if j in indices_to_drop:
                continue
            # If the similarity is higher than our threshold, mark the second one for deletion
            if cosine_scores[i][j].item() > SIMILARITY_THRESHOLD:
                indices_to_drop.add(j)
                stats["semantic_duplicate"] += 1

    # --- RECONSTRUCT DATASET ---
    print("\n💾 Reconstructing and saving final dataset...")
    
    # Create empty shells for the final records
    final_records = [{"metadata": r["metadata"], "source_text": r["source_text"], "faqs": []} for r in records]
    
    total_kept = 0
    for i, item in enumerate(tier1_passed_faqs):
        if i not in indices_to_drop:
            final_records[item["record_idx"]]["faqs"].append({
                "question": item["question"],
                "answer": item["answer"]
            })
            total_kept += 1

    # Write output
    with open(OUTPUT_FILTERED, 'w', encoding='utf-8') as outfile:
        for record in final_records:
            # Only write chunks that still have FAQs attached to them
            if record["faqs"]:
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    # --- REPORT ---
    print("\n" + "="*50)
    print("✨ FILTERING COMPLETE ✨")
    print("="*50)
    print(f"Total Generated:         {stats['original']}")
    print(f"Total Kept (Gold Data):  {total_kept}")
    print(f"Total Dropped:           {stats['original'] - total_kept}")
    print("\nBreakdown of Drops:")
    print(f" ❌ Missing '?':           {stats['no_q_mark']}")
    print(f" ❌ Answer too short:      {stats['too_short']}")
    print(f" ❌ Exact Duplicates:      {stats['exact_duplicate']}")
    print(f" 🧠 Semantic Duplicates:   {stats['semantic_duplicate']} (Similarity > {SIMILARITY_THRESHOLD})")
    print("="*50)

if __name__ == "__main__":
    filter_dataset()