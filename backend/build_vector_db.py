import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DATA = PROJECT_ROOT / "data" / "final_gold_faqs.jsonl"
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

def build_database():
    if not INPUT_DATA.exists():
        print(f"❌ Cannot find input data at {INPUT_DATA}")
        return

    print("🚀 Initializing ChromaDB Multi-Tier Indexer...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Reset/Create TWO collections
    for name in ["faqs", "raw_chunks"]:
        try:
            client.delete_collection(name=name)
        except Exception:
            pass 
            
    faq_collection = client.create_collection(name="faqs", embedding_function=sentence_transformer_ef)
    chunk_collection = client.create_collection(name="raw_chunks", embedding_function=sentence_transformer_ef)

    faq_docs, faq_metas, faq_ids = [], [], []
    chunk_docs, chunk_metas, chunk_ids = [], [], []
    seen_chunks = set() # To avoid duplicating chunks

    faq_counter = 0
    chunk_counter = 0

    print("📖 Reading Gold Dataset...")
    with open(INPUT_DATA, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            source_text = record["source_text"]
            section = record.get("metadata", {}).get("section", "Unknown")
            
            # 1. Store the Raw Chunk (if we haven't seen it yet)
            if source_text not in seen_chunks:
                chunk_docs.append(source_text)
                chunk_metas.append({"section": section})
                chunk_ids.append(f"chunk_{chunk_counter}")
                seen_chunks.add(source_text)
                chunk_counter += 1
            
            # 2. Store the FAQs
            for faq in record.get("faqs", []):
                faq_docs.append(faq["question"])
                faq_metas.append({
                    "answer": faq["answer"], 
                    "source_text": source_text, 
                    "section": section
                })
                faq_ids.append(f"faq_{faq_counter}")
                faq_counter += 1

    print(f"📦 Inserting {len(faq_docs)} FAQs into Tier 1...")
    faq_collection.add(documents=faq_docs, metadatas=faq_metas, ids=faq_ids)
    
    print(f"📦 Inserting {len(chunk_docs)} Raw Legal Chunks into Tier 2...")
    chunk_collection.add(documents=chunk_docs, metadatas=chunk_metas, ids=chunk_ids)

    print("✨ DUAL VECTOR DATABASE BUILT SUCCESSFULLY ✨")

if __name__ == "__main__":
    build_database()