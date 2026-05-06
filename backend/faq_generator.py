import dspy
import json
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CHUNKS = PROJECT_ROOT / "data" / "final_hybrid_chunks.jsonl"
GOLD_EXAMPLES = PROJECT_ROOT / "data" / "gold" / "gold.jsonl"
OUTPUT_FAQS = PROJECT_ROOT / "data" / "synthetic_faqs.jsonl"

# 1. Setup Gemma 3 27B
print("🧠 Connecting to Gemma 3 27B...")
gemma3 = dspy.LM('ollama_chat/gemma3:27b', api_base='http://localhost:11434', api_key='none')
dspy.configure(lm=gemma3)

# 2. The Generation Signature (Static Style Guide + Guardrails)
class GenerateStudentFAQs(dspy.Signature):
    """You are a helpful, empathetic academic advisor at a German university.
    Read the provided SOURCE TEXT and the REFERENCE EXAMPLES.
    
    CRITICAL RULES:
    1. IGNORE PREAMBLES: If the SOURCE TEXT is just a title, table of contents, legal preamble, or contains no actionable academic rules, you MUST output the exact word: SKIP
    2. EXACT TERMINOLOGY: Do NOT alter, invent, or generalize the names of degree programs (Studiengänge) oder modules.
    3. NO HALLUCINATION: Generate 2 to 3 realistic questions. Provide clear, accurate answers based ONLY on the provided text, matching the tone of the REFERENCE EXAMPLES.
    
    OUTPUT FORMAT:
    If rules exist, output a strict JSON array: [{"question": "...", "answer": "..."}]
    If NO rules exist, output ONLY the word: SKIP
    """
    reference_examples = dspy.InputField(desc="A list of golden examples showing the exact tone and style you must copy.")
    source_text = dspy.InputField(desc="The text chunk you must evaluate and generate FAQs for.")
    
    json_faqs = dspy.OutputField(desc="A valid JSON array, OR the exact word SKIP.")

def clean_json_string(raw_string):
    """Strips markdown code blocks safely."""
    cleaned = raw_string.strip()
    marker = "`" * 3
    if cleaned.startswith(marker + "json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith(marker):
        cleaned = cleaned[3:]
    if cleaned.endswith(marker):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def load_examples_as_string():
    """Reads your original FAQs and turns them into a single reference string."""
    examples_text = ""
    if GOLD_EXAMPLES.exists():
        with open(GOLD_EXAMPLES, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                # Matches your original {"question": "...", "answer": "..."} format
                examples_text += f"Q: {data['question']}\nA: {data['answer']}\n\n"
        return examples_text.strip()
    else:
        return "No reference examples provided."

def run_generator():
    if not INPUT_CHUNKS.exists():
        print(f"❌ Input file missing: {INPUT_CHUNKS}")
        return

    with open(INPUT_CHUNKS, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]

    print(f"📚 Loaded {len(chunks)} filtered rules.")

    # Load your 10 FAQs as a single block of text
    style_guide = load_examples_as_string()
    print(f"📝 Loaded golden FAQs as a style guide.")

    # Basic predictor - NO BootstrapFewShot compilation needed
    generator = dspy.Predict(GenerateStudentFAQs)

    print("🚀 Starting Guardrailed FAQ generation on FULL dataset...\n")

    with open(OUTPUT_FAQS, 'w', encoding='utf-8') as outfile:
        # Loop through ALL chunks in the file
        for i, chunk in enumerate(chunks):
            current_chunk = chunk['text']
            section_name = chunk['metadata'].get('section', 'General')
            print(f"Processing Rule {i+1}/{len(chunks)}: {section_name}")
            
            try:
                # Pass both the style guide and the current text
                result = generator(
                    reference_examples=style_guide,
                    source_text=current_chunk
                )
                
                raw_json = clean_json_string(result.json_faqs)
                
                # Check for the Skip Token FIRST
                if raw_json.strip().upper() == "SKIP":
                    print("   ⏭️ No actionable rules found. Skipped chunk.")
                    continue
                
                # If it didn't skip, parse the JSON
                faqs = json.loads(raw_json)
                
                final_record = {
                    "metadata": chunk['metadata'],
                    "source_text": current_chunk,
                    "faqs": faqs
                }
                
                outfile.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                print(f"   ✅ Generated {len(faqs)} FAQs successfully.")
                
            except json.JSONDecodeError:
                print("   ⚠️ LLM failed to format JSON properly. Skipping.")
            except Exception as e:
                print(f"   ❌ Error: {e}")

    print("\n" + "="*50)
    print("✨ GENERATION COMPLETE ✨")
    print(f"Check {OUTPUT_FAQS.name} to view your final dataset!")
    print("="*50)

if __name__ == "__main__":
    run_generator()