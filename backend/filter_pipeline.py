import dspy
import json
from pathlib import Path
from dspy.teleprompt import BootstrapFewShot

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CHUNKS = PROJECT_ROOT / "data" / "structured_chunks.jsonl"
OUTPUT_CHUNKS = PROJECT_ROOT / "data" / "final_hybrid_chunks.jsonl"

# 1. Setup the Model
gemma_micro = dspy.LM('ollama_chat/gemma2:2b', api_base='http://localhost:11434', api_key='none')
dspy.configure(lm=gemma_micro)

# 2. Semantic Signature
class FilterChunk(dspy.Signature):
    """You are evaluating German university regulations. 
    The obvious formatting junk has already been removed. Your job is to check for semantic usefulness.
    
    KEEP (True): The text contains concrete academic rules, exam requirements, credit points, or module definitions.
    JUNK (False): The text is just a vague introduction, a bureaucratic preamble (e.g., "Aufgrund des Gesetzes..."), or legal boilerplate that a student doesn't need to know.
    
    Does this paragraph contain a concrete, useful academic rule?
    """
    context_text = dspy.InputField(desc="The German text chunk.")
    is_useful = dspy.OutputField(desc="Respond EXACTLY with 'True' or 'False'.")

class ChunkFilter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(FilterChunk)
        
    def forward(self, context_text):
        return self.evaluate(context_text=context_text)

def run_hybrid_filter():
    if not INPUT_CHUNKS.exists():
        print(f"Input file missing: {INPUT_CHUNKS}")
        return

    # Read all chunks
    with open(INPUT_CHUNKS, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 3. Training Data
    print("Bootstrapping AI for semantic evaluation...")
    training_data = [
        {"text": "[General] > Aufgrund des § 2 Abs. 4 und des § 64 Abs. 1 des Gesetzes über die Hochschulen des Landes Westfalen...", "target": "False"},
        {"text": "[§ 2 Studium: Qualifikationsziele] > (1) Der Bachelorstudiengang soll den Studierenden eine fundierte wissenschaftliche Grundausbildung in ihrem Fach vermitteln.", "target": "False"},
        {"text": "[§ 8 Bachelorprüfung: Regeln] > (4) Prüfungsleistungen im Sinne dieser Prüfungsordnung werden durch benotete Prüfungen erbracht und begründen die Modulnote.", "target": "True"},
        {"text": "[§ 14 Modulprüfungen: Wiederholung] > (3) Eine nicht bestandene Modulprüfung kann zweimal wiederholt werden.", "target": "True"}
    ]

    trainset = [dspy.Example(context_text=item["text"], is_useful=item["target"]).with_inputs('context_text') for item in training_data]
    
    teleprompter = BootstrapFewShot(metric=None, max_bootstrapped_demos=2, max_labeled_demos=4)
    compiled_filter = teleprompter.compile(ChunkFilter(), trainset=trainset)
    print("AI compilation complete.\n")

    kept_chunks = 0
    python_dropped = 0
    ai_dropped = 0

    print("Running hybrid cascade filter...\n")

    with open(OUTPUT_CHUNKS, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(lines, start=1):
            chunk = json.loads(line)
            text = chunk['text']
            section = chunk['metadata'].get('section', '')
            
            # STAGE 1: PYTHON FILTER
            if text.strip().endswith("> ---") or len(text) < 50:
                python_dropped += 1
                continue
            if "Inhaltsübersicht" in section or "Inhaltsübersicht" in text:
                python_dropped += 1
                continue
            if "Artikel I" in section or "Artikel II" in section or "Übergangsbestimmungen" in section:
                python_dropped += 1
                continue
                
            # Scrub trailing footers
            text = text.replace("---- **HHU Amtliche Bekanntmachungen Nr. 53/2021** ---", "").strip()
            text = text.replace("HHU Amtliche Bekanntmachungen Nr. 53/2021 ---", "").strip()
            chunk['text'] = text
            
            # STAGE 2: AI SEMANTIC FILTER
            try:
                result = compiled_filter(context_text=text)
                decision = result.is_useful.strip().lower()
                
                if "true" in decision:
                    outfile.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                    kept_chunks += 1
                else:
                    ai_dropped += 1
            except Exception:
                ai_dropped += 1
                
            if i % 10 == 0 or i == len(lines):
                print(f"Processed {i}/{len(lines)}... (Python Dropped: {python_dropped} | AI Dropped: {ai_dropped} | Kept: {kept_chunks})")

    print("\n" + "="*50)
    print("HYBRID FILTERING COMPLETE")
    print(f"Junk caught by Python: {python_dropped}")
    print(f"Fluff caught by AI: {ai_dropped}")
    print(f"Valid rules kept: {kept_chunks}")
    print(f"Saved to: {OUTPUT_CHUNKS.name}")
    print("="*50)

if __name__ == "__main__":
    run_hybrid_filter()