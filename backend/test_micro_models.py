import dspy
import json
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CHUNKS = PROJECT_ROOT / "data" / "structured_chunks.jsonl"

class FilterChunk(dspy.Signature):
    """You are a strict data filter for German university regulations. Evaluate the chunk based on these numbered rules:
    
    1. JUNK (False): If the text is merely a list of paragraphs or a Table of Contents (e.g., '- § 1 Name', '- § 2 Ziel'), it is useless.
    2. JUNK (False): If the text is a bureaucratic footer, publication date, or 'Amtliche Bekanntmachungen', it is useless.
    3. KEEP (True): If the text contains concrete academic rules, definitions, or requirements (e.g., workload hours, exam rules, committee members), it is highly useful.
    4. KEEP (True): If the text is a functional table containing credit points or module details, it is highly useful.
    
    Does this text contain a concrete rule or academic fact that a student could actually use?
    """
    context_text = dspy.InputField(desc="The raw German text chunk.")
    is_useful = dspy.OutputField(desc="Respond EXACTLY with 'True' or 'False'.", prefix="Is Useful: ")

class ChunkFilter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(FilterChunk)
        
    def forward(self, text):
        return self.evaluate(context_text=text)

def run_ultimate_battle():
    if not INPUT_CHUNKS.exists():
        print(f"❌ Input file missing: {INPUT_CHUNKS}")
        return

    print("Loading the complete roster of models...")
    
    # Define our contenders 
    model_configs = {
        "Llama 3.1 (8B Baseline)": "ollama_chat/llama3.1",
        "Llama 3.2 (3B)": "ollama_chat/llama3.2",
        "Gemma 2 (2B)": "ollama_chat/gemma2:2b",
        "Qwen 2.5 (1.5B)": "ollama_chat/qwen2.5:1.5b",
        "Llama 3.2 (1B Super Small)": "ollama_chat/llama3.2:1b",
        "Qwen 2.5 (0.5B Underdog)": "ollama_chat/qwen2.5:0.5b"
    }
    
    loaded_models = {}
    scores = {name: 0 for name in model_configs.keys()}

    for name, path in model_configs.items():
        loaded_models[name] = dspy.LM(path, api_base='http://localhost:11434', api_key='none')

    filter_module = ChunkFilter()

    # Read the first 10 chunks
    with open(INPUT_CHUNKS, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:10] 

    # --- THE UPDATED GROUND TRUTH ---
    ground_truth = ["False", "False", "False", "False", "True", "True", "True", "True", "True", "True"]


    for i, line in enumerate(lines):
        text = json.loads(line)['text']
        correct_answer = ground_truth[i]
        
        snippet = text[:50].replace('\n', ' ') + "..."
        print(f"--- Chunk {i+1} | Target: {correct_answer} ---")
        print(f"Text: '{snippet}'")
        
        for model_name, lm_instance in loaded_models.items():
            with dspy.context(lm=lm_instance):
                try:
                    result = filter_module(text=text).is_useful.strip()
                except Exception:
                    result = "CRASHED"
            
            is_correct = correct_answer in result
            if is_correct:
                scores[model_name] += 1
                
            icon = "✅" if is_correct else "❌"
            print(f"  {icon} {model_name.ljust(28)} : {result}")
        print("") 

    # --- FINAL LEADERBOARD ---
    print("\n📊 FINAL LEADERBOARD (Accuracy) 📊")
    print("="*45)
    
    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    for rank, (name, score) in enumerate(sorted_scores, 1):
        accuracy = score * 10
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {name.ljust(28)} : {accuracy}%")
    print("="*45)

if __name__ == "__main__":
    run_ultimate_battle()