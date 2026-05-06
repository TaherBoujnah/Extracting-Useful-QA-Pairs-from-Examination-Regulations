import dspy
import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FAQS = PROJECT_ROOT / "data" / "filtered_faqs.jsonl"
OUTPUT_GOLD = PROJECT_ROOT / "data" / "final_gold_faqs.jsonl"

# 1. Setup the Judge Model (Gemma 3 27B is great at logic evaluation)
print("⚖️ Connecting to llama 3.1 8B (The Judge)...")
llm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='none')
dspy.configure(lm=llm)

# 2. The Judge Signature
class GradeFAQ(dspy.Signature):
    """You are an extremely strict university auditor.
    Your job is to evaluate a generated FAQ based strictly on the provided SOURCE TEXT.
    
    Evaluate the FAQ on a scale of 1 to 5 based on these criteria:
    5: Perfect. The question is highly relevant, and the answer is 100% factually accurate based ONLY on the text.
    4: Good. Accurate, but slightly oddly phrased or misses a tiny nuance.
    3: Mediocre. Partially correct, or includes information not explicitly in the text.
    2: Bad. Misinterprets the rule.
    1: Fatal. Pure hallucination. Invents deadlines, numbers, or rules not in the text.
    
    You must output a brief reasoning, followed by ONLY a single integer for the score.
    """
    source_text = dspy.InputField(desc="The official university rule.")
    question = dspy.InputField(desc="The generated question.")
    answer = dspy.InputField(desc="The generated answer.")
    
    reasoning = dspy.OutputField(desc="A strict, 1-sentence explanation of your evaluation.")
    score = dspy.OutputField(desc="A single integer between 1 and 5.")

def run_judge():
    if not INPUT_FAQS.exists():
        print(f"❌ Input file missing: {INPUT_FAQS}")
        return

    judge = dspy.Predict(GradeFAQ)
    
    total_evaluated = 0
    total_kept = 0
    scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    print("🚀 Starting LLM-as-a-Judge Evaluation...\n")

    with open(INPUT_FAQS, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_GOLD, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip(): continue
            record = json.loads(line)
            source_text = record["source_text"]
            
            validated_faqs = []
            
            for faq in record.get("faqs", []):
                total_evaluated += 1
                q = faq["question"]
                a = faq["answer"]
                
                try:
                    # Ask the LLM to grade the FAQ
                    result = judge(source_text=source_text, question=q, answer=a)
                    
                    # Safely extract the integer score
                    score_str = result.score.strip()
                    # Just grab the first digit if it wrote extra text
                    score_val = int(''.join(filter(str.isdigit, score_str))[0]) 
                    
                    scores[score_val] = scores.get(score_val, 0) + 1
                    
                    # STRICT RULE: Only keep 4s and 5s
                    if score_val >= 4:
                        validated_faqs.append(faq)
                        total_kept += 1
                    else:
                        print(f"   ❌ Dropped (Score {score_val}): {q}")
                        print(f"      Reason: {result.reasoning}")
                        
                except Exception as e:
                    print(f"   ⚠️ Grading failed for a question: {e}")
                    # If the judge crashes, we drop the question to be safe
                    pass 
            
            # Save the record if it still has valid FAQs
            if validated_faqs:
                record["faqs"] = validated_faqs
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("\n" + "="*50)
    print("⚖️ JUDGMENT COMPLETE ⚖️")
    print("="*50)
    print(f"Total Evaluated: {total_evaluated}")
    print(f"Total Kept (Score 4+): {total_kept}")
    print(f"Total Dropped: {total_evaluated - total_kept}")
    print("\nScore Distribution:")
    for s in range(5, 0, -1):
        print(f" ⭐ {s}: {scores.get(s, 0)}")
    print("="*50)

if __name__ == "__main__":
    run_judge()