import re
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_MD = PROJECT_ROOT / "data" / "parsed_regulations.md"
OUTPUT_MD = PROJECT_ROOT / "data" / "cleaned_informatik_regulations.md"

# Format: "Section Name": (Start Page, End Page)
TARGET_SECTIONS = {
    "General Bachelor Rules": (10, 27),
    "Bachelor Informatik": (54, 56),
    "General Master Rules": (78, 94),
    "Master Informatik": (111, 112),
    "Master AI & Data Science": (113, 115) # Verify these numbers in the PDF!
}

def remove_irrelevant_paragraphs(text):
    """
    Slices text into paragraphs and removes any paragraph that talks ONLY 
    about other degrees (like Biology or Chemistry) without mentioning ours.
    """
    paragraphs = text.split('\n\n')
    kept_paragraphs = []
    
    # Trigger words for other degrees
    other_degrees = ["biochemie", "biologie", "chemie", "medizinische physik", "industrial pharmacy", "quantitative biology"]
    
    for p in paragraphs:
        p_lower = p.lower()
        has_other_degree = any(deg in p_lower for deg in other_degrees)
        has_our_degree = "informatik" in p_lower or "artificial intelligence" in p_lower or "data science" in p_lower or "mathematik" in p_lower
        
        # If the paragraph mentions other degrees but NOT ours, we drop it to prevent hallucinations
        if has_other_degree and not has_our_degree:
            continue
            
        kept_paragraphs.append(p)
        
    return '\n\n'.join(kept_paragraphs)

def validate_extraction(file_path):
    """
    The Validation Layer: Checks the final markdown to ensure it is healthy.
    """
    print("\n" + "="*50)
    print("🔍 RUNNING VALIDATION LAYER...")
    print("="*50)
    
    if not file_path.exists():
        print("❌ FAILED: The output file was not created.")
        return False
        
    text = file_path.read_text(encoding="utf-8")
    
    # 1. Check Document Length (Should be at least a few thousand characters)
    if len(text) < 5000:
        print(f"❌ FAILED: File is suspiciously short ({len(text)} characters).")
        return False
    else:
        print(f"✅ Pass: Document length is healthy ({len(text)} characters).")

    # 2. Check for Required Sections
    missing_sections = [sec for sec in TARGET_SECTIONS.keys() if f"# {sec}" not in text]
    if missing_sections:
        print(f"❌ FAILED: Missing these mandatory headers: {missing_sections}")
        return False
    else:
        print(f"✅ Pass: All {len(TARGET_SECTIONS)} target sections are present.")

    # 3. Hallucination Check (Did any forbidden words slip through?)
    forbidden_words = ["Biochemie", "Industrial Pharmacy", "Quantitative Biology"]
    slipped_words = [word for word in forbidden_words if word.lower() in text.lower()]
    
    if slipped_words:
        print(f"⚠️ WARNING: Found mentions of irrelevant degrees: {slipped_words}. The LLM might hallucinate.")
    else:
        print(f"✅ Pass: Zero mentions of forbidden degrees. Hallucination risk minimized.")

    print("\n🎉 VALIDATION SUCCESSFUL! The file is ready for the RAG pipeline.")
    return True


def clean_by_pages():
    if not INPUT_MD.exists():
        print(f"❌ File not found: {INPUT_MD}")
        return

    text = INPUT_MD.read_text(encoding="utf-8")
    
    # Slice the document exactly at the footers (e.g., "Seite 54 von 119")
    segments = re.split(r'(?i)Seite\s+(\d+)\s+von\s+\d+', text)
    pages_text = {}
    
    for i in range(0, len(segments) - 1, 2):
        text_block = segments[i].strip()
        page_num = int(segments[i+1])
        pages_text[page_num] = text_block
        if i == len(segments) - 3:
            pages_text[page_num] += "\n" + segments[-1].strip()

    combined_text = ""
    for section_title, (start_page, end_page) in TARGET_SECTIONS.items():
        combined_text += f"# {section_title}\n\n"
        for p in range(start_page, end_page + 1):
            if p in pages_text:
                combined_text += pages_text[p] + "\n\n"
        combined_text += "---\n\n"

    # Step A: Scrub out the massive comma-separated lists of degrees in §1
    combined_text = re.sub(
        r"(?:Biochemie|Biologie|Chemie)[\s\S]{10,250}?(?:Physik|Naturwissenschaften)“?", 
        "Informatik und Artificial Intelligence and Data Science", 
        combined_text, flags=re.IGNORECASE
    )

    # Step B: Remove paragraphs that exclusively talk about other degrees
    combined_text = remove_irrelevant_paragraphs(combined_text)

    # Save the file
    OUTPUT_MD.write_text(combined_text, encoding="utf-8")
    print(f"\n📁 File saved to: {OUTPUT_MD.name}")
    
    # Run the Validation Layer
    validate_extraction(OUTPUT_MD)

if __name__ == "__main__":
    clean_by_pages()