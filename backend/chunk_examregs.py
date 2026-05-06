import json
import re
from pathlib import Path

# Setup paths (Update INPUT_MD if your markdown file is named differently)
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_MD = PROJECT_ROOT / "data" / "cleaned_informatik_regulations.md" # <-- CHECK THIS PATH
OUTPUT_JSONL = PROJECT_ROOT / "data" / "structured_chunks.jsonl"

def process_document():
    if not INPUT_MD.exists():
        print(f"❌ Input file not found: {INPUT_MD}")
        return

    with open(INPUT_MD, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunks = []
    
    # State variables (The "Memory" of the chunker)
    current_program = "Allgemeine Prüfungsordnung"
    current_section = "Allgemeine Bestimmungen"
    current_paragraph_id = "General"
    
    # Buffer to hold broken PDF lines so we can stitch them together
    current_text_buffer = []

    # Regex patterns
    h1_pattern = re.compile(r'^#\s+(.*)')                  # Matches Document Title
    h2_pattern = re.compile(r'^##\s+(.*)')                 # Matches Section Headers
    section_fallback = re.compile(r'^(§\s*\d+.*)')         # Matches "§ 10..." even if missing "##"
    para_pattern = re.compile(r'^(\(\d+\)|\d+\.)\s+(.*)')  # Matches legal markers: "(1) " or "1. "

    def save_chunk():
        if current_text_buffer:
            # 1. THE TEXT STITCHER: Join broken PDF lines with a space
            stitched_text = " ".join([line.strip() for line in current_text_buffer if line.strip()])
            
            # 2. CONTEXT BUNDLING: Inject the headers directly into the text
            bundled_text = f"[{current_program}] > [{current_section}] > {stitched_text}"
            
            chunk = {
                "metadata": {
                    "program": current_program,
                    "section": current_section,
                    "paragraph_id": current_paragraph_id,
                    "chunk_type": "legal_rule"
                },
                "text": bundled_text
            }
            chunks.append(chunk)
            current_text_buffer.clear()

    print("🔪 Slicing document with Deterministic Context Bundling...")

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # Check for H1 (Document / Program Title)
        h1_match = h1_pattern.match(stripped_line)
        if h1_match:
            save_chunk()
            current_program = h1_match.group(1).strip()
            current_section = "General"
            current_paragraph_id = "General"
            continue

        # Check for H2 or a Section (§) marker
        h2_match = h2_pattern.match(stripped_line)
        sec_match = section_fallback.match(stripped_line)
        if h2_match or sec_match:
            save_chunk()
            current_section = h2_match.group(1).strip() if h2_match else sec_match.group(1).strip()
            current_paragraph_id = "General"
            continue

        # Check for legal paragraph markers like (1) or 1.
        para_match = para_pattern.match(stripped_line)
        if para_match:
            save_chunk() # Save the PREVIOUS rule we were building
            current_paragraph_id = para_match.group(1).strip() # Update ID to "(1)"
            current_text_buffer.append(stripped_line) # Start building the NEW rule
            continue

        # If it's none of the above, it's a broken sentence or a bullet point.
        # Append it to the current buffer so it gets stitched!
        current_text_buffer.append(stripped_line)

    # Don't forget to save the very last chunk when the file ends!
    save_chunk()

    # Write the beautifully bundled chunks to JSONL
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f"✅ Successfully bundled {len(chunks)} legal paragraphs!")
    print(f"📁 Saved to: {OUTPUT_JSONL.name}")

if __name__ == "__main__":
    process_document()