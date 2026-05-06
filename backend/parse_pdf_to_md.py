import time
import requests
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "exam_regulations.pdf"
OUTPUT_MD_PATH = DATA_DIR / "parsed_regulations.md"

# Put your actual LlamaCloud API Key here
API_KEY = "your_llamacloud_api_key_here"
BASE_URL = "https://api.cloud.llamaindex.ai/api/parsing"

def parse_via_rest():
    print(f"🚀 Uploading {PDF_PATH.name} to LlamaParse API...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }
    
    # 1. Upload the file to LlamaCloud
    with open(PDF_PATH, "rb") as f:
        files = {"file": (PDF_PATH.name, f, "application/pdf")}
        data = {
            "language": "de", 
            "parsing_instruction": "This is a German university examination regulation document. Extract all text, headers, and tables cleanly as markdown."
        }
        
        upload_res = requests.post(f"{BASE_URL}/upload", headers=headers, files=files, data=data)
        upload_res.raise_for_status()
        job_id = upload_res.json()["id"]
        
    print(f"✅ Upload successful! Job ID: {job_id}")
    print("⏳ Waiting for parsing to complete (this will take 1-3 minutes for 119 pages)...")
    
    # 2. Poll the server every 10 seconds until it's done
    while True:
        time.sleep(10)
        status_res = requests.get(f"{BASE_URL}/job/{job_id}", headers=headers)
        status_res.raise_for_status()
        status = status_res.json()["status"]
        
        if status == "SUCCESS":
            print("\n✅ Parsing complete on the server!")
            break
        elif status == "ERROR":
            print("\n❌ Parsing failed on the server.")
            return
        else:
            print(".", end="", flush=True) # Print a dot to show it's still thinking
            
    # 3. Download the Markdown result
    print(f"📥 Downloading markdown...")
    result_res = requests.get(f"{BASE_URL}/job/{job_id}/result/markdown", headers=headers)
    result_res.raise_for_status()
    
    markdown_content = result_res.json()["markdown"]
    
    # 4. Save to file
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"🎉 Success! Beautiful markdown saved to {OUTPUT_MD_PATH.name}")

if __name__ == "__main__":
    parse_via_rest()