import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import ollama
from pathlib import Path

# --- Configuration ---
# Fixed the path to look one folder higher so it finds the 'data' folder properly!
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
LLM_MODEL = "gemma3:12b"

# Looser thresholds to allow for typos, slang, and varied sentence structures
FAQ_THRESHOLD = 1.2    
CHUNK_THRESHOLD = 1.5  

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="UniBot: Academic Advisor", page_icon="🎓", layout="centered")
st.title("🎓 UniBot: RAG Academic Advisor")

# --- 2. CONNECT TO DATABASE ---
@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    faq_col = client.get_collection(name="faqs", embedding_function=ef)
    chunk_col = client.get_collection(name="raw_chunks", embedding_function=ef)
    return faq_col, chunk_col

faq_collection, chunk_collection = load_db()

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("z.B. Ich habe gepennt und die Prüfung verpasst, was nun?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        # --- TIER 1: Search FAQs (Fetching Top 3) ---
        faq_results = faq_collection.query(query_texts=[prompt], n_results=3)
        faq_distance = faq_results['distances'][0][0] if faq_results['distances'][0] else 999

        retrieved_context = ""
        source_used = ""

        if faq_distance <= FAQ_THRESHOLD:
            # Bundle the top 3 results: The raw text + the synthetic answer
            for i in range(len(faq_results['documents'][0])):
                meta = faq_results['metadatas'][0][i]
                retrieved_context += f"Regel-Auszug {i+1}: {meta['source_text']}\nKontext/Erklärung: {meta['answer']}\n\n"
            
            source_used = f"✅ (FAQ Match | Beste Distanz: {faq_distance:.2f})"
            
        else:
            # --- TIER 2: Search Raw Chunks (Fetching Top 3) ---
            chunk_results = chunk_collection.query(query_texts=[prompt], n_results=3)
            chunk_distance = chunk_results['distances'][0][0] if chunk_results['distances'][0] else 999
            
            if chunk_distance <= CHUNK_THRESHOLD:
                for i in range(len(chunk_results['documents'][0])):
                    retrieved_context += f"Text-Auszug {i+1}: {chunk_results['documents'][0][i]}\n\n"
                
                source_used = f"⚠️ (Raw Chunk Match | Beste Distanz: {chunk_distance:.2f})"
                
            else:
                # --- TIER 3: Fallback ---
                fallback_msg = """
                Es tut mir leid, aber meine Suche in der Prüfungsordnung hat dazu keinen passenden Paragraphen gefunden. 
                
                Bitte wende dich für diese spezifische Frage direkt an das Prüfungsamt:
                📧 **E-Mail:** spv-informatik@hhu.de
                """
                st.markdown(fallback_msg)
                st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
                st.stop()

        # --- GENERATE STREAMING RESPONSE ---
        sys_prompt = f"""Du bist ein hilfsbereiter, empathischer Studienberater der Uni Düsseldorf.
        Beantworte die Frage des Studenten basierend auf den folgenden Auszügen aus der Prüfungsordnung.
        Kombiniere die Informationen sinnvoll. Wenn eine spezifische Zahl (wie erlaubte Fehlversuche) im Text fehlt, erkläre dem Studenten, was laut dem Text generell passiert, weise aber darauf hin, dass du die genaue Zahl für sein Fachgebiet nicht kennst.
        Sei niemals roboterhaft.
        
        GEFUNDENE TEXTE:
        {retrieved_context}
        """

        def generate_response():
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                stream=True
            )
            for chunk in response:
                yield chunk['message']['content']

        final_answer = st.write_stream(generate_response)
        st.caption(f"Search Route: {source_used}")
        
        st.session_state.messages.append({"role": "assistant", "content": final_answer})