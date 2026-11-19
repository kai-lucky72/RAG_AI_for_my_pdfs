import utils.session
import os
import time
import requests
import pdfplumber
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === CONFIGURATION ===
PDF_FOLDER = "/home/lamelo/Downloads/tesingFolders"
PINECONE_API_KEY = "pcsk_eUSKR_FS38c6tLMTE23nn3eGJNFGdRj9MUGydH9Zzy5h68fRfXHfsaj2yzqXq9P4EH3FG"
INDEX_NAME = "document-chunks"
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

# === TEXT CHUNKING ===
def extract_text(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"⚠️ Failed to process {file_path}: {e}")
        return None

def load_and_chunk_pdfs(folder_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            text = extract_text(path)
            if not text:
                continue
            for i, chunk in enumerate(splitter.split_text(text)):
                all_chunks.append({
                    "id": f"{filename}-{i}",
                    "text": chunk,
                    "metadata": {"source": filename, "text": chunk}
                })
    return all_chunks

# === EMBEDDING ===
def embed_batch(texts, model):
    inputs = [f"passage: {t}" for t in texts]
    return model.encode(inputs, normalize_embeddings=True).tolist()

# === PINECONE SETUP ===
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

# === INDEXING ===
def upload_chunks_to_pinecone(chunks, embedder, index):
    BATCH_SIZE = 50
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="🔼 Uploading"):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        vectors = embed_batch(texts, embedder)

        pinecone_vectors = []
        for item, vector in zip(batch, vectors):
            pinecone_vectors.append({
                "id": item["id"],
                "values": vector,
                "metadata": item["metadata"]
            })

        try:
            index.upsert(vectors=pinecone_vectors)
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            break
        time.sleep(1)

# === RAG QUERY USING OPENROUTER ===
def rag_query(query_text, index, embedder):
    query_embedding = embed_batch([query_text], embedder)[0]
    results = index.query(
        vector=query_embedding,
        top_k=TOP_K_RESULTS,
        include_metadata=True
    )

    relevant_chunks = [match['metadata']['text'] for match in results['matches']]
    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are an AI assistant. Use the context below to answer the user's question accurately.

Context:
{context}

Question: {query_text}
Answer:"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers based on provided documents."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("❌ Error:", response.text)
        return "Error: Failed to get response from OpenRouter."

# === MAIN ===
if __name__ == "__main__":
    print("🚀 Starting PDF to Pinecone pipeline...")

    # Step 1: Load and chunk PDFs
    chunks = load_and_chunk_pdfs(PDF_FOLDER)
    print(f"✅ Extracted {len(chunks)} chunks.")

    # Step 2: Load embedding model
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Loaded embedding model.")

    # Step 3: Init Pinecone and upload
    pinecone_index = init_pinecone()
    upload_chunks_to_pinecone(chunks, embedder, pinecone_index)

    # Step 4: Ask your question
    while True:
        user_query = input("\n❓ Ask a question about your documents (or type 'exit'): ")
        if user_query.lower() == "exit":
            break
        response = rag_query(user_query, pinecone_index, embedder)
        print(f"\n🤖 Answer: {response}\n")
