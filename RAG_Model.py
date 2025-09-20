from fastapi import FastAPI , UploadFile , File , HTTPException ,Depends,Query
from pydantic import BaseModel
import os 
import google.generativeai as genai
os.environ["USE_TF"] = "0"
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import codecs , uuid 
from contextlib import asynccontextmanager
import numpy as np
import chromadb
import asyncio
# Data model for text input.
class EmbedInput(BaseModel):
    text : str
    language : str | None = None
    note : str   | None = None

model_name = "all-MiniLM-L6-v2"

@asynccontextmanager
async def lifespan(app : FastAPI):
    app.state.model = SentenceTransformer(model_name)
    print(f"loaded model : {model_name}")
    yield
    print(f"Server shutting down and cleaning up.")
    del app.state.model

# create application with life span.
app = FastAPI(lifespan=lifespan)

# function to embedd text.
def embed_text_with_model(text : str):
    model = app.state.model
    return model.encode(text).tolist()    

# First endpoint for plain text.
@app.post("/api/v1/embed/text")
async def Embed_text( input_data : EmbedInput ):
    vector = embed_text_with_model(input_data.text)
    return { 
        "text" : input_data.text ,
        "language" : input_data.language,
        "vector":vector
        }

# Function to split text into chunks.
def split_text(text:str, chunk_size:int = 500 ,overlap :int=50) -> list[str]:
    if chunk_size<= 0:    
       raise ValueError("chunk_size must be >0 ") 
    if overlap <0 or overlap >=chunk_size:
        raise ValueError("chunk must be > 0 and < chunk_size ")
    words = text.split()
    chunks : list[str]=[]
    start = 0

    # Loop to slice text into chunks.
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start+=chunk_size-overlap
    
    return chunks

# Endpoint for file upload (streaming style)
@app.post("/api/v1/embed/file")
async def Embed_file(file: UploadFile = File(...), chunk_size: int = 100, overlap: int = 20):
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = []   # Temporary buffer to accumulate words until chunk_size is reached
    total = 0
    max_bytes = 50 * 1024 * 1024   # Max allowed file size: 50MB

    chunks, vectors, ids, metadatas = [], [], [], []

    # Read file piece by piece (64KB each time)
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break

        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, "The file is too long")

        # Decode to text and split into words
        text_piece = decoder.decode(chunk)
        words = text_piece.split()

        # Add words to buffer and create chunks when enough words are available
        buffer.extend(words)
        while len(buffer) >= chunk_size:
            current_chunk = " ".join(buffer[:chunk_size])
            chunks.append(current_chunk)

            # Create embedding for the current chunk
            vector = embed_text_with_model(current_chunk)
            vectors.append(vector)

            # Assign unique ID and metadata
            ids.append(str(uuid.uuid4()))
            metadatas.append({
                "source": file.filename,
                "chunk_index": len(chunks) - 1,
                "preview": current_chunk[:150],   # Small snippet for quick inspection
                "words_count": len(current_chunk.split())
            })

            # Keep the overlap words in buffer and discard the rest
            buffer = buffer[chunk_size - overlap:]

    # Handle leftover words in buffer (last incomplete chunk)
    if buffer:
        current_chunk = " ".join(buffer)
        chunks.append(current_chunk)

        vector = embed_text_with_model(current_chunk)
        vectors.append(vector)
        ids.append(str(uuid.uuid4()))
        metadatas.append({
            "source": file.filename,
            "chunk_index": len(chunks) - 1,
            "words_count": len(current_chunk.split())
        })

    # Add processed chunks to ChromaDB
    if chunks:
        collection.add(
            documents=chunks,
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas
        )

    return {
        "file name": file.filename,
        "file type": file.content_type,
        "total chunks": len(chunks)
    }

chroma_client = chromadb.PersistentClient(path="chroma_data")

try:
    collection = chroma_client.get_collection(name="hr_collection")
except chromadb.errors.NotFoundError:
    collection = chroma_client.create_collection(
        name="hr_collection",
        metadata={"title": "HR Docs", "description": "Embeddings of HR Policies"}
    )

# create an end point for sematic search.
class SearchInput(BaseModel):
    query:str
    top_k: int = 5  #number of results that return 
@app.post("/api/v1/search")
async def search_collection(input_data : SearchInput):
    query_vector = embed_text_with_model(input_data.query)

    results = collection.query(query_embeddings = [query_vector],n_results = input_data.top_k)
    return{
        "query":input_data.query,
        "results" : results
    }

# call google LLM (gemini).
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# Endpoint for RAG.
class AskInput(BaseModel):
    question : str
    top_k : int=3

@app.post("/api/v1/ask")
async def ask_gemini(input_data : AskInput):
    query_vector = embed_text_with_model(input_data.question)
    search_result = collection.query(query_embeddings =[query_vector],n_results = input_data.top_k)
    
    retrieved_chunks = [doc for doc in search_result["documents"][0]]

    # prompt preparing
    prompt = f""" 
    You are an AI assistant specialized in HR documents.
    Always give clear, professional, and structerd answers.

    If the answer exists is the context , provide it
    If not, say: "The document does not contain enough information."

    Context = {retrieved_chunks}
    Qustions = {input_data.question}
    """
    
    # call Gemini
    response = model.generate_content(prompt)
    return{
        "question":input_data.question,
        "answer":response.text
    }
