from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
import json
from pathlib import Path

app = FastAPI(title="Chatbot RAG com Embeddings")

# ======================
# MODELO DE EMBEDDING
# ======================

model = SentenceTransformer("all-MiniLM-L6-v2")

# ======================
# BANCO DE DADOS
# ======================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "embeddings.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS documentos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    texto TEXT,
    embedding TEXT
)
""")
conn.commit()

# ======================
# MODELOS
# ======================

class Documento(BaseModel):
    texto: str

class Pergunta(BaseModel):
    pergunta: str

# ======================
# UTILIDADES
# ======================

def salvar_documento(texto: str):
    emb = model.encode(texto).tolist()
    cursor.execute(
        "INSERT INTO documentos (texto, embedding) VALUES (?, ?)",
        (texto, json.dumps(emb))
    )
    conn.commit()

def buscar_similares(pergunta: str, top_k=2):
    q_emb = model.encode(pergunta)
    cursor.execute("SELECT texto, embedding FROM documentos")
    resultados = []

    for texto, emb_json in cursor.fetchall():
        emb = np.array(json.loads(emb_json))
        sim = np.dot(q_emb, emb) / (
            np.linalg.norm(q_emb) * np.linalg.norm(emb)
        )
        resultados.append((texto, sim))

    resultados.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in resultados[:top_k]]

# ======================
# ENDPOINTS
# ======================

@app.post("/documento")
def adicionar_documento(d: Documento):
    salvar_documento(d.texto)
    return {"status": "Documento indexado com embedding"}

@app.post("/chat")
def chat(p: Pergunta):
    docs = buscar_similares(p.pergunta)

    if not docs:
        raise HTTPException(404, "Nenhum contexto encontrado")

    return {
        "resposta": "Resposta baseada nos documentos recuperados",
        "contexto": docs
    }

@app.get("/status")
def status():
    return {"status": "RAG com embeddings ativo"}
