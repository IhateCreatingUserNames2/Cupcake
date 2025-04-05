# memory_weighting.py
import uuid

def add_weighted_memory(collection, text, embedding, emotion_score, source="user"):
    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "emotion_score": emotion_score,
            "source": source  # "user" ou "thought"
        }]
    )


def get_weighted_memories(collection, top_k=5, allowed_sources=["user"]):
    memories = collection.get(include=['documents', 'metadatas'])
    filtered_memories = [
        (doc, meta) for doc, meta in zip(memories['documents'], memories['metadatas'])
        if meta.get("source", "user") in allowed_sources
    ]
    sorted_memories = sorted(filtered_memories, key=lambda x: x[1]['emotion_score'], reverse=True)
    return [doc for doc, _ in sorted_memories[:top_k]]

def inject_memory(collection, document, embed_fn, emotion_score=0.9, source="legado"):
    embedding = embed_fn(document).tolist()  # <- garante que está como lista
    metadata = {'emotion_score': emotion_score, 'source': source}
    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata]
    )
    print(f"🧠 Memória injetada: {document} (fonte: {source})")

def search_similar_memories(collection, query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=['documents']
    )
    return results['documents'][0] if results['documents'] else []


