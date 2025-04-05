import chromadb

# Inicializa cliente persistente
client_db = chromadb.PersistentClient(path="./cupcake_memory_db")

# Tenta remover coleção anterior (ignora erro se não existir)
try:
    client_db.delete_collection(name="cupcake_memory")
except:
    pass

# Cria nova coleção com suporte a metadados
collection = client_db.create_collection(
    name="cupcake_memory",
    metadata={"hnsw:space": "cosine"}  # importante para buscas semânticas
)
