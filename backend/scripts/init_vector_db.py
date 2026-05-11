import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from app.qdrant_conn import get_qdrant_client
from app.config import QDRANT_COLLECTION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_vector_db():
    logger.info("Mempersiapkan Vector Database (Qdrant)...")
    
    client = get_qdrant_client()
    
    # Cek apakah koleksi sudah ada
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if QDRANT_COLLECTION in collection_names:
        logger.warning(f"Koleksi '{QDRANT_COLLECTION}' ditemukan.")
        logger.warning("Karena kita beralih ke Native Hybrid Search (BM42), "
                       "kita harus menghapus dan membuat ulang koleksi dengan skema baru.")
        
        if "--force" in sys.argv:
            client.delete_collection(QDRANT_COLLECTION)
            logger.info(f"Koleksi '{QDRANT_COLLECTION}' berhasil dihapus.")
        else:
            logger.info("Operasi dibatalkan. Tambahkan flag --force untuk menghapus secara otomatis.")
            logger.info("Contoh: uv run python scripts/init_vector_db.py --force")
            return
    else:
        logger.info(f"Koleksi '{QDRANT_COLLECTION}' belum ada. Membuat koleksi baru...")

    from qdrant_client import models
    from app.embeddings import get_embeddings
    
    embeddings = get_embeddings()
    # Get vector size dynamically
    vector_size = len(embeddings.embed_query("test"))
    
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    
    logger.info(f"Koleksi '{QDRANT_COLLECTION}' berhasil dibuat dengan dukungan Hybrid Search (Dense + Sparse).")
    logger.info("Selesai! Vector DB sekarang siap.")
    logger.info("Gunakan app/index_data.py (atau script ingestion Anda) untuk mulai memasukkan dokumen baru.")

if __name__ == "__main__":
    reset_vector_db()
