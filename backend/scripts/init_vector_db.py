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
        logger.warning("Karena kita menggunakan model embedding baru (FastEmbed Multilingual), "
                       "kita harus menghapus koleksi lama agar tidak terjadi konflik vector space.")
        
        if "--force" in sys.argv:
            client.delete_collection(QDRANT_COLLECTION)
            logger.info(f"Koleksi '{QDRANT_COLLECTION}' berhasil dihapus.")
        else:
            logger.info("Operasi dibatalkan. Tambahkan flag --force untuk menghapus secara otomatis.")
            logger.info("Contoh: uv run python scripts/init_vector_db.py --force")
            return
    else:
        logger.info(f"Koleksi '{QDRANT_COLLECTION}' belum ada. Aman untuk melanjutkan.")
        
    logger.info("Selesai! Vector DB sekarang bersih.")
    logger.info("Gunakan app/index_data.py (atau script ingestion Anda) untuk mulai memasukkan dokumen baru.")

if __name__ == "__main__":
    reset_vector_db()
