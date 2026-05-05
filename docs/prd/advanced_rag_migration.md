# PRD: Advanced RAG Migration & Optimization (Groq, Hybrid Fusion, Multilingual)

## Problem Statement

Sistem RAG saat ini memiliki beberapa kelemahan fundamental yang membatasi akurasi dan pengalaman pengguna:

1. **LLM**: Bergantung pada Gemini API yang berbayar, dengan integrasi yang tightly coupled sehingga sulit diganti.
2. **Ingestion**: `PyPDFLoader` dan `RecursiveCharacterTextSplitter` berbasis karakter sering memotong konteks penting di tengah paragraf, terutama pada buku terstruktur (bab/sub-bab). Tidak ada informasi posisi (bab, halaman, seksi) yang ikut disimpan bersama chunk.
3. **Retrieval**: Strategi single-query tidak mampu menangkap informasi yang tersebar di berbagai bab. Query rewrite dilakukan secara sinkron sebelum retrieval dimulai, menambah latensi yang tidak perlu. Terdapat double-call ke Qdrant hanya untuk mendapatkan skor relevansi.
4. **Embedding & Reranking**: Model embedding (`all-MiniLM-L6-v2`) dan reranker (`ms-marco-MiniLM-L-6-v2`) hanya dilatih pada teks bahasa Inggris, sehingga akurasi menurun drastis untuk dokumen multibahasa atau berbahasa Indonesia.
5. **Latensi & UX**: Tidak ada streaming, sehingga pengguna harus menunggu seluruh jawaban selesai diproses. Jika stream terputus karena error, pengguna hanya melihat blank screen tanpa penjelasan.
6. **Caching**: Tidak ada mekanisme caching sehingga query yang identik tetap memanggil LLM dan Vector DB setiap saat.
7. **API**: Endpoint berbasis job polling (`/query/jobs`) menambah kompleksitas arsitektur yang tidak diperlukan dengan adanya streaming.

## Solution

Membangun ulang pipeline RAG secara menyeluruh dengan pendekatan bertahap yang dioptimalkan untuk RAM 8GB, deployment di Hugging Face Spaces, dan dokumen berbahasa Indonesia/multibahasa:

- **LLM**: Migrasi ke Llama 3.1 8B via Groq (async, streaming, efisiensi token).
- **Ingestion**: PyMuPDF + Semantic Chunking + Parent-Document Retrieval + Rich Metadata Injection (bab, sub-bab, halaman).
- **Retrieval**: RAG Fusion (2 variasi query paralel + RRF) + Hybrid Search (Dense + BM25), dengan async query rewrite yang berjalan paralel dengan inisialisasi retriever.
- **Embedding**: FastEmbed + `paraphrase-multilingual-MiniLM-L12-v2` (ONNX Runtime, tanpa PyTorch, hemat ~1GB RAM).
- **Reranking**: FlashRank + model multilingual (ONNX Runtime).
- **Caching**: Dual-layer cache — Diskcache untuk lokal, Redis via Docker untuk produksi (HF Spaces).
- **API**: Dua endpoint — `POST /query` (async, non-streaming) dan `POST /query/stream` (SSE streaming). Endpoint `/query/jobs` dihapus.
- **UX**: Graceful error boundary di sisi Streamlit untuk menangani stream terputus.

## User Stories

1. Sebagai pengguna buku PDF berbahasa Indonesia, saya ingin sistem memahami struktur bab dan sub-bab, sehingga jawaban yang diberikan memiliki konteks yang utuh dan tidak terpotong di tengah kalimat.
2. Sebagai pengguna yang mengutamakan kecepatan, saya ingin jawaban muncul secara streaming (kata demi kata) agar saya bisa mulai membaca tanpa menunggu proses LLM selesai sepenuhnya.
3. Sebagai peneliti, saya ingin sistem mencari informasi tidak hanya berdasarkan kata kunci (lexical), tetapi juga berdasarkan makna tersirat (semantic), sehingga pertanyaan yang ambigu tetap mendapatkan jawaban relevan.
4. Sebagai pengguna multi-turn, saya ingin sistem mengingat konteks percakapan sebelumnya, sehingga saya bisa bertanya lanjutan seperti "jelaskan lebih detail" tanpa mengulang konteks dari awal.
5. Sebagai developer, saya ingin sistem ini ringan di lokal (RAM < 7GB) namun siap dideploy ke Hugging Face Spaces dengan konfigurasi minimal.
6. Sebagai admin, saya ingin hasil query yang sering ditanyakan disimpan dalam cache, sehingga menghemat biaya token Groq dan mempercepat respons untuk query yang berulang.
7. Sebagai pengguna, saya ingin sistem otomatis membuat beberapa variasi pertanyaan saya (Query Expansion via RAG Fusion) sehingga informasi yang tersebar di berbagai bab tetap dapat ditemukan.
8. Sebagai pengguna, jika terjadi error (misalnya koneksi ke Groq terputus atau rate limit tercapai), saya ingin melihat pesan error yang jelas dan ramah, bukan blank screen atau teks yang terpotong tiba-tiba.
9. Sebagai developer, saya ingin konfigurasi provider LLM, jumlah variasi query (RAG Fusion), dan jenis cache dapat diubah hanya melalui file `.env` tanpa menyentuh kode.
10. Sebagai pengguna, saya ingin setiap jawaban disertai informasi sumber yang jelas (nama file, nomor halaman, bab) sehingga saya bisa memverifikasi informasi ke dokumen asli.
11. Sebagai admin, saya ingin dashboard metrik yang ringkas menampilkan waktu retrieval, waktu generasi, jumlah dokumen yang diproses, dan mode pencarian yang digunakan (hybrid/dense).
12. Sebagai developer, saya ingin pipeline ingestion berjalan secara batch untuk menjaga konsumsi RAM tetap stabil meskipun memproses buku yang sangat tebal (500+ halaman).

## Implementation Decisions

### LLM Engine
- Migrasi dari `langchain-google-genai` ke `langchain-groq` menggunakan model `llama3-8b-8192`.
- Default model dikonfigurasi via env var `INVENIOAI_LLM_MODEL` agar mudah diganti.
- Jumlah variasi query RAG Fusion dibatasi ke **2** secara default (dikonfigurasi via `INVENIOAI_NUM_FUSION_QUERIES`) untuk menjaga konsumsi Groq rate limit (target: ~10 query/menit di free tier).

### Ingestion Module
- Ganti `PyPDFLoader` dengan `PyMuPDFLoader` untuk ekstraksi teks yang lebih bersih dan akurat, khususnya pada buku dengan layout kompleks.
- Terapkan **Semantic Chunking** (memotong berdasarkan perubahan makna, bukan jumlah karakter) dengan batasan ukuran chunk untuk menjaga RAM 8GB.
- Terapkan **Parent-Document Retrieval**: child chunks (kecil, untuk pencarian presisi) disimpan di Qdrant, parent chunks (besar, untuk konteks jawaban) disimpan di local document store.
- Terapkan **Rich Metadata Injection**: setiap chunk diperkaya dengan metadata `chapter`, `section`, `page_number`, `source_file` yang diekstrak dari struktur dokumen saat indexing. Ini menggantikan Contextual Chunking (yang membutuhkan LLM call per chunk) tanpa overhead biaya API.

### Embedding Stack
- Ganti `sentence-transformers` (PyTorch) dengan **FastEmbed** (ONNX Runtime) menggunakan model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- FastEmbed mengeliminasi ketergantungan pada PyTorch untuk inferensi embedding, menghemat ~1GB RAM dan mempercepat indexing 2-3x.
- FastEmbed terintegrasi natively dengan Qdrant (same vendor).

### Reranker Stack
- Ganti `cross-encoder` (PyTorch) dengan **FlashRank** (ONNX Runtime) menggunakan model multilingual `ms-marco-MultiBERT-L-12`.
- Stack embedding + reranker sepenuhnya berbasis ONNX Runtime, bebas PyTorch.

### Retrieval Strategy
- **Async Query Rewrite**: Proses `rewrite_query` (mengubah pertanyaan chat history-aware menjadi standalone query) dijalankan secara async di awal pipeline, paralel dengan inisialisasi retriever.
- **RAG Fusion**: Generate 2 variasi query secara paralel menggunakan asyncio, lalu gabungkan hasil pencarian Dense + BM25 menggunakan algoritma Reciprocal Rank Fusion (RRF).
- **Hybrid Search**: Tetap mempertahankan Hybrid Search (Dense Vector + BM25 Lexical). RRF scores dari proses fusion digunakan langsung sebagai skor relevansi untuk dashboard, menghilangkan double-call ke Qdrant.

### Caching Strategy
- Modul `CacheManager` sebagai abstraksi tunggal:
  - **Lokal (development)**: Diskcache (SQLite-based, zero infrastructure).
  - **Produksi (HF Spaces / Docker)**: Redis via Docker Compose.
  - Deteksi otomatis berdasarkan env var `CACHE_TYPE=redis|diskcache`.
- Cache TTL dapat dikonfigurasi via env var.

### API Contract
- `POST /query` — Async, mengembalikan JSON lengkap (answer, sources, metrics). Digunakan untuk testing programmatic.
- `POST /query/stream` — Async SSE (Server-Sent Events), mengirimkan token jawaban secara real-time ke Streamlit UI.
- `DELETE /query/jobs` endpoint — **Dihapus** (digantikan oleh SSE streaming).

### Dashboard & Metrics
- Dashboard disederhanakan: hanya menampilkan metrik yang tersedia gratis dari pipeline (total_time, retrieval_time, generation_time, docs_retrieved, retrieval_mode, rrf_scores).
- Menghilangkan panggilan `similarity_search_with_relevance_scores` yang redundan ke Qdrant.

### Error Handling & UX
- Graceful error boundary di Streamlit: jika SSE stream terputus karena error (Groq rate limit, network timeout, dll), UI menampilkan pesan yang ramah dan tombol "Coba Lagi" — tidak pernah blank screen.

### Deployment
- `Dockerfile` dan `docker-compose.yml` untuk menjalankan App + Qdrant + Redis secara bersamaan di lokal dan HF Spaces.
- Ingestion berjalan dalam batch kecil untuk menjaga RAM < 7GB saat memproses buku tebal.

## Testing Decisions

Prinsip utama: **hanya uji perilaku eksternal modul, bukan detail implementasi internal**. Sebuah test yang baik tidak perlu diubah ketika refactoring internal modul dilakukan.

### Modul yang Diuji
- **RRF Fusion Logic** (`retriever` module): Uji bahwa dokumen dengan rank lebih tinggi di beberapa daftar mendapatkan skor fusi lebih tinggi. Ini adalah logika murni yang mudah diisolasi.
- **CacheManager**: Uji bahwa cache hit mengembalikan data identik tanpa memanggil retriever, dan cache miss memicu pencarian ke Vector DB.
- **Ingestion Pipeline** (`index_data` module): Uji bahwa setelah indexing sebuah PDF, koleksi Qdrant berisi chunk dengan metadata yang benar (page_number, source_file, chapter).
- **Streaming Endpoint** (`/query/stream`): Uji bahwa endpoint mengembalikan `Content-Type: text/event-stream` dan bahwa token mengalir secara bertahap.

### Performa Benchmarking
- Target **Time to First Token (TTFT)**: < 500ms.
- Target **RAM usage** saat indexing buku 100+ halaman: < 7GB.
- Benchmark dilakukan dengan script otomatis yang membandingkan pipeline baru (async + fusion) vs pipeline lama (sync + single-query).

## Out of Scope
- Migrasi Vector DB dari Qdrant ke provider lain.
- Penanganan dokumen selain format PDF (DOCX, Excel, dll).
- Fitur multi-user authentication.
- **Contextual Chunking** (LLM call per chunk): Tidak kompatibel dengan rate limit Groq free tier untuk proses ingestion. Dapat ditambahkan di masa depan sebagai fitur opsional jika menggunakan model lokal (Ollama).
- Fallback otomatis ke LLM provider alternatif (SambaNova/Cerebras): Digantikan oleh strategi pembatasan jumlah query fusion (2 variasi) untuk menjaga konsumsi rate limit tetap aman.

## Further Notes

- **Urutan migrasi embedding**: Saat mengganti model embedding dari `all-MiniLM-L6-v2` ke `paraphrase-multilingual-MiniLM-L12-v2`, seluruh koleksi Qdrant **harus di-recreate** karena dimensi vektor berubah. Ini adalah breaking change yang perlu dikomunikasikan ke pengguna.
- **Stack bebas PyTorch**: Dengan FastEmbed + FlashRank berbasis ONNX Runtime, deployment di HF Spaces menjadi lebih stabil karena tidak perlu menginstall paket PyTorch yang besar (~2GB). Ini secara signifikan mempercepat cold start dan mengurangi memory footprint di container.
- **Konfigurasi via `.env`**: Semua parameter kritis (`GROQ_API_KEY`, `INVENIOAI_LLM_MODEL`, `INVENIOAI_NUM_FUSION_QUERIES`, `CACHE_TYPE`, `REDIS_URL`) harus tersedia di `.env.example` yang terdokumentasi dengan baik sebagai panduan deployment.
