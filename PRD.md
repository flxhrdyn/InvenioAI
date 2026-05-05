# PRD: InvenioAI LinkedIn-Ready Overhaul

## Problem Statement
Aplikasi InvenioAI saat ini memiliki mesin RAG yang fungsional, namun belum memenuhi standar "Production-Ready" untuk dipamerkan kepada recruiter atau di LinkedIn. Masalah utama meliputi:
- **UX yang Kaku**: Input chat terlalu besar (tinggi), gelembung pesan menggunakan desain default yang kurang premium, dan loading halaman terasa lambat.
- **Kurangnya Transparansi**: Tidak ada sitasi halaman (page citations) yang jelas, sehingga user sulit memverifikasi jawaban AI terhadap dokumen asli.
- **Reliabilitas Dashboard**: Metrik kueri sering tertahan di angka 0 dan tidak menampilkan data performa (latensi) yang kritikal bagi penilaian teknis.

## Solution
Melakukan perombakan menyeluruh (Overhaul) pada sisi Frontend dan Metric Logging untuk menciptakan pengalaman "Gemini-like" yang transparan, cepat, dan terukur. Fokus utama adalah pada rekayasa *State Management* agar aplikasi terasa instan dan persisten.

## User Stories
1. **As a Recruiter**, I want to see exactly which page a piece of information came from, so that I can trust the AI's accuracy and verify its sources.
2. **As a User**, I want a slim and modern chat input, so that the interface feels professional and "airy" instead of clunky.
3. **As a Developer**, I want to see query latency (response time) in the dashboard, so that I can monitor and showcase system performance to technical reviewers.
4. **As a User**, I want my messages to appear instantly in the chat without waiting for the backend response, so that the app feels responsive.
5. **As a User**, I want the list of indexed documents to remain visible in the sidebar even after I navigate between pages or send a message, so that I don't lose context.
6. **As a User**, I want to click on a "Source Pill" to see the original context, so that I can quickly verify facts without leaving the chat flow.
7. **As a User**, I want to delete documents individually from the library, so that I don't have to clear the entire database to remove just one file.

## Implementation Decisions

### 1. Frontend: Premium UI/UX & State Management
- **Instant Messaging Feedback**: Merender pesan pengguna ke UI secara asinkron sebelum melakukan panggilan API ke backend untuk menghilangkan efek "Opaque/Running" yang mengganggu.
- **Persistent Sidebar Library**: Menyimpan daftar dokumen terindeks ke dalam `st.session_state` dan melakukan sinkronisasi otomatis agar data tidak hilang saat navigasi halaman.
- **Individual Deletion UI**: Menambahkan kontrol (ikon trash) pada setiap item dokumen di sidebar untuk penghapusan granular.
- **Slim Chat Input**: Mengurangi padding vertikal dan mengatur `textarea` agar tipis (sleek) saat kosong. Menggunakan desain *floating* dengan *shadow* yang halus.
- **Modern Message Bubbles**: Mengganti ikon kotak kaku dengan avatar bulat dan tipografi yang lebih bersih (Inter/Roboto).
- **Source Pills Component**: Mengimplementasikan barisan tombol kecil (`Hal. X`) di bawah jawaban AI.
- **Interactive Citations**: Menggunakan metadata `page_number` untuk memetakan sitasi ke potongan teks asli (menggunakan *expander* atau *popover*).

### 2. Backend: Performance & Reliability Metrics
- **Granular Document Deletion**: Membuat endpoint untuk menghapus poin spesifik di Qdrant berdasarkan metadata `source_file`.
- **Metric Logging Fix**: Memperbaiki sinkronisasi file `metrics.json` agar jumlah kueri tercatat secara akurat dan *real-time*.
- **Latency Tracking**: Menambahkan penghitung waktu (*timer*) pada proses RAG dan menyimpannya ke dalam metrik.
- **Metadata Passthrough**: Memastikan modul `retriever.py` mengirimkan seluruh objek metadata (halaman, nama file) ke frontend.

### 3. Architecture & Performance
- **Document Caching**: Menerapkan `st.cache_data` pada fungsi `list_documents` agar sidebar tidak me-load ulang secara sinkron yang menyebabkan *delay*.
- **REST Stability**: Memastikan koneksi ke Qdrant tetap menggunakan REST API untuk menghindari masalah port/firewall pada lingkungan demo.

## Testing Decisions
- **Behavioral Testing**: Menambahkan integrasi test untuk memastikan latensi kueri tercatat dengan benar di file metrik.
- **Metadata Verification**: Unit test untuk memastikan `process_pdf_documents` menghasilkan metadata `page_number` yang akurat (1-indexed).

## Out of Scope
- Fitur feedback Thumbs Up/Down (untuk menjaga kesederhanaan demo).
- User Authentication (Multi-user support).
- Pencarian lintas dokumen yang sangat kompleks (multi-doc synthesis UI).

## Further Notes
Aplikasi ini ditujukan untuk **Tracer Bullet** vertikal: menunjukkan kemampuan full-stack AI Engineering dari pemrosesan dokumen, optimasi database vektor, hingga desain UI modern.
