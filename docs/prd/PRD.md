# PRD: InvenioAI Intelligent Reasoning Overhaul

## Problem Statement
Aplikasi InvenioAI saat ini memiliki mesin RAG yang fungsional, namun interaksinya masih terasa seperti "black box". User tidak tahu bagaimana AI menghubungkan titik-titik informasi dari berbagai dokumen sebelum memberikan jawaban. Masalah utama meliputi:
- **Kurangnya Transparansi Logika**: User hanya melihat jawaban akhir tanpa mengetahui proses penalaran (Chain-of-Thought) di baliknya.
- **Loading State yang Generik**: Saat AI bekerja, user hanya melihat spinner standar yang kurang memberikan konteks tentang apa yang sedang dilakukan sistem.
- **Reliabilitas Metrik**: Beberapa metrik kualitas masih memerlukan kalibrasi agar mencerminkan performa nyata dari model reranker yang digunakan.

## Solution
Mengimplementasikan **"Dynamic Reasoning Engine"** yang menampilkan proses berpikir teknis (CoT) LLM secara transparan. Fitur ini akan menggunakan animasi pemuatan yang informatif saat AI "berpikir", yang kemudian bertransformasi menjadi dropdown detail (expander) setelah jawaban mulai diberikan.

## User Stories
1. **As a User**, I want to see a technical snippet of what the AI is currently thinking while I wait, so the app feels alive and intelligent.
2. **As a User**, I want the thinking process to collapse into a dropdown after the answer starts, so it doesn't clutter the main conversation.
3. **As a Developer**, I want to verify the AI's logic (CoT) in the dashboard history, so I can debug and improve the retrieval quality.
4. **As a Recruiter**, I want to see a clear link between the AI's "Thought Process" and the final "Sources", demonstrating a robust RAG architecture.

## Implementation Decisions

### 1. Dynamic Reasoning Protocol (CoT)
- **Backend Tagging**: LLM diinstruksikan melalui `RAG_PROMPT` untuk membungkus penalaran teknisnya di dalam tag `<thinking>...</thinking>`.
- **Hybrid Streaming**: Backend men-stream konten di dalam tag sebagai `step: "thinking"` dan konten di luar tag sebagai `step: "token"`.
- **Frontend Transformation**: 
    - Saat menerima `step: "thinking"`, UI menampilkan animasi "Thinking..." dengan potongan teks penalaran terbaru.
    - Begitu `step: "token"` pertama tiba, animasi tersebut berubah menjadi `st.expander` ("Thought Process") dan jawaban final mulai di-stream di bawahnya.

### 2. UI/UX Professionalization (Updated)
- **Minimalist Chat Layout**: Menggunakan gelembung chat native yang airy dengan input yang ramping.
- **Interactive Source Pills**: Menampilkan sitasi dokumen di dalam gelembung pesan yang dapat diklik untuk melihat konteks asli.
- **Sidebar Persistence**: Daftar dokumen terindeks tetap sinkron dan tidak hilang saat navigasi antar halaman.

### 3. Analytics & Metrics
- **Refined Thresholds**: Menggunakan threshold relevansi yang lebih sensitif (`0.01`) untuk mengakomodasi karakteristik skor model reranker modern (FlashRank).
- **History Enhancement**: Menyimpan data `thoughts` ke dalam riwayat kueri di `metrics.json` untuk evaluasi kualitas jangka panjang.

## Testing Decisions
- **Parser Validation**: Memastikan parser frontend mampu menangani tag XML yang tidak tertutup dengan sempurna (graceful degradation).
- **Latency Impact**: Memantau apakah penambahan CoT meningkatkan TTFT (Time To First Token) secara signifikan dan melakukan optimasi prompt jika diperlukan.

## Out of Scope
- Pengeditan manual pada teks CoT yang sudah dihasilkan.
- Dukungan multi-bahasa untuk proses penalaran (default mengikuti bahasa pertanyaan).

## Further Notes
Fitur ini bertujuan untuk menaikkan level InvenioAI dari sekadar chatbot RAG biasa menjadi platform **"Agentic-Reasoning"** yang transparan dan dapat dipercaya oleh pengguna profesional.
