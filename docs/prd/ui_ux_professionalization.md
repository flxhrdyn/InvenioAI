# PRD: InvenioAI UI/UX Professionalization (Hybrid Minimalist & Transparent RAG)

## Problem Statement

Meskipun sistem RAG InvenioAI sudah fungsional secara teknis, antarmuka pengguna (UI) saat ini masih menghadapi beberapa kendala:
1. **Opasitas Proses**: Pengguna tidak mengetahui apa yang dilakukan sistem "di balik layar" (seperti proses rewrite query atau retrieval) karena statusnya hanya muncul sekejap lalu hilang.
2. **Kerapuhan UI**: Kode CSS yang terlalu agresif menimpa elemen dasar Streamlit membuat aplikasi sulit dipelihara dan berisiko berantakan jika ada pembaruan versi Streamlit.
3. **Keterbacaan Sitasi**: Format sitasi saat ini (menggunakan HTML custom) terasa kurang menyatu dengan desain aplikasi dan sulit untuk dibaca dengan cepat.
4. **Theme Inconsistency**: Pengaturan tema masih tersebar antara file Python dan CSS, tidak memanfaatkan fitur Native Theme dari Streamlit.

## Solution

Melakukan transformasi UI menjadi **Hybrid Minimalist** yang mengutamakan transparansi proses (Thinking Process) dan kemudahan verifikasi sumber (Sources), dengan arsitektur yang lebih stabil dan maintainable:
- **Thinking Process**: Implementasi blok status permanen yang menunjukkan langkah-langkah logika AI (Chain-of-Thought).
- **Hybrid Minimalist UI**: Menggunakan komponen asli Streamlit (`st.chat_message`) namun dipoles dengan CSS premium untuk tipografi, spasi, dan warna.
- **Native Theme**: Mengintegrasikan palette warna premium langsung ke konfigurasi Streamlit agar sinkron secara native.
- **Improved Citations**: Menggunakan `st.expander` yang distyling khusus untuk menampilkan sumber referensi secara bersih dan terstruktur.

## User Stories

1. Sebagai pengguna, saya ingin melihat **Thinking Process** AI (misal: "Sedang mencari di dokumen...", "Sedang merumuskan jawaban...") agar saya merasa yakin bahwa AI benar-benar memproses data saya secara cerdas.
2. Sebagai peneliti, saya ingin blok **Thinking Process** tetap ada (collapsible) bahkan setelah jawaban selesai, sehingga saya bisa meninjau kembali kueri apa yang digunakan AI untuk mencari informasi.
3. Sebagai pengguna, saya ingin **Sitasi (Sources)** ditampilkan secara rapi di bawah jawaban menggunakan gaya asli Streamlit yang bersih, sehingga saya bisa memverifikasi informasi dengan cepat.
4. Sebagai pengguna, saya ingin setiap sumber referensi menunjukkan potongan teks (snippets) yang relevan dan nomor halaman yang jelas di dalam folder expander yang cantik.
5. Sebagai developer, saya ingin kode UI lebih bersih dengan meminimalkan penggunaan HTML/CSS kustom yang kompleks, agar aplikasi lebih stabil dan mudah dikembangkan.
6. Sebagai pengguna, saya ingin layout chat terasa luas (airy) dan minimalis seperti aplikasi AI modern (Gemini/Claude), namun tetap dengan nuansa warna premium (Dark Mode) yang konsisten.
7. Sebagai pengguna, saya ingin melihat indikator status koneksi ke database atau backend yang jelas di sidebar.

## Implementation Decisions

### 1. Hybrid Minimalist Architecture
- Menghapus CSS agresif yang menimpa `st.chat_message` secara struktural.
- Mengandalkan `st.chat_message` asli untuk reliabilitas layout.
- Menambahkan CSS ringan hanya untuk:
    - Tipografi (menggunakan Google Fonts: Inter/Outfit).
    - Styling `st.expander` agar memiliki border halus dan shadow premium.
    - Penyesuaian lebar kontainer (`max-width: 1200px`) agar chat tetap terpusat dan mudah dibaca di layar lebar.

### 2. Transparent Thinking Process
- Menggunakan komponen `st.status` untuk menangkap event dari backend (`rewriting`, `retrieving`, `reranking`, `generating`).
- Status ini akan bersifat permanen (bisa di-collapse) di bagian atas bubble jawaban asisten.
- Detail di dalam status akan menampilkan kueri hasil rewrite dan jumlah dokumen yang ditemukan.

### 3. Native Theme Integration
- Membuat file `.streamlit/config.toml` untuk mendefinisikan `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, dan `textColor`.
- Sinkronisasi warna antara `theme.py` dan `config.toml` sebagai *Single Source of Truth*.

### 4. Professional Citations Layout
- Mengganti kode HTML `<details>` dengan `st.expander`.
- Styling expander dengan CSS agar memiliki label "📚 Lihat Sumber Referensi" yang menonjol namun elegan.
- Di dalam expander, gunakan layout "Quote" untuk snippet teks sumber agar terlihat berbeda dari jawaban utama.

## Testing Decisions

- **Visual Regression**: Memastikan layout tidak "pecah" saat berpindah antara resolusi mobile dan desktop.
- **State Persistence**: Memastikan blok Thinking Process dan Sources tetap konsisten saat user melakukan scroll atau navigasi sidebar.
- **Theme Stability**: Memastikan aplikasi tetap terbaca dengan baik meskipun browser user memiliki setelan default Light Mode (dengan tetap memaksakan Dark Theme kita sebagai default).

## Out of Scope
- Implementasi sistem voting (Thumbs Up/Down).
- Penambahan mode Light Mode secara fungsional (untuk saat ini fokus tetap pada Premium Dark Theme).
- Perubahan pada logika RAG di backend (fokus murni pada presentasi data di frontend).

## Further Notes
- Transformasi ini akan membuat file `streamlit_app.py` jauh lebih ringkas (mengurangi baris kode CSS hingga ~50%).
- Penggunaan komponen asli Streamlit akan meningkatkan performa rendering di sisi browser, terutama untuk percakapan yang panjang.
