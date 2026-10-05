# VisionLab AI - Praktikum 04: Pengolahan Citra Digital, Konvolusi & Deteksi Tepi

Aplikasi Laboratorium Interaktif Web untuk Pembelajaran Mahasiswa pada Mata Kuliah **Komputer Visi dan Natural Language Processing (NLP)**.

---

## 🎯 Identitas Praktikum & Capaian Pembelajaran

- **Topik:** Teknik dasar pengolahan citra seperti filtering, konvolusi, dan deteksi tepi.
- **Kemampuan Akhir yang Diharapkan:** Mahasiswa mampu memahami konsep matematis dan menerapkan secara praktis teknik dasar pengolahan citra (*filtering*, *2D convolution*, *edge detection*) dalam pemecahan masalah Computer Vision nyata.
- **Materi Ajar:**
  1. **Spatial Image Filtering:** Mean/Box, Gaussian Blur, Median, Bilateral Filter, Sharpening, Unsharp Masking, Min/Max Morphological.
  2. **Operasi Konvolusi 2D:** Interactive sliding window kernel arithmetic, divisor normalization, offset bias, padding border mode, serta step-by-step pixel math probe.
  3. **Deteksi Tepi (First & Second-Order Derivatives):** Operator Sobel ($G_x, G_y, |G|, \theta$), Prewitt, Scharr, Roberts Cross, Laplacian, dan Laplacian of Gaussian (LoG).
  4. **Canny 5-Stage Edge Pipeline:** Grayscale $\to$ Gaussian Smoothing $\to$ Gradient Calculation $\to$ Non-Maximum Suppression (NMS) $\to$ Dual Threshold & Hysteresis Tracking.
  5. **Implementasi Nyata Computer Vision:**
     - *Lane Detection* pada Autonomous Driving (Canny + ROI + Hough Transform)
     - *Document Scanner & Perspective Rectification* (Canny + Contour + Homography Warp)
     - *License Plate Feature Localization* (Blackhat + Sobel Vertical + Otsu)
     - *Medical X-Ray Bone Enhancement* (CLAHE + Laplacian + High-Boost)
  6. **Live Camera Real-time Processing:** Pemrosesan streaming webcam instan.
  7. **Evaluasi & Kuis Pemahaman:** 10 Soal interaktif dilengkapi skor & pembahasan mendalam.

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Prasyarat Sistem
Pastikan Python 3 telah terpasang beserta dependensi yang diperlukan:
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Images (Opsional jika ingin regenerate)
```bash
python3 generate_samples.py
```

### 2. Menjalankan Aplikasi Streamlit (Pilih salah satu perintah)

**Opsi A (Rekomendasi Utama):**
```bash
python3 -m streamlit run streamlit_app.py
```

**Opsi B (Menggunakan Python Launcher):**
```bash
python3 run.py
```

**Opsi C (Menggunakan Shell Script):**
```bash
./run_streamlit.sh
```
Akses aplikasi melalui browser di: **`http://localhost:8501`**.

### 3. Menjalankan Server Flask Alternatif
```bash
python3 app.py
```
Akses di: **`http://localhost:5004`**.

---

## 📁 Struktur Direktori Proyek

```
Praktikum04/
├── app.py                   # Backend Flask & API pengolahan citra OpenCV/NumPy/SciPy
├── generate_samples.py      # Generator sampel dataset citra realistis
├── requirements.txt         # Daftar paket dependensi Python
├── README.md                # Dokumentasi petunjuk praktikum
├── templates/
│   └── index.html           # Antarmuka SPA (Single Page App) modern dark glassmorphism
└── static/
    ├── css/
    │   └── style.css        # Desain visual, tema gelap neon, responsif UI
    ├── js/
    │   ├── app.js           # Manajemen state, tab switching, filter & histogram canvas
    │   ├── convolution.js   # Editor matriks konvolusi 2D & interactive pixel probe
    │   ├── canny_pipeline.js# Visualisasi deteksi tepi & 5 tahapan pipeline Canny
    │   ├── live_camera.js   # Pemrosesan real-time kamera berbasis TypedArray/Canvas
    │   └── quiz.js          # Mesin kuis evaluasi mahasiswa & loader studi kasus
    └── samples/             # Dataset sampel citra (Jalan raya, dokumen, plat, xray, dll)
```

---

## 📚 Ringkasan Formula Matematis

### 1. Konvolusi 2D Diskrit
$$g(x,y) = f(x,y) * h(x,y) = \sum_{i=-a}^{a} \sum_{j=-b}^{b} f(x-i, y-j) \cdot h(i,j)$$

### 2. Filter Gaussian 2D
$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

### 3. Gradien & Arah Vektor Tepi Sobel
$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I$$
$$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$

### 4. Laplacian (Turunan Orde-2 Isotropik)
$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix} * f$$

---

## 👨‍🏫 Panduan Aktivitas Praktikum untuk Mahasiswa

1. **Eksplorasi Derau (Noise Sandbox):** Tambahkan derau *Salt & Pepper* pada citra, lalu bandingkan efektivitas **Mean Filter** vs **Median Filter**. Amati nilai PSNR dan SSIM.
2. **Kalkulasi Titik Pixel Konvolusi:** Pada modul *Convolution & Probe*, arahkan kursor ke area tepi dan area rata. Amati bagaimana perkalian matriks $P \odot K$ menghasilkan nilai nol pada area homogen dan nilai tinggi pada area tepi.
3. **Analisis Canny 5-Stage:** Ubah *Low Threshold* ($T_{low}$) dan *High Threshold* ($T_{high}$). Amati pergeseran jumlah *Strong Edges* (hijau) vs *Weak Edges* (kuning) serta eliminasi tepi palsu (*false edges*).
4. **Studi Kasus Computer Vision:** Amati tahapan *Lane Detection* dan *Document Scanner*. Pahami bagaimana deteksi tepi menjadi blok fondasi penting sebelum algoritma tingkat lanjut (*Hough Transform*, *Perspective Homography*).
5. **Penyelesaian Kuis:** Kerjakan 10 soal evaluasi pada Modul 7 dan pelajari pembahasan pada setiap jawaban yang salah.
