import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="VisionLab AI | Praktikum 04",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean White / Light Theme Styling
st.markdown("""
<style>
    /* Main container and typography */
    .main {
        background-color: #f8fafc;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: #2563eb;
        font-weight: 700;
    }
    /* Badges and Highlights */
    .module-badge {
        background-color: #eff6ff;
        color: #2563eb;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #bfdbfe;
        display: inline-block;
        margin-bottom: 8px;
    }
    .callout-box {
        background-color: #ffffff;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- UTILITY FUNCTIONS -----------------
@st.cache_data
def load_sample_image(filename):
    path = os.path.join('static/samples', filename)
    if os.path.exists(path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Fallback dummy
    dummy = np.zeros((300, 400, 3), dtype=np.uint8)
    dummy[:] = (240, 240, 240)
    cv2.putText(dummy, "Image Not Found", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    return dummy

def add_noise(image, noise_type, intensity=0.05):
    noisy = image.copy()
    if noise_type == "gaussian":
        mean = 0
        sigma = intensity * 100
        gauss = np.random.normal(mean, sigma, noisy.shape)
        noisy = np.clip(noisy.astype(np.float64) + gauss, 0, 255).astype(np.uint8)
    elif noise_type == "salt_pepper":
        num_salt = np.ceil(intensity * noisy.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy.shape[:2]]
        noisy[tuple(coords)] = 255
        num_pepper = np.ceil(intensity * noisy.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy.shape[:2]]
        noisy[tuple(coords)] = 0
    elif noise_type == "speckle":
        gauss = np.random.normal(0, intensity, noisy.shape)
        noisy = np.clip(noisy.astype(np.float64) + noisy.astype(np.float64) * gauss, 0, 255).astype(np.uint8)
    return noisy

def calculate_metrics(img1_rgb, img2_rgb):
    g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    
    mse = np.mean((g1 - g2) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
    # SSIM approximation
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim = float(np.mean(ssim_map))
    
    return round(psnr, 2), round(ssim, 4), round(mse, 2)

# ----------------- SIDEBAR: GLOBAL CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("VisionLab AI")
    st.markdown("**Praktikum 04: Filtering, Konvolusi, & Deteksi Tepi**")
    st.divider()
    
    st.subheader("📁 Pilih Citra Masukan")
    input_source = st.radio("Sumber Citra:", ["Preset Sampel Laboratorium", "Unggah Gambar Sendiri (Upload)"])
    
    sample_files = {
        "Jalan Raya (Lane Markings)": "road_lane.jpg",
        "Dokumen Miring (Invoice / Kuitansi)": "document.jpg",
        "Pelat Nomor Kendaraan (ANPR)": "license_plate.jpg",
        "Citra Medis Rontgen (X-Ray Bone)": "xray_bone.jpg",
        "Koin & Objek Melingkar": "coins.jpg",
        "Papan Sirkuit (PCB Traces)": "circuit_board.jpg"
    }
    
    if input_source == "Preset Sampel Laboratorium":
        selected_sample_name = st.selectbox("Pilih Sampel:", list(sample_files.keys()))
        current_img = load_sample_image(sample_files[selected_sample_name])
    else:
        uploaded_file = st.file_uploader("Unggah file JPG/PNG:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            current_img = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            current_img = load_sample_image("road_lane.jpg")
            
    # Resize if too large
    max_dim = 640
    h, w = current_img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        current_img = cv2.resize(current_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
    st.caption(f"Dimensi Citra: {current_img.shape[1]} × {current_img.shape[0]} px")
    st.divider()
    st.info("Laboratorium Komputer Visi & NLP 2026")

# ----------------- MAIN TABS NAVIGATION -----------------
tabs = st.tabs([
    "🔬 1. Image Filtering",
    "⚙️ 2. Konvolusi 2D & Probe",
    "⚡ 3. Edge Detection Studio",
    "🚀 4. Canny 5-Stage Pipeline",
    "🚗 5. Kasus Computer Vision",
    "📹 6. Live Camera Sandbox",
    "📝 7. Kuis & Evaluasi",
    "💻 8. Contoh Kode per Metode"
])

# ==============================================================================
# TAB 1: IMAGE FILTERING LAB
# ==============================================================================
with tabs[0]:
    st.markdown('<span class="module-badge">Modul 1: Spatial Domain Filtering</span>', unsafe_allow_html=True)
    st.header("Image Filtering (Linear & Non-Linear Filters)")
    st.write("Eksplorasi teknik penghalusan citra (*smoothing*), pengurangan derau (*noise reduction*), dan penajaman citra (*sharpening*).")
    
    col_ctrl, col_view = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("⚙️ Parameter Filter")
        filter_choice = st.selectbox(
            "Pilih Algoritma Filter:",
            ["Gaussian Blur", "Mean (Box) Filter", "Median Filter", "Bilateral Filter", "Sharpening", "Unsharp Masking", "Min Filter (Erosion)", "Max Filter (Dilation)"]
        )
        
        # Noise Sandbox
        st.markdown("---")
        st.markdown("**🧪 Uji Ketahanan Derau (Noise Sandbox)**")
        noise_type = st.radio("Injeksi Derau:", ["Tanpa Derau", "Salt & Pepper", "Gaussian Noise", "Speckle Noise"], horizontal=True)
        noise_map = {"Tanpa Derau": "none", "Salt & Pepper": "salt_pepper", "Gaussian Noise": "gaussian", "Speckle Noise": "speckle"}
        
        noise_intensity = 0.05
        if noise_type != "Tanpa Derau":
            noise_intensity = st.slider("Intensitas Derau:", 0.01, 0.30, 0.05, 0.01, format="%.2f")
            
        # Apply noise
        noisy_img = add_noise(current_img, noise_map[noise_type], noise_intensity) if noise_type != "Tanpa Derau" else current_img.copy()
        
        st.markdown("---")
        # Dynamic Parameters
        ksize = st.slider("Ukuran Kernel (K × K):", 1, 31, 5, step=2)
        
        sigma = 1.5
        sigma_color, sigma_space = 75, 75
        
        if filter_choice in ["Gaussian Blur", "Unsharp Masking"]:
            sigma = st.slider("Standar Deviasi (σ):", 0.1, 10.0, 1.5, 0.1)
        elif filter_choice == "Bilateral Filter":
            sigma_color = st.slider("Sigma Color (σ_color):", 10, 250, 75, 5)
            sigma_space = st.slider("Sigma Space (σ_space):", 10, 250, 75, 5)
            
    # Compute Filter
    if filter_choice == "Gaussian Blur":
        processed_img = cv2.GaussianBlur(noisy_img, (ksize, ksize), sigma)
        py_code = f"processed = cv2.GaussianBlur(img, ({ksize}, {ksize}), sigmaX={sigma})"
    elif filter_choice == "Mean (Box) Filter":
        processed_img = cv2.blur(noisy_img, (ksize, ksize))
        py_code = f"processed = cv2.blur(img, ({ksize}, {ksize}))"
    elif filter_choice == "Median Filter":
        processed_img = cv2.medianBlur(noisy_img, ksize)
        py_code = f"processed = cv2.medianBlur(img, {ksize})"
    elif filter_choice == "Bilateral Filter":
        processed_img = cv2.bilateralFilter(noisy_img, d=ksize, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        py_code = f"processed = cv2.bilateralFilter(img, d={ksize}, sigmaColor={sigma_color}, sigmaSpace={sigma_space})"
    elif filter_choice == "Sharpening":
        kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        processed_img = cv2.filter2D(noisy_img, -1, kernel_sharpen)
        py_code = "kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)\nprocessed = cv2.filter2D(img, -1, kernel)"
    elif filter_choice == "Unsharp Masking":
        gauss = cv2.GaussianBlur(noisy_img, (ksize, ksize), sigma)
        processed_img = cv2.addWeighted(noisy_img, 1.5, gauss, -0.5, 0)
        py_code = f"gauss = cv2.GaussianBlur(img, ({ksize}, {ksize}), {sigma})\nprocessed = cv2.addWeighted(img, 1.5, gauss, -0.5, 0)"
    elif filter_choice == "Min Filter (Erosion)":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        processed_img = cv2.erode(noisy_img, kernel)
        py_code = f"kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({ksize}, {ksize}))\nprocessed = cv2.erode(img, kernel)"
    elif filter_choice == "Max Filter (Dilation)":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        processed_img = cv2.dilate(noisy_img, kernel)
        py_code = f"kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({ksize}, {ksize}))\nprocessed = cv2.dilate(img, kernel)"

    psnr_val, ssim_val, mse_val = calculate_metrics(current_img, processed_img)

    with col_view:
        # Metrics Row
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("PSNR (Peak Signal-to-Noise)", f"{psnr_val} dB")
        mcol2.metric("SSIM (Structural Similarity)", f"{ssim_val}")
        mcol3.metric("MSE (Mean Squared Error)", f"{mse_val}")
        
        # Image Comparison
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(noisy_img, caption="Citra Masukan / Derau", use_column_width=True)
        with img_col2:
            st.image(processed_img, caption=f"Hasil Filter: {filter_choice}", use_column_width=True)
            
        # Histogram
        st.subheader("📊 Perbandingan Histogram Intensitas Grayscale")
        fig, ax = plt.subplots(figsize=(8, 2.2))
        g_in = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2GRAY)
        g_out = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
        ax.hist(g_in.ravel(), bins=256, range=[0, 256], color='#0284c7', alpha=0.5, label='Citra Masukan')
        ax.hist(g_out.ravel(), bins=256, range=[0, 256], color='#059669', alpha=0.6, histtype='step', linewidth=1.8, label='Hasil Filter')
        ax.set_xlim([0, 256])
        ax.legend(loc='upper right', fontsize=8)
        ax.set_facecolor('#ffffff')
        fig.patch.set_facecolor('#f8fafc')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("**Python OpenCV Snippet:**")
        st.code(py_code, language="python")

# ==============================================================================
# TAB 2: 2D CONVOLUTION & PIXEL PROBE
# ==============================================================================
with tabs[1]:
    st.markdown('<span class="module-badge">Modul 2: 2D Convolution Arithmetic Engine</span>', unsafe_allow_html=True)
    st.header("2D Convolution Studio & Pixel Math Probe")
    st.markdown(r"Konvolusi spasial diskrit: $g(x,y) = \sum_{i=-a}^{a} \sum_{j=-b}^{b} f(x-i, y-j) K(i,j)$")
    
    conv_col1, conv_col2 = st.columns([1, 1])
    
    with conv_col1:
        st.subheader("⚙️ Konfigurasi Kernel")
        preset_name = st.selectbox(
            "Pilih Preset Kernel Populer:",
            ["Sharpen (3×3)", "Box Blur (3×3)", "Gaussian Blur (3×3)", "Laplacian (3×3)", "Sobel Horizontal (Gx)", "Sobel Vertikal (Gy)", "Emboss (3×3)", "Ridge Detection (3×3)", "Identity (Pass)"]
        )
        
        presets = {
            "Sharpen (3×3)": ([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 1.0, 0.0),
            "Box Blur (3×3)": ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 9.0, 0.0),
            "Gaussian Blur (3×3)": ([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 16.0, 0.0),
            "Laplacian (3×3)": ([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 1.0, 128.0),
            "Sobel Horizontal (Gx)": ([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 1.0, 128.0),
            "Sobel Vertikal (Gy)": ([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 1.0, 128.0),
            "Emboss (3×3)": ([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], 1.0, 128.0),
            "Ridge Detection (3×3)": ([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 1.0, 0.0),
            "Identity (Pass)": ([[0, 0, 0], [0, 1, 0], [0, 0, 0]], 1.0, 0.0)
        }
        
        default_matrix, def_div, def_bias = presets[preset_name]
        
        st.write("Edit Nilai Matriks Kernel (3×3):")
        k_df = pd.DataFrame(default_matrix, columns=["Kolom 0", "Kolom 1", "Kolom 2"])
        edited_df = st.data_editor(k_df, use_container_width=True)
        custom_kernel = edited_df.values.astype(np.float32)
        
        c_div_col, c_bias_col = st.columns(2)
        with c_div_col:
            divisor = st.number_input("Pembagi (Divisor / Normalisasi):", value=float(def_div), min_value=0.001)
        with c_bias_col:
            bias = st.number_input("Bias / Offset (+):", value=float(def_bias))
            
        border_mode_str = st.selectbox("Padding Border Mode:", ["Replicate", "Reflect", "Constant (Zero)", "Wrap"])
        border_map = {
            "Replicate": cv2.BORDER_REPLICATE,
            "Reflect": cv2.BORDER_REFLECT,
            "Constant (Zero)": cv2.BORDER_CONSTANT,
            "Wrap": cv2.BORDER_WRAP
        }
        
        # Apply Convolution
        norm_kernel = custom_kernel / divisor if divisor != 0 else custom_kernel
        conv_res = cv2.filter2D(current_img, -1, norm_kernel, borderType=border_map[border_mode_str])
        if bias != 0:
            conv_res = np.clip(conv_res.astype(np.float32) + bias, 0, 255).astype(np.uint8)

    with conv_col2:
        st.subheader("🖼️ Hasil Citra Konvolusi")
        st.image(conv_res, caption="Hasil Konvolusi 2D", use_column_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Interactive Pixel Math Probe")
        st.write("Pilih koordinat titik pixel untuk melihat rincian kalkulasi matematika sub-matriks:")
        
        h_img, w_img = current_img.shape[:2]
        pcol_x, pcol_y = st.columns(2)
        probe_x = pcol_x.slider("Koordinat X:", 1, w_img - 2, w_img // 2)
        probe_y = pcol_y.slider("Koordinat Y:", 1, h_img - 2, h_img // 2)
        
        # Extract 3x3 patch (Grayscale)
        gray_orig = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
        patch = gray_orig[probe_y-1:probe_y+2, probe_x-1:probe_x+2]
        
        # Calculations
        products = patch * custom_kernel
        sum_products = np.sum(products)
        divided_sum = sum_products / divisor
        final_pixel = int(np.clip(divided_sum + bias, 0, 255))
        
        pr_col1, pr_col2, pr_col3 = st.columns(3)
        with pr_col1:
            st.caption("1. Patch Pixel f(x,y)")
            st.dataframe(pd.DataFrame(patch), use_container_width=True)
        with pr_col2:
            st.caption("2. Kernel K")
            st.dataframe(pd.DataFrame(custom_kernel), use_container_width=True)
        with pr_col3:
            st.caption("3. Hasil Kali (P ⊙ K)")
            st.dataframe(pd.DataFrame(products), use_container_width=True)
            
        st.success(f"**Kalkulasi Titik ({probe_x}, {probe_y}):** ∑ Produk = {sum_products:.1f} → / {divisor} = {divided_sum:.1f} → + {bias} = **{final_pixel}** (Intensitas Output)")

# ==============================================================================
# TAB 3: EDGE DETECTION STUDIO
# ==============================================================================
with tabs[2]:
    st.markdown('<span class="module-badge">Modul 3: First & Second Order Derivatives</span>', unsafe_allow_html=True)
    st.header("Edge Detection Studio")
    st.write("Analisis komparatif operator turunan orde pertama (Sobel, Prewitt, Scharr, Roberts) dan orde kedua (Laplacian, LoG).")
    
    edge_ctrl, edge_view = st.columns([1, 2])
    
    with edge_ctrl:
        st.subheader("⚙️ Pilihan Operator")
        edge_op = st.selectbox(
            "Pilih Operator Deteksi Tepi:",
            ["Sobel Magnitude (√(Gx² + Gy²))", "Sobel Horizontal (Gx - Tepi Vertikal)", "Sobel Vertikal (Gy - Tepi Horizontal)", "Prewitt Magnitude", "Scharr Operator (Presisi Tinggi)", "Laplacian (Orde-2 ∇²f)", "Laplacian of Gaussian (LoG)", "Roberts Cross (2×2)"]
        )
        
        pre_blur = st.slider("Pre-Smoothing Gaussian Blur (K × K):", 1, 15, 3, step=2)
        ksize_edge = st.slider("Aperture / Orde Kernel Sobel/Laplacian:", 1, 7, 3, step=2)
        scale_edge = st.slider("Faktor Skala Pengali (Scale):", 0.2, 5.0, 1.0, 0.1)
        thresh_edge = st.slider("Binarisasi Ambang Batas (Thresholding):", 0, 255, 0)

    # Process Edges
    gray_src = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
    if pre_blur > 1:
        gray_src = cv2.GaussianBlur(gray_src, (pre_blur, pre_blur), 0)
        
    angle_vis = None
    if "Sobel Horizontal" in edge_op:
        gx = cv2.Sobel(gray_src, cv2.CV_64F, 1, 0, ksize=ksize_edge, scale=scale_edge)
        edge_result = cv2.convertScaleAbs(gx)
    elif "Sobel Vertikal" in edge_op:
        gy = cv2.Sobel(gray_src, cv2.CV_64F, 0, 1, ksize=ksize_edge, scale=scale_edge)
        edge_result = cv2.convertScaleAbs(gy)
    elif "Sobel Magnitude" in edge_op:
        gx = cv2.Sobel(gray_src, cv2.CV_64F, 1, 0, ksize=ksize_edge, scale=scale_edge)
        gy = cv2.Sobel(gray_src, cv2.CV_64F, 0, 1, ksize=ksize_edge, scale=scale_edge)
        mag = np.sqrt(gx**2 + gy**2)
        edge_result = cv2.convertScaleAbs(mag)
        
        # Angle Map
        angles = np.mod(np.arctan2(gy, gx) * (180.0 / np.pi), 180).astype(np.uint8)
        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hsv = np.zeros((gray_src.shape[0], gray_src.shape[1], 3), dtype=np.uint8)
        hsv[:, :, 0] = angles
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = norm_mag
        angle_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    elif "Prewitt" in edge_op:
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        gx = cv2.filter2D(gray_src, cv2.CV_64F, kx) * scale_edge
        gy = cv2.filter2D(gray_src, cv2.CV_64F, ky) * scale_edge
        edge_result = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))
    elif "Scharr" in edge_op:
        gx = cv2.Scharr(gray_src, cv2.CV_64F, 1, 0, scale=scale_edge)
        gy = cv2.Scharr(gray_src, cv2.CV_64F, 0, 1, scale=scale_edge)
        edge_result = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))
    elif "Laplacian of Gaussian" in edge_op:
        blurred = cv2.GaussianBlur(gray_src, (ksize_edge, ksize_edge), 1.4)
        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize_edge, scale=scale_edge)
        edge_result = cv2.convertScaleAbs(lap)
    elif "Laplacian" in edge_op:
        lap = cv2.Laplacian(gray_src, cv2.CV_64F, ksize=ksize_edge, scale=scale_edge)
        edge_result = cv2.convertScaleAbs(lap)
    elif "Roberts" in edge_op:
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gx = cv2.filter2D(gray_src, cv2.CV_64F, kx) * scale_edge
        gy = cv2.filter2D(gray_src, cv2.CV_64F, ky) * scale_edge
        edge_result = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))

    if thresh_edge > 0:
        _, edge_result = cv2.threshold(edge_result, thresh_edge, 255, cv2.THRESH_BINARY)
        
    edge_density = (np.count_nonzero(edge_result > 50) / edge_result.size) * 100

    with edge_view:
        st.subheader("🖼️ Peta Tepi & Visualisasi Orientasi")
        if angle_vis is not None:
            ev_col1, ev_col2 = st.columns(2)
            with ev_col1:
                st.image(edge_result, caption=f"Peta Magnitudo Tepi (Kerapatan: {edge_density:.2f}%)", use_column_width=True)
            with ev_col2:
                st.image(angle_vis, caption="Peta Arah Gradien (θ - False Color HSV)", use_column_width=True)
                st.caption("🔴 Merah: Horizontal (0°/180°) | 🟢 Hijau: Vertikal (90°) | 🔵 Biru: Diagonal")
        else:
            st.image(edge_result, caption=f"Hasil Deteksi Tepi (Kerapatan: {edge_density:.2f}%)", use_column_width=True)

# ==============================================================================
# TAB 4: CANNY 5-STAGE PIPELINE EXPLORER
# ==============================================================================
with tabs[3]:
    st.markdown('<span class="module-badge">Modul 4: Optimal Edge Detector (Canny 1986)</span>', unsafe_allow_html=True)
    st.header("Canny Edge Detector 5-Stage Deep Dive")
    st.write("Eksplorasi mendalam 5 tahapan berurutan algoritma Canny.")
    
    can_col1, can_col2, can_col3, can_col4 = st.columns(4)
    low_t = can_col1.slider("Low Threshold (T_low):", 5, 250, 50)
    high_t = can_col2.slider("High Threshold (T_high):", 10, 255, 150)
    g_k = can_col3.slider("Gaussian Kernel (K × K):", 3, 15, 5, step=2)
    g_sig = can_col4.slider("Gaussian σ:", 0.5, 5.0, 1.4, 0.1)
    
    if low_t > high_t:
        high_t = low_t
        
    # Stage 1: Gray
    s1_gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
    # Stage 2: Gaussian Blur
    s2_blur = cv2.GaussianBlur(s1_gray, (g_k, g_k), g_sig)
    # Stage 3: Sobel Gradient
    gx = cv2.Sobel(s2_blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(s2_blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag_norm = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else mag.astype(np.uint8)
    theta = np.arctan2(gy, gx) * 180. / np.pi
    theta[theta < 0] += 180
    
    # Stage 4: NMS
    M, N = s2_blur.shape
    nms = np.zeros((M, N), dtype=np.uint8)
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q, r = 255, 255
            ang = theta[i, j]
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q, r = mag[i, j+1], mag[i, j-1]
            elif (22.5 <= ang < 67.5):
                q, r = mag[i+1, j-1], mag[i-1, j+1]
            elif (67.5 <= ang < 112.5):
                q, r = mag[i+1, j], mag[i-1, j]
            elif (112.5 <= ang < 157.5):
                q, r = mag[i-1, j-1], mag[i+1, j+1]
            if (mag[i, j] >= q) and (mag[i, j] >= r):
                nms[i, j] = int(mag_norm[i, j])
                
    # Stage 5: Dual Threshold & Final Canny
    s5_canny = cv2.Canny(s2_blur, low_t, high_t, L2gradient=True)
    
    # Visualization Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("**Tahap 1: Grayscale**")
        st.image(s1_gray, use_column_width=True)
        st.caption("Konversi ke 1 kanal intensitas.")
    with c2:
        st.markdown("**Tahap 2: Gaussian**")
        st.image(s2_blur, use_column_width=True)
        st.caption("Peredaman noise.")
    with c3:
        st.markdown("**Tahap 3: Sobel Mag**")
        st.image(mag_norm, use_column_width=True)
        st.caption("Kalkulasi gradien & sudut.")
    with c4:
        st.markdown("**Tahap 4: NMS Thinning**")
        st.image(nms, use_column_width=True)
        st.caption("Penipisan kontur tepi (1 px).")
    with c5:
        st.markdown("**Tahap 5: Hysteresis**")
        st.image(s5_canny, use_column_width=True)
        st.caption("Hasil akhir Canny.")

# ==============================================================================
# TAB 5: COMPUTER VISION CASE STUDIES
# ==============================================================================
with tabs[4]:
    st.markdown('<span class="module-badge">Modul 5: Real-World Applications</span>', unsafe_allow_html=True)
    st.header("Contoh Penggunaan dalam Computer Vision Modern")
    
    case_choice = st.radio(
        "Pilih Studi Kasus:",
        [
            "1. Lane Detection (Deteksi Jalur Jalan Raya - Autonomous Driving)",
            "2. Document Scanner & Perspective De-skewing",
            "3. License Plate Feature Localization (ANPR)",
            "4. Medical X-Ray Bone Detail Enhancement"
        ],
        horizontal=True
    )
    
    if "Lane Detection" in case_choice:
        lane_img = load_sample_image("road_lane.jpg")
        h, w = lane_img.shape[:2]
        gray = cv2.cvtColor(lane_img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        
        mask = np.zeros_like(canny)
        roi_pts = np.array([[(0, h), (w * 0.42, h * 0.42), (w * 0.58, h * 0.42), (w, h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_pts, 255)
        masked_canny = cv2.bitwise_and(canny, mask)
        
        lines = cv2.HoughLinesP(masked_canny, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=100)
        overlay = lane_img.copy()
        if lines is not None:
            for line in lines:
                coords = line.ravel()
                if len(coords) >= 4:
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4, cv2.LINE_AA)
                
        st.subheader("🚗 Pipeline Deteksi Marka Jalan")
        k1, k2, k3, k4 = st.columns(4)
        k1.image(lane_img, caption="1. Citra Asli", use_column_width=True)
        k2.image(canny, caption="2. Canny Edges", use_column_width=True)
        k3.image(masked_canny, caption="3. Masked ROI Edges", use_column_width=True)
        k4.image(overlay, caption="4. Hough Line Overlay", use_column_width=True)
        
    elif "Document Scanner" in case_choice:
        doc_img = load_sample_image("document.jpg")
        gray = cv2.cvtColor(doc_img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 75, 200)
        
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        doc_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break
                
        vis = doc_img.copy()
        warped_res = doc_img.copy()
        if doc_contour is not None:
            cv2.drawContours(vis, [doc_contour], -1, (0, 255, 0), 3)
            pts = doc_contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            (tl, tr, br, bl) = rect
            maxWidth = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
            maxHeight = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped_res = cv2.warpPerspective(doc_img, M, (maxWidth, maxHeight))
            
        st.subheader("📄 Pipeline Scanner Dokumen & Homografi Perspektif")
        d1, d2, d3 = st.columns(3)
        d1.image(doc_img, caption="1. Foto Dokumen Miring", use_column_width=True)
        d2.image(vis, caption="2. Deteksi Kontur 4 Sudut", use_column_width=True)
        d3.image(warped_res, caption="3. Hasil Pelurusan (De-skew)", use_column_width=True)
        
    elif "License Plate" in case_choice:
        plate_img = load_sample_image("license_plate.jpg")
        gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)))
        sobel_x = cv2.convertScaleAbs(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3))
        _, thresh = cv2.threshold(cv2.GaussianBlur(sobel_x, (5, 5), 0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
        
        vis = plate_img.copy()
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 2.0 <= (w / float(h)) <= 6.0 and w > 80:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
        st.subheader("🚘 Pipeline Lokalisasi Pelat Nomor Kendaraan")
        p1, p2, p3, p4 = st.columns(4)
        p1.image(plate_img, caption="1. Citra Bumper Mobil", use_column_width=True)
        p2.image(blackhat, caption="2. Morfologi Blackhat", use_column_width=True)
        p3.image(sobel_x, caption="3. Sobel Vertical Edge", use_column_width=True)
        p4.image(vis, caption="4. Bounding Box Plat", use_column_width=True)
        
    elif "Medical X-Ray" in case_choice:
        xray_img = load_sample_image("xray_bone.jpg")
        gray = cv2.cvtColor(xray_img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        lap = cv2.convertScaleAbs(cv2.Laplacian(clahe, cv2.CV_64F, ksize=3))
        sharp = cv2.addWeighted(clahe, 1.3, lap, -0.3, 0)
        
        st.subheader("🩻 Pipeline Peningkatan Detail Citra Medis Rontgen")
        x1, x2, x3, x4 = st.columns(4)
        x1.image(gray, caption="1. Citra Asli X-Ray", use_column_width=True)
        x2.image(clahe, caption="2. CLAHE Contrast", use_column_width=True)
        x3.image(lap, caption="3. Laplacian Edge Map", use_column_width=True)
        x4.image(sharp, caption="4. Sharpened High-Boost", use_column_width=True)

# ==============================================================================
# TAB 6: LIVE CAMERA SANDBOX
# ==============================================================================
with tabs[5]:
    st.markdown('<span class="module-badge">Modul 6: WebCam Snapshot Studio</span>', unsafe_allow_html=True)
    st.header("Real-time Webcam Snapshot Processing")
    st.write("Ambil foto dari webcam komputer Anda untuk menguji algoritma filter secara instan.")
    
    cam_picture = st.camera_input("Ambil Foto dari Kamera:")
    if cam_picture is not None:
        bytes_data = cam_picture.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        filter_mode = st.selectbox(
            "Pilih Mode Filter Kamera:",
            ["Sobel Glowing Edges", "Canny Edges", "Inverted Pencil Sketch", "Cartoonifier Effect", "Gaussian Blur", "Sharpen"]
        )
        
        if filter_mode == "Sobel Glowing Edges":
            gray = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2) * 1.5)
            res_cam = np.zeros_like(cv_rgb)
            res_cam[:, :, 1] = mag # Green/Cyan glow
            res_cam[:, :, 2] = mag
        elif filter_mode == "Canny Edges":
            res_cam = cv2.Canny(cv_rgb, 50, 150)
        elif filter_mode == "Inverted Pencil Sketch":
            gray = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2GRAY)
            inv_blur = cv2.GaussianBlur(255 - gray, (21, 21), 0)
            res_cam = cv2.divide(gray, 255 - inv_blur, scale=256)
        elif filter_mode == "Cartoonifier Effect":
            gray = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(cv_rgb, 9, 250, 250)
            res_cam = cv2.bitwise_and(color, color, mask=edges)
        elif filter_mode == "Gaussian Blur":
            res_cam = cv2.GaussianBlur(cv_rgb, (15, 15), 0)
        elif filter_mode == "Sharpen":
            kernel_s = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            res_cam = cv2.filter2D(cv_rgb, -1, kernel_s)
            
        c_v1, c_v2 = st.columns(2)
        c_v1.image(cv_rgb, caption="Foto Asli", use_column_width=True)
        c_v2.image(res_cam, caption=f"Hasil: {filter_mode}", use_column_width=True)

# ==============================================================================
# TAB 7: QUIZ & THEORY EVALUATION
# ==============================================================================
with tabs[6]:
    st.markdown('<span class="module-badge">Modul 7: Evaluasi Kompetensi</span>', unsafe_allow_html=True)
    st.header("Evaluasi Pemahaman & Rangkuman Teori Praktikum")
    
    quiz_col1, quiz_col2 = st.columns([1, 1])
    
    with quiz_col1:
        st.subheader("📝 Kuis Pemahaman Mahasiswa")
        
        q1 = st.radio(
            "1. Mengapa kernel Gaussian Blur lebih disukai daripada Mean Filter dalam menghaluskan citra sebelum deteksi tepi?",
            [
                "A. Gaussian Blur memberikan bobot lebih tinggi pada pixel tengah sehingga mempertahankan transisi tepi secara natural",
                "B. Gaussian Blur selalu menghasilkan gambar biner hitam-putih",
                "C. Mean filter membutuhkan komputasi GPU khusus"
            ],
            key="q1"
        )
        
        q2 = st.radio(
            "2. Filter manakah yang paling ampuh membersihkan derau bintik 'Salt & Pepper' tanpa merusak ketajaman batas tepi?",
            [
                "A. Mean / Box Filter",
                "B. Median Filter (Non-Linear)",
                "C. Laplacian Filter"
            ],
            key="q2"
        )
        
        q3 = st.radio(
            "3. Jika sebuah kernel konvolusi memiliki jumlah seluruh elemennya sama dengan 0 (nol), fungsi kernel tersebut adalah:",
            [
                "A. Pencerahan citra (Brightness)",
                "B. Deteksi Tepi / Ekstraksi Fitur Turunan (Gradient / Edges)",
                "C. Penghalusan citra (Blurring)"
            ],
            key="q3"
        )
        
        q4 = st.radio(
            "4. Apa tujuan tahap 'Non-Maximum Suppression' (NMS) pada Canny Edge Detector?",
            [
                "A. Menipiskan kontur tepi yang tebal menjadi setebal 1 pixel",
                "B. Mengubah format warna citra ke CMYK",
                "C. Menambahkan derau acak"
            ],
            key="q4"
        )
        
        q5 = st.radio(
            "5. Apa keunggulan utama Bilateral Filter dibandingkan Gaussian Blur standar?",
            [
                "A. Berjalan lebih cepat dari semua filter",
                "B. Menghaluskan area datar namun tetap mempertahankan ketajaman tepi (Edge-Preserving)",
                "C. Hanya bekerja pada citra 1-bit"
            ],
            key="q5"
        )
        
        if st.button("Kumpulkan & Evaluasi Kuis", type="primary"):
            score = 0
            if q1.startswith("A."): score += 1
            if q2.startswith("B."): score += 1
            if q3.startswith("B."): score += 1
            if q4.startswith("A."): score += 1
            if q5.startswith("B."): score += 1
            
            st.markdown("---")
            if score == 5:
                st.balloons()
                st.success(f"🎉 **Skor Sempurna: {score}/5 (100%)**! Pemahaman konsep praktikum Anda sangat memuaskan.")
            else:
                st.info(f"📊 **Skor Anda: {score}/5 ({score*20}%)**. Pelajari kembali rangkuman teori di panel sebelah kanan.")
                
    with quiz_col2:
        st.subheader("📚 Rangkuman Teori & Formula Matematis")
        
        with st.expander("1. Operasi Konvolusi 2D", expanded=True):
            st.latex(r"g(x,y) = \sum_{i=-a}^{a} \sum_{j=-b}^{b} f(x-i, y-j) K(i,j)")
            st.write("Operasi *sliding window* perkalian element-wise antara matriks citra dan bobot kernel spasial.")
            
        with st.expander("2. Filter Gaussian 2D"):
            st.latex(r"G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}")
            st.write("Distribusi normal multivariat 2 dimensi untuk peredaman derau Gaussian secara natural.")
            
        with st.expander("3. Operator Turunan Sobel"):
            st.latex(r"G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I")
            st.latex(r"|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)")
            
        with st.expander("4. Operator Laplacian (Orde-2)"):
            st.latex(r"\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix} * f")

# ==============================================================================
# TAB 8: CONTOH KODE PER METODE (PYTHON OPENCV + MATPLOTLIB)
# ==============================================================================
with tabs[7]:
    st.markdown('<span class="module-badge">Modul 8: Template Kode Praktikum & Matplotlib</span>', unsafe_allow_html=True)
    st.header("📚 Contoh Kode Pemrograman per Metode")
    st.write("Gunakan template kode Python OpenCV + Matplotlib di bawah ini sebagai referensi pembuatan skrip praktikum, laporan, atau tugas akhir.")
    
    method_select = st.selectbox(
        "Pilih Metode Pengolahan Citra:",
        [
            "1. Gaussian Blur (cv2.GaussianBlur)",
            "2. Mean / Box Filter (cv2.blur)",
            "3. Median Filter (cv2.medianBlur)",
            "4. Bilateral Filter (cv2.bilateralFilter)",
            "5. Konvolusi 2D Kustom (cv2.filter2D - Sharpening & Emboss)",
            "6. Deteksi Tepi Sobel (cv2.Sobel - Gx, Gy, & Magnitude)",
            "7. Deteksi Tepi Laplacian (cv2.Laplacian)",
            "8. Canny Edge Detection (cv2.Canny)",
            "9. Unsharp Masking & Penajaman Citra (cv2.addWeighted)",
            "10. Padding Citra / Border Extrapolation (cv2.copyMakeBorder)"
        ]
    )
    
    st.markdown("---")
    
    # Generate live interactive visualization & snippet based on method
    if "1. Gaussian Blur" in method_select:
        # 1. Gaussian Blur
        blurred_5x5 = cv2.GaussianBlur(current_img, (5, 5), sigmaX=1.0)
        blurred_15x15 = cv2.GaussianBlur(current_img, (15, 15), sigmaX=3.0)
        
        # Matplotlib Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
        axes[0].imshow(current_img)
        axes[0].axis('off')
        
        axes[1].set_title("Gaussian Blur (5x5, σ=1.0)", fontsize=11, fontweight='bold')
        axes[1].imshow(blurred_5x5)
        axes[1].axis('off')
        
        axes[2].set_title("Gaussian Blur (15x15, σ=3.0)", fontsize=11, fontweight='bold')
        axes[2].imshow(blurred_15x15)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        code_str = """import cv2
import matplotlib.pyplot as plt

# 1. Baca gambar
image = cv2.imread('input.jpg')

# 2. Terapkan Gaussian Blur
# Parameter: (input_image, kernel_size, sigmaX)
# kernel_size harus berupa bilangan ganjil, misal (5, 5) atau (15, 15)
blurred_5x5 = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
blurred_15x15 = cv2.GaussianBlur(image, (15, 15), sigmaX=3.0)

# 3. Tampilkan Hasil dengan Matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Gambar Asli")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gaussian Blur (5x5, σ=1.0)")
plt.imshow(cv2.cvtColor(blurred_5x5, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Gaussian Blur (15x15, σ=3.0)")
plt.imshow(cv2.cvtColor(blurred_15x15, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
"""

    elif "2. Mean / Box Filter" in method_select:
        mean_5x5 = cv2.blur(current_img, (5, 5))
        mean_15x15 = cv2.blur(current_img, (15, 15))
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
        axes[0].imshow(current_img)
        axes[0].axis('off')
        
        axes[1].set_title("Mean Blur (5x5)", fontsize=11, fontweight='bold')
        axes[1].imshow(mean_5x5)
        axes[1].axis('off')
        
        axes[2].set_title("Mean Blur (15x15)", fontsize=11, fontweight='bold')
        axes[2].imshow(mean_15x15)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        code_str = """import cv2
import matplotlib.pyplot as plt

# 1. Baca gambar
image = cv2.imread('input.jpg')

# 2. Terapkan Mean / Box Filter (Rata-rata aritmatika tetangga)
# Parameter: (input_image, (ksize_width, ksize_height))
mean_5x5 = cv2.blur(image, (5, 5))
mean_15x15 = cv2.blur(image, (15, 15))

# 3. Tampilkan Hasil
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Gambar Asli")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Mean Blur (5x5)")
plt.imshow(cv2.cvtColor(mean_5x5, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Mean Blur (15x15)")
plt.imshow(cv2.cvtColor(mean_15x15, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
"""

    elif "3. Median Filter" in method_select:
        noisy_sp = add_noise(current_img, 'salt_pepper', 0.08)
        median_3 = cv2.medianBlur(noisy_sp, 3)
        median_7 = cv2.medianBlur(noisy_sp, 7)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Derau Salt & Pepper", fontsize=11, fontweight='bold')
        axes[0].imshow(noisy_sp)
        axes[0].axis('off')
        
        axes[1].set_title("Median Filter (k=3)", fontsize=11, fontweight='bold')
        axes[1].imshow(median_3)
        axes[1].axis('off')
        
        axes[2].set_title("Median Filter (k=7)", fontsize=11, fontweight='bold')
        axes[2].imshow(median_7)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        code_str = """import cv2
import matplotlib.pyplot as plt

# 1. Baca gambar (contoh dengan derau Salt & Pepper)
image = cv2.imread('input_noisy.jpg')

# 2. Terapkan Median Filter (Sangat efektif untuk Salt & Pepper Noise)
# Parameter: (input_image, ksize) -> ksize harus integer ganjil positif
median_3 = cv2.medianBlur(image, 3)
median_7 = cv2.medianBlur(image, 7)

# 3. Tampilkan Hasil
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Gambar Berderau (Salt & Pepper)")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Median Filter (k=3)")
plt.imshow(cv2.cvtColor(median_3, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Median Filter (k=7)")
plt.imshow(cv2.cvtColor(median_7, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
"""

    elif "4. Bilateral Filter" in method_select:
        bilateral_1 = cv2.bilateralFilter(current_img, d=9, sigmaColor=75, sigmaSpace=75)
        bilateral_2 = cv2.bilateralFilter(current_img, d=15, sigmaColor=150, sigmaSpace=150)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
        axes[0].imshow(current_img)
        axes[0].axis('off')
        
        axes[1].set_title("Bilateral (d=9, σ_c=75, σ_s=75)", fontsize=11, fontweight='bold')
        axes[1].imshow(bilateral_1)
        axes[1].axis('off')
        
        axes[2].set_title("Bilateral (d=15, σ_c=150, σ_s=150)", fontsize=11, fontweight='bold')
        axes[2].imshow(bilateral_2)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        code_str = """import cv2
import matplotlib.pyplot as plt

# 1. Baca gambar
image = cv2.imread('input.jpg')

# 2. Terapkan Bilateral Filter (Edge-Preserving Smoothing)
# Parameter: (input_image, diameter, sigmaColor, sigmaSpace)
# sigmaColor: toleransi perbedaan warna yang dibaurkan
# sigmaSpace: jangkauan piksel spasial tetangga
bilateral_1 = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
bilateral_2 = cv2.bilateralFilter(image, d=15, sigmaColor=150, sigmaSpace=150)

# 3. Tampilkan Hasil
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Gambar Asli")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Bilateral (d=9, σc=75, σs=75)")
plt.imshow(cv2.cvtColor(bilateral_1, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Bilateral (d=15, σc=150, σs=150)")
plt.imshow(cv2.cvtColor(bilateral_2, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
"""

    elif "5. Konvolusi 2D Kustom" in method_select:
        k_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        k_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        
        res_sharpen = cv2.filter2D(current_img, -1, k_sharpen)
        res_emboss = cv2.filter2D(current_img, -1, k_emboss) + 128
        res_emboss = np.clip(res_emboss, 0, 255).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
        axes[0].imshow(current_img)
        axes[0].axis('off')
        
        axes[1].set_title("Konvolusi Sharpening", fontsize=11, fontweight='bold')
        axes[1].imshow(res_sharpen)
        axes[1].axis('off')
        
        axes[2].set_title("Konvolusi 3D Emboss (+128)", fontsize=11, fontweight='bold')
        axes[2].imshow(res_emboss)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        code_str = """import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Baca gambar
image = cv2.imread('input.jpg')

# 2. Definisikan Matriks Kernel 2D Kustom
kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

kernel_emboss = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]
], dtype=np.float32)

# 3. Terapkan Konvolusi 2D Spasial (cv2.filter2D)
sharpened = cv2.filter2D(image, ddepth=-1, kernel=kernel_sharpen)
embossed = cv2.filter2D(image, ddepth=-1, kernel=kernel_emboss) + 128
embossed = np.clip(embossed, 0, 255).astype(np.uint8)

# 4. Tampilkan Hasil
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Gambar Asli")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Hasil Sharpening")
plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Hasil 3D Emboss")
plt.imshow(cv2.cvtColor(embossed, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
"""

    elif "6. Deteksi Tepi Sobel" in method_select:
        gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Sobel Horizontal (Gx - Vertikal)", fontsize=11, fontweight='bold')
        axes[0].imshow(cv2.convertScaleAbs(gx), cmap='gray')
        axes[0].axis('off')
        
        axes[1].set_title("Sobel Vertikal (Gy - Horizontal)", fontsize=11, fontweight='bold')
        axes[1].imshow(cv2.convertScaleAbs(gy), cmap='gray')
        axes[1].axis('off')
        
        axes[2].set_title("Sobel Magnitude: √(Gx² + Gy²)", fontsize=11, fontweight='bold')
        axes[2].imshow(sobel_mag, cmap='gray')
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        code_str = """import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Baca gambar & konversi ke Grayscale
image = cv2.imread('input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Hitung Gradien Turunan Pertama (Sobel Gx dan Gy)
# Parameter: (src, ddepth, dx, dy, ksize)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# 3. Hitung Magnitudo Gradien Total
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_mag = cv2.convertScaleAbs(magnitude)

# 4. Tampilkan Hasil
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Sobel Gx (Tepi Vertikal)")
plt.imshow(cv2.convertScaleAbs(sobel_x), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Sobel Gy (Tepi Horizontal)")
plt.imshow(cv2.convertScaleAbs(sobel_y), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Sobel Magnitude Total")
plt.imshow(sobel_mag, cmap='gray')
plt.axis('off')

plt.show()
"""

    elif "7. Deteksi Tepi Laplacian" in method_select:
        gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
        lap_raw = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
        
        # Laplacian of Gaussian (LoG)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        lap_log = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F, ksize=3))
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Gambar Asli (Grayscale)", fontsize=11, fontweight='bold')
        axes[0].imshow(gray, cmap='gray')
        axes[0].axis('off')
        
        axes[1].set_title("Laplacian (Standar Orde-2)", fontsize=11, fontweight='bold')
        axes[1].imshow(lap_raw, cmap='gray')
        axes[1].axis('off')
        
        axes[2].set_title("Laplacian of Gaussian (LoG)", fontsize=11, fontweight='bold')
        axes[2].imshow(lap_log, cmap='gray')
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        code_str = """import cv2
import matplotlib.pyplot as plt

# 1. Baca gambar & ubah ke Grayscale
image = cv2.imread('input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Laplacian Standar (Derivatif Orde-2)
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# 3. Laplacian of Gaussian (LoG - Meredam noise terlebih dahulu)
blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)
log_laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
log_abs = cv2.convertScaleAbs(log_laplacian)

# 4. Tampilkan Hasil
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Gambar Grayscale Asli")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Laplacian Standar")
plt.imshow(laplacian_abs, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Laplacian of Gaussian (LoG)")
plt.imshow(log_abs, cmap='gray')
plt.axis('off')

plt.show()
"""

    elif "8. Canny Edge Detection" in method_select:
        gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
        canny_low = cv2.Canny(gray, 50, 150)
        canny_high = cv2.Canny(gray, 120, 220)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
        axes[0].imshow(current_img)
        axes[0].axis('off')
        
        axes[1].set_title("Canny (50, 150)", fontsize=11, fontweight='bold')
        axes[1].imshow(canny_low, cmap='gray')
        axes[1].axis('off')
        
        axes[2].set_title("Canny (120, 220)", fontsize=11, fontweight='bold')
        axes[2].imshow(canny_high, cmap='gray')
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        code_str = """import cv2
import matplotlib.pyplot as plt

# 1. Baca gambar & ubah ke Grayscale
image = cv2.imread('input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Terapkan Canny Edge Detector (5-Tahap Optimal Edge Detector)
# Parameter: (src, threshold1_low, threshold2_high)
canny_low = cv2.Canny(gray, threshold1=50, threshold2=150)
canny_high = cv2.Canny(gray, threshold1=120, threshold2=220)

# 3. Tampilkan Hasil
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Gambar Asli")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Canny (T_low=50, T_high=150)")
plt.imshow(canny_low, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Canny (T_low=120, T_high=220)")
plt.imshow(canny_high, cmap='gray')
plt.axis('off')

plt.show()
"""

    elif "9. Unsharp Masking & Penajaman Citra" in method_select:
        gauss = cv2.GaussianBlur(current_img, (9, 9), 2.0)
        unsharp = cv2.addWeighted(current_img, 1.5, gauss, -0.5, 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Gambar Asli", fontsize=11, fontweight='bold')
        axes[0].imshow(current_img)
        axes[0].axis('off')
        
        axes[1].set_title("Gaussian Blur Komponen", fontsize=11, fontweight='bold')
        axes[1].imshow(gauss)
        axes[1].axis('off')
        
        axes[2].set_title("Hasil Unsharp Masking", fontsize=11, fontweight='bold')
        axes[2].imshow(unsharp)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        code_str = """import cv2
import matplotlib.pyplot as plt

# 1. Baca gambar
image = cv2.imread('input.jpg')

# 2. Buat komponen blur (low-pass)
blurred = cv2.GaussianBlur(image, (9, 9), sigmaX=2.0)

# 3. Unsharp Masking Formula: Enhanced = 1.5 * Original - 0.5 * Blurred
sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, gamma=0)

# 4. Tampilkan Hasil
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Gambar Asli")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Komponen Blur Gaussian")
plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Hasil Penajaman Unsharp Mask")
plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
"""

    elif "10. Padding Citra" in method_select:
        # Define border sizes
        top, bottom, left, right = 40, 40, 40, 40
        
        # 3 types of padding
        pad_constant = cv2.copyMakeBorder(current_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[37, 99, 235])
        pad_replicate = cv2.copyMakeBorder(current_img, top, bottom, left, right, cv2.BORDER_REPLICATE)
        pad_reflect = cv2.copyMakeBorder(current_img, top, bottom, left, right, cv2.BORDER_REFLECT)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Constant Border (Blue 40px)", fontsize=11, fontweight='bold')
        axes[0].imshow(pad_constant)
        axes[0].axis('off')
        
        axes[1].set_title("Replicate Padding (Duplikasi Tepi)", fontsize=11, fontweight='bold')
        axes[1].imshow(pad_replicate)
        axes[1].axis('off')
        
        axes[2].set_title("Reflect Padding (Cermin Tepi)", fontsize=11, fontweight='bold')
        axes[2].imshow(pad_reflect)
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        code_str = """import cv2
import matplotlib.pyplot as plt

# 1. Baca gambar
image = cv2.imread('input.jpg')

# 2. Definisikan ketebalan padding (atas, bawah, kiri, kanan dalam piksel)
top, bottom, left, right = 40, 40, 40, 40

# 3. Terapkan berbagai metode Padding (Border Extrapolation)
# A. BORDER_CONSTANT: Menambahkan border dengan warna konstan (misal: biru [255, 0, 0] atau hitam [0, 0, 0])
pad_constant = cv2.copyMakeBorder(
    image, top, bottom, left, right, 
    borderType=cv2.BORDER_CONSTANT, 
    value=[255, 0, 0] # Warna border BGR
)

# B. BORDER_REPLICATE: Menduplikasi piksel paling pinggir sepanjang batas
pad_replicate = cv2.copyMakeBorder(
    image, top, bottom, left, right, 
    borderType=cv2.BORDER_REPLICATE
)

# C. BORDER_REFLECT: Mencerminkan piksel di sekitar perbatasan citra
pad_reflect = cv2.copyMakeBorder(
    image, top, bottom, left, right, 
    borderType=cv2.BORDER_REFLECT
)

# 4. Tampilkan Hasil dengan Matplotlib
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.title("Constant Border (40px)")
plt.imshow(cv2.cvtColor(pad_constant, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Replicate Padding (Duplikasi)")
plt.imshow(cv2.cvtColor(pad_replicate, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Reflect Padding (Pencerminan)")
plt.imshow(cv2.cvtColor(pad_reflect, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
"""

    st.subheader("📋 Salin Skrip Python Lengkap (Siap Eksekusi)")
    st.code(code_str, language="python")
    
    st.download_button(
        label="💾 Download Skrip Python (.py)",
        data=code_str,
        file_name=f"praktikum04_{method_select[:15].lower().replace(' ', '_')}.py",
        mime="text/plain"
    )
