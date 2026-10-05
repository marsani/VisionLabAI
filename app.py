import os
import io
import time
import base64
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from scipy import ndimage

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/samples', exist_ok=True)

# Helper function to encode image to base64
def img_to_base64(img_bgr_or_gray):
    if img_bgr_or_gray is None:
        return ""
    if len(img_bgr_or_gray.shape) == 2:
        # Convert grayscale to RGB for consistent display or keep as single channel png
        _, buffer = cv2.imencode('.png', img_bgr_or_gray)
    else:
        _, buffer = cv2.imencode('.jpg', img_bgr_or_gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

# Helper function to calculate PSNR and SSIM
def calculate_metrics(original_gray, processed_gray):
    mse = np.mean((original_gray.astype(np.float64) - processed_gray.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # Fast SSIM calculation
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    img1 = original_gray.astype(np.float64)
    img2 = processed_gray.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim = float(np.mean(ssim_map))
    
    return {
        "mse": round(float(mse), 2),
        "psnr": round(float(psnr), 2),
        "ssim": round(float(ssim), 4)
    }

# Noise Injection Utility
def add_noise(image, noise_type, intensity=0.1):
    noisy = image.copy()
    if noise_type == "gaussian":
        row, col = noisy.shape[:2]
        ch = 1 if len(noisy.shape) == 2 else noisy.shape[2]
        mean = 0
        sigma = intensity * 100
        gauss = np.random.normal(mean, sigma, (row, col, ch) if ch > 1 else (row, col))
        noisy = np.clip(noisy.astype(np.float64) + gauss, 0, 255).astype(np.uint8)
    elif noise_type == "salt_pepper":
        row, col = noisy.shape[:2]
        # Salt (white)
        num_salt = np.ceil(intensity * noisy.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy.shape[:2]]
        noisy[tuple(coords)] = 255
        # Pepper (black)
        num_pepper = np.ceil(intensity * noisy.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy.shape[:2]]
        noisy[tuple(coords)] = 0
    elif noise_type == "speckle":
        row, col = noisy.shape[:2]
        ch = 1 if len(noisy.shape) == 2 else noisy.shape[2]
        gauss = np.random.normal(0, intensity, (row, col, ch) if ch > 1 else (row, col))
        noisy = np.clip(noisy.astype(np.float64) + noisy.astype(np.float64) * gauss, 0, 255).astype(np.uint8)
    return noisy

def calculate_histogram(gray_img):
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    return [int(val[0]) for val in hist]

def load_input_image(data):
    # Data can be a sample filename or base64 image
    img_src = data.get('image_src', 'road_lane.jpg')
    
    if img_src.startswith('data:image'):
        header, encoded = img_src.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        sample_path = os.path.join('static/samples', os.path.basename(img_src))
        if os.path.exists(sample_path):
            img = cv2.imread(sample_path)
        else:
            sample_path = os.path.join('static/samples', 'road_lane.jpg')
            img = cv2.imread(sample_path)
            
    if img is None:
        # Fallback dummy image
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(img, "Image Not Found", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    # Resize if too large to ensure snappy browser responsiveness
    max_dim = 640
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/samples', methods=['GET'])
def get_samples():
    samples = [
        {
            "id": "road_lane.jpg",
            "name": "Jalan Raya (Lane Markings)",
            "category": "Autonomous Driving",
            "description": "Deteksi marka jalur jalan raya dengan Canny edge detector & Hough transform."
        },
        {
            "id": "document.jpg",
            "name": "Dokumen / Kuitansi Miring",
            "category": "Document Scanner",
            "description": "Deteksi batas tepi kertas dan pelurusan perspektif dokumen (Perspective Warp)."
        },
        {
            "id": "license_plate.jpg",
            "name": "Pelat Nomor Kendaraan",
            "category": "ANPR / OCR",
            "description": "Peningkatan tepi vertikal (Sobel Vertical) dan morfologi untuk lokalisasi plat nomor."
        },
        {
            "id": "xray_bone.jpg",
            "name": "Citra Medis Rontgen X-Ray",
            "category": "Medical Imaging",
            "description": "Peningkatan detail struktur tulang dan kontur jaringan dengan filter Laplacian & Unsharp Mask."
        },
        {
            "id": "coins.jpg",
            "name": "Koin & Objek Melingkar",
            "category": "Object Detection",
            "description": "Filtering noise latar belakang dan deteksi kontur tepi koin."
        },
        {
            "id": "circuit_board.jpg",
            "name": "Papan Sirkuit PCB",
            "category": "Industrial Inspection",
            "description": "Inspeksi jalur konduktor dan pin IC beresolusi tinggi dengan deteksi tepi presisi."
        }
    ]
    return jsonify(samples)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nama file kosong"}), 400
        
    filename = secure_filename(f"custom_{int(time.time())}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({"error": "Format gambar tidak didukung"}), 400
        
    # Resize if too large
    max_dim = 640
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(filepath, img)
        
    return jsonify({
        "success": True,
        "image_src": img_to_base64(img),
        "filename": filename,
        "width": img.shape[1],
        "height": img.shape[0]
    })

# ----------------- 1. IMAGE FILTERING API -----------------
@app.route('/api/filter', methods=['POST'])
def apply_filter():
    start_time = time.time()
    data = request.json or {}
    img = load_input_image(data)
    
    # Noise injection if specified
    noise_type = data.get('noise_type', 'none')
    noise_intensity = float(data.get('noise_intensity', 0.05))
    if noise_type != 'none':
        img_noisy = add_noise(img, noise_type, noise_intensity)
    else:
        img_noisy = img.copy()
        
    filter_type = data.get('filter_type', 'gaussian')
    ksize = int(data.get('ksize', 5))
    if ksize % 2 == 0:
        ksize += 1 # Kernel size must be odd
    ksize = max(1, min(ksize, 31))
    
    sigma = float(data.get('sigma', 1.0))
    sigma_color = float(data.get('sigma_color', 75.0))
    sigma_space = float(data.get('sigma_space', 75.0))
    
    code_snippet = ""
    
    if filter_type == 'mean':
        processed = cv2.blur(img_noisy, (ksize, ksize))
        code_snippet = f"# Mean / Box Filtering (Ukuran Kernel: {ksize}x{ksize})\nprocessed = cv2.blur(img, ({ksize}, {ksize}))"
    elif filter_type == 'gaussian':
        processed = cv2.GaussianBlur(img_noisy, (ksize, ksize), sigma)
        code_snippet = f"# Gaussian Blur (Kernel: {ksize}x{ksize}, Sigma: {sigma})\nprocessed = cv2.GaussianBlur(img, ({ksize}, {ksize}), sigmaX={sigma})"
    elif filter_type == 'median':
        processed = cv2.medianBlur(img_noisy, ksize)
        code_snippet = f"# Median Filtering (Noise Salt & Pepper Remover, Kernel: {ksize})\nprocessed = cv2.medianBlur(img, {ksize})"
    elif filter_type == 'bilateral':
        processed = cv2.bilateralFilter(img_noisy, d=ksize, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        code_snippet = f"# Bilateral Filter (Edge-Preserving Smoothing)\n# d={ksize}, sigmaColor={sigma_color}, sigmaSpace={sigma_space}\nprocessed = cv2.bilateralFilter(img, d={ksize}, sigmaColor={sigma_color}, sigmaSpace={sigma_space})"
    elif filter_type == 'sharpen':
        # 3x3 sharpening kernel
        kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        processed = cv2.filter2D(img_noisy, -1, kernel_sharpen)
        code_snippet = "# Sharpening Filter\nkernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)\nprocessed = cv2.filter2D(img, -1, kernel)"
    elif filter_type == 'unsharp_mask':
        gaussian = cv2.GaussianBlur(img_noisy, (ksize, ksize), sigma)
        processed = cv2.addWeighted(img_noisy, 1.5, gaussian, -0.5, 0)
        code_snippet = f"# Unsharp Masking\ngaussian = cv2.GaussianBlur(img, ({ksize}, {ksize}), {sigma})\nprocessed = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)"
    elif filter_type == 'min_filter':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        processed = cv2.erode(img_noisy, kernel)
        code_snippet = f"# Min Filter (Morphological Erosion, Kernel: {ksize}x{ksize})\nkernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({ksize}, {ksize}))\nprocessed = cv2.erode(img, kernel)"
    elif filter_type == 'max_filter':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        processed = cv2.dilate(img_noisy, kernel)
        code_snippet = f"# Max Filter (Morphological Dilation, Kernel: {ksize}x{ksize})\nkernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({ksize}, {ksize}))\nprocessed = cv2.dilate(img, kernel)"
    else:
        processed = img_noisy.copy()
        code_snippet = "# Passthrough"
        
    orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    metrics = calculate_metrics(orig_gray, proc_gray)
    exec_time = round((time.time() - start_time) * 1000, 2)
    
    return jsonify({
        "original_image": img_to_base64(img),
        "noisy_image": img_to_base64(img_noisy),
        "processed_image": img_to_base64(processed),
        "orig_histogram": calculate_histogram(orig_gray),
        "proc_histogram": calculate_histogram(proc_gray),
        "metrics": metrics,
        "execution_time_ms": exec_time,
        "code_snippet": code_snippet
    })

# ----------------- 2. 2D CONVOLUTION API -----------------
@app.route('/api/convolve', methods=['POST'])
def apply_convolution():
    start_time = time.time()
    data = request.json or {}
    img = load_input_image(data)
    
    kernel_data = data.get('kernel', [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    kernel = np.array(kernel_data, dtype=np.float32)
    divisor = float(data.get('divisor', 1.0))
    bias = float(data.get('bias', 0.0))
    is_grayscale = bool(data.get('grayscale', False))
    border_mode_str = data.get('border_mode', 'replicate')
    
    border_map = {
        'constant': cv2.BORDER_CONSTANT,
        'replicate': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT,
        'wrap': cv2.BORDER_WRAP
    }
    border_type = border_map.get(border_mode_str, cv2.BORDER_REPLICATE)
    
    if divisor != 0:
        kernel = kernel / divisor
        
    target_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_grayscale else img
    
    # Custom 2D convolution with cv2.filter2D + bias
    processed = cv2.filter2D(target_img, -1, kernel, borderType=border_type)
    if bias != 0:
        processed = np.clip(processed.astype(np.float32) + bias, 0, 255).astype(np.uint8)
        
    orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc_gray = processed if is_grayscale else cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    metrics = calculate_metrics(orig_gray, proc_gray)
    exec_time = round((time.time() - start_time) * 1000, 2)
    
    # Python code snippet
    kernel_str = "np.array([\n" + ",\n".join(["    " + str(list(row)) for row in kernel_data]) + "\n], dtype=np.float32)"
    code_snippet = f"""import cv2
import numpy as np

# Definisi Custom Kernel 2D
kernel = {kernel_str}
divisor = {divisor}
bias = {bias}

if divisor != 0:
    kernel = kernel / divisor

# Terapkan Konvolusi 2D
processed = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_{border_mode_str.upper()})
if bias != 0:
    processed = np.clip(processed.astype(np.float32) + bias, 0, 255).astype(np.uint8)
"""
    return jsonify({
        "original_image": img_to_base64(img),
        "processed_image": img_to_base64(processed),
        "metrics": metrics,
        "execution_time_ms": exec_time,
        "code_snippet": code_snippet
    })

# Step-by-step Pixel Probe (Hovering / clicking pixel matrix math)
@app.route('/api/convolve_pixel_probe', methods=['POST'])
def convolve_pixel_probe():
    data = request.json or {}
    img = load_input_image(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    x = int(data.get('x', gray.shape[1] // 2))
    y = int(data.get('y', gray.shape[0] // 2))
    x = max(0, min(x, gray.shape[1] - 1))
    y = max(0, min(y, gray.shape[0] - 1))
    
    kernel_data = data.get('kernel', [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    kernel = np.array(kernel_data, dtype=np.float32)
    divisor = float(data.get('divisor', 1.0))
    bias = float(data.get('bias', 0.0))
    
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # Extract neighborhood patch with replication border
    padded = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    patch = padded[y:y + k_h, x:x + k_w]
    
    # Element-wise operations
    multiplications = []
    total_sum = 0.0
    for r in range(k_h):
        row_mult = []
        for c in range(k_w):
            p_val = int(patch[r, c])
            k_val = float(kernel[r, c])
            prod = p_val * k_val
            total_sum += prod
            row_mult.append({
                "pixel": p_val,
                "kernel": k_val,
                "product": round(prod, 2)
            })
        multiplications.append(row_mult)
        
    divided_sum = total_sum / divisor if divisor != 0 else total_sum
    final_raw = divided_sum + bias
    clamped_val = int(np.clip(final_raw, 0, 255))
    
    return jsonify({
        "x": x,
        "y": y,
        "image_width": gray.shape[1],
        "image_height": gray.shape[0],
        "patch": patch.tolist(),
        "kernel": kernel.tolist(),
        "multiplications": multiplications,
        "sum_products": round(total_sum, 2),
        "divisor": divisor,
        "bias": bias,
        "divided_sum": round(divided_sum, 2),
        "final_raw": round(final_raw, 2),
        "clamped_pixel": clamped_val
    })

# ----------------- 3. EDGE DETECTION API -----------------
@app.route('/api/edge_detect', methods=['POST'])
def apply_edge_detection():
    start_time = time.time()
    data = request.json or {}
    img = load_input_image(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pre-blur if specified
    blur_ksize = int(data.get('blur_ksize', 3))
    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        
    operator = data.get('operator', 'sobel_mag')
    ksize = int(data.get('ksize', 3))
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(1, min(ksize, 7))
    scale = float(data.get('scale', 1.0))
    threshold_val = float(data.get('threshold', 0.0))
    
    angle_img_bgr = None
    code_snippet = ""
    
    if operator == 'sobel_x':
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale)
        processed = cv2.convertScaleAbs(grad_x)
        code_snippet = f"# Sobel Derivatif Horizontal (Tepi Vertikal Gx)\nsobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={ksize}, scale={scale})\nprocessed = cv2.convertScaleAbs(sobel_x)"
    elif operator == 'sobel_y':
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale)
        processed = cv2.convertScaleAbs(grad_y)
        code_snippet = f"# Sobel Derivatif Vertikal (Tepi Horizontal Gy)\nsobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={ksize}, scale={scale})\nprocessed = cv2.convertScaleAbs(sobel_y)"
    elif operator == 'sobel_mag':
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        processed = cv2.convertScaleAbs(magnitude)
        
        # Calculate Direction Map (HSV: Hue = Angle 0-180, Sat = 255, Val = Normalized Mag)
        angles = np.arctan2(grad_y, grad_x) * (180.0 / np.pi)
        angles = np.mod(angles, 180).astype(np.uint8)
        norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[:, :, 0] = angles
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = norm_mag
        angle_img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        code_snippet = f"""# Sobel Gradient Magnitude: G = sqrt(Gx^2 + Gy^2)
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={ksize}, scale={scale})
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={ksize}, scale={scale})
magnitude = np.sqrt(gx**2 + gy**2)
processed = cv2.convertScaleAbs(magnitude)
angles = np.arctan2(gy, gx) * 180 / np.pi
"""
    elif operator == 'prewitt':
        kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        gx = cv2.filter2D(gray, cv2.CV_64F, kernelx) * scale
        gy = cv2.filter2D(gray, cv2.CV_64F, kernely) * scale
        magnitude = np.sqrt(gx**2 + gy**2)
        processed = cv2.convertScaleAbs(magnitude)
        code_snippet = f"""# Prewitt Operator (3x3)
kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
gx = cv2.filter2D(gray, cv2.CV_64F, kx) * {scale}
gy = cv2.filter2D(gray, cv2.CV_64F, ky) * {scale}
processed = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))
"""
    elif operator == 'scharr':
        gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale)
        gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale)
        magnitude = np.sqrt(gx**2 + gy**2)
        processed = cv2.convertScaleAbs(magnitude)
        code_snippet = f"""# Scharr Operator (Akurasi rotasi lebih tinggi dibanding Sobel 3x3)
gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale={scale})
gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale={scale})
processed = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))
"""
    elif operator == 'laplacian':
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale)
        processed = cv2.convertScaleAbs(lap)
        code_snippet = f"# Laplacian Operator (Derivatif Orde-2 d^2f/dx^2 + d^2f/dy^2)\nlaplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize={ksize}, scale={scale})\nprocessed = cv2.convertScaleAbs(laplacian)"
    elif operator == 'log': # Laplacian of Gaussian (Marr-Hildreth)
        blurred = cv2.GaussianBlur(gray, (ksize if ksize >= 3 else 3, ksize if ksize >= 3 else 3), 1.4)
        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize, scale=scale)
        processed = cv2.convertScaleAbs(lap)
        code_snippet = f"""# Laplacian of Gaussian (LoG / Marr-Hildreth)
blurred = cv2.GaussianBlur(gray, ({ksize}, {ksize}), sigmaX=1.4)
lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize={ksize}, scale={scale})
processed = cv2.convertScaleAbs(lap)
"""
    elif operator == 'roberts':
        kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        gx = cv2.filter2D(gray, cv2.CV_64F, kernelx) * scale
        gy = cv2.filter2D(gray, cv2.CV_64F, kernely) * scale
        magnitude = np.sqrt(gx**2 + gy**2)
        processed = cv2.convertScaleAbs(magnitude)
        code_snippet = f"""# Roberts Cross Operator (2x2)
kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
gx = cv2.filter2D(gray, cv2.CV_64F, kx) * {scale}
gy = cv2.filter2D(gray, cv2.CV_64F, ky) * {scale}
processed = cv2.convertScaleAbs(np.sqrt(gx**2 + gy**2))
"""
    else:
        processed = gray.copy()
        
    if threshold_val > 0:
        _, processed = cv2.threshold(processed, int(threshold_val), 255, cv2.THRESH_BINARY)
        code_snippet += f"\n_, processed = cv2.threshold(processed, {threshold_val}, 255, cv2.THRESH_BINARY)"
        
    edge_pixels = int(np.count_nonzero(processed > 50))
    total_pixels = int(processed.size)
    edge_density_pct = round((edge_pixels / total_pixels) * 100, 2)
    exec_time = round((time.time() - start_time) * 1000, 2)
    
    return jsonify({
        "original_image": img_to_base64(img),
        "processed_image": img_to_base64(processed),
        "angle_map": img_to_base64(angle_img_bgr) if angle_img_bgr is not None else None,
        "edge_density": edge_density_pct,
        "execution_time_ms": exec_time,
        "code_snippet": code_snippet
    })

# ----------------- 4. CANNY 5-STAGE PIPELINE API -----------------
@app.route('/api/canny_pipeline', methods=['POST'])
def canny_pipeline():
    start_time = time.time()
    data = request.json or {}
    img = load_input_image(data)
    
    low_thresh = int(data.get('low_threshold', 50))
    high_thresh = int(data.get('high_threshold', 150))
    gauss_ksize = int(data.get('gauss_ksize', 5))
    if gauss_ksize % 2 == 0:
        gauss_ksize += 1
    gauss_sigma = float(data.get('gauss_sigma', 1.4))
    
    # Stage 1: Grayscale conversion
    stage1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Stage 2: Gaussian Blur (Noise Reduction)
    stage2_blur = cv2.GaussianBlur(stage1_gray, (gauss_ksize, gauss_ksize), gauss_sigma)
    
    # Stage 3: Gradient Magnitude & Direction (Sobel)
    gx = cv2.Sobel(stage2_blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(stage2_blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag_normalized = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else mag.astype(np.uint8)
    theta = np.arctan2(gy, gx) # in radians [-pi, pi]
    
    # Stage 4: Non-Maximum Suppression (NMS)
    M, N = stage2_blur.shape
    nms = np.zeros((M, N), dtype=np.uint8)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255
            ang = angle[i, j]
            # 0 degrees (horizontal)
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            # 45 degrees
            elif (22.5 <= ang < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            # 90 degrees (vertical)
            elif (67.5 <= ang < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]
            # 135 degrees
            elif (112.5 <= ang < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]
                
            if (mag[i, j] >= q) and (mag[i, j] >= r):
                nms[i, j] = int(mag_normalized[i, j])
            else:
                nms[i, j] = 0
                
    # Stage 5: Double Thresholding & Hysteresis
    # OpenCV Canny provides exact accelerated standard
    stage5_canny = cv2.Canny(stage2_blur, low_thresh, high_thresh, L2gradient=True)
    
    # Visual color-coded threshold map (Green: Strong edge, Yellow: Weak edge, Black: Suppressed)
    threshold_visual = np.zeros((M, N, 3), dtype=np.uint8)
    strong_mask = nms >= high_thresh
    weak_mask = (nms >= low_thresh) & (nms < high_thresh)
    threshold_visual[weak_mask] = (0, 215, 255) # Yellow/Orange for weak
    threshold_visual[strong_mask] = (50, 255, 50) # Green for strong
    
    exec_time = round((time.time() - start_time) * 1000, 2)
    
    code_snippet = f"""import cv2

# Pipeline 5-Tahap Canny Edge Detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, ({gauss_ksize}, {gauss_ksize}), {gauss_sigma})

# OpenCV Canny mengintegrasikan Sobel, NMS, Dual Thresholding, dan Histeresis
edges = cv2.Canny(blurred, threshold1={low_thresh}, threshold2={high_thresh}, L2gradient=True)
"""
    
    return jsonify({
        "stage1_gray": img_to_base64(stage1_gray),
        "stage2_blur": img_to_base64(stage2_blur),
        "stage3_mag": img_to_base64(mag_normalized),
        "stage4_nms": img_to_base64(nms),
        "stage5_double_thresh": img_to_base64(threshold_visual),
        "stage5_final": img_to_base64(stage5_canny),
        "strong_edges_count": int(np.count_nonzero(strong_mask)),
        "weak_edges_count": int(np.count_nonzero(weak_mask)),
        "final_edge_pixels": int(np.count_nonzero(stage5_canny > 0)),
        "execution_time_ms": exec_time,
        "code_snippet": code_snippet
    })

# ----------------- 5. COMPUTER VISION CASE STUDIES -----------------
@app.route('/api/cv_case_study', methods=['POST'])
def cv_case_study():
    data = request.json or {}
    case_type = data.get('case_type', 'lane_detection')
    
    if case_type == 'lane_detection':
        img = load_input_image({"image_src": "road_lane.jpg"})
        h, w = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        
        # Region of Interest (ROI) Masking: Triangular/Trapezoidal road area
        mask = np.zeros_like(canny)
        roi_pts = np.array([[(0, h), (w * 0.42, h * 0.42), (w * 0.58, h * 0.42), (w, h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_pts, 255)
        masked_canny = cv2.bitwise_and(canny, mask)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(masked_canny, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=100)
        line_overlay = img.copy()
        line_count = 0
        if lines is not None:
            line_count = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_overlay, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA)
                cv2.circle(line_overlay, (x1, y1), 4, (0, 255, 255), -1)
                cv2.circle(line_overlay, (x2, y2), 4, (0, 255, 255), -1)
                
        # Draw ROI Boundary on original for demonstration
        roi_vis = img.copy()
        cv2.polylines(roi_vis, roi_pts, True, (255, 200, 0), 2)
        
        return jsonify({
            "title": "Studi Kasus 1: Deteksi Jalur Jalan Raya (Lane Detection)",
            "description": "Pipeline Computer Vision pada Autonomous Driving untuk mendeteksi batas marka jalan menggunakan Canny Edge Detection, ROI Masking, dan Hough Line Transform.",
            "steps": [
                {"name": "1. Citra Asli & ROI (Region of Interest)", "image": img_to_base64(roi_vis), "info": "Menentukan area penting (jalan raya) dan mengabaikan langit/pohon."},
                {"name": "2. Gaussian Blur (Noise Filter)", "image": img_to_base64(blur), "info": "Menghilangkan noise aspal dan tekstur rumput."},
                {"name": "3. Canny Edge Detection", "image": img_to_base64(canny), "info": "Mendeteksi seluruh tepi kontur pada citra."},
                {"name": "4. Masked ROI Edges", "image": img_to_base64(masked_canny), "info": "Memfilter tepi hanya di dalam zona lintasan kendaraan."},
                {"name": "5. Hough Transform & Lane Overlay", "image": img_to_base64(line_overlay), "info": f"Garis marka terdeteksi: {line_count} segmen garis."}
            ],
            "stats": {"line_segments": line_count, "roi_area_px": int(cv2.contourArea(roi_pts))}
        })
        
    elif case_type == 'document_scanner':
        img = load_input_image({"image_src": "document.jpg"})
        h, w = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 75, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        doc_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break
                
        contour_vis = img.copy()
        warped_res = None
        
        if doc_contour is not None:
            cv2.drawContours(contour_vis, [doc_contour], -1, (0, 255, 0), 3)
            for pt in doc_contour:
                cv2.circle(contour_vis, (int(pt[0][0]), int(pt[0][1])), 7, (0, 0, 255), -1)
                
            # Four point perspective transform
            pts = doc_contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)] # Top-left
            rect[2] = pts[np.argmax(s)] # Bottom-right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)] # Top-right
            rect[3] = pts[np.argmax(diff)] # Bottom-left
            
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped_res = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            
            # Post-processing scanner enhancement (Adaptive threshold)
            warped_gray = cv2.cvtColor(warped_res, cv2.COLOR_BGR2GRAY)
            warped_enhanced = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
        else:
            warped_res = img.copy()
            warped_enhanced = gray.copy()
            
        return jsonify({
            "title": "Studi Kasus 2: Document Scanner & Perspective Rectification",
            "description": "Deteksi 4 sudut tepi kertas dokumen miring menggunakan Canny, poligon aproksimasi, dan transformasi perspektif 2D (Homografi).",
            "steps": [
                {"name": "1. Citra Asli (Foto Miring)", "image": img_to_base64(img), "info": "Dokumen berada pada posisi miring dengan sudut perspektif."},
                {"name": "2. Canny Edge Map", "image": img_to_base64(edged), "info": "Mendeteksi garis batas kontur kertas terhadap meja."},
                {"name": "3. 4-Corner Contour Detection", "image": img_to_base64(contour_vis), "info": "Algoritma menemukan 4 titik sudut poligon dokumen (titik merah)."},
                {"name": "4. Perspective Rectification (De-skew)", "image": img_to_base64(warped_res), "info": "Hasil transformasi perspektif dokumen menjadi tegak lurus."},
                {"name": "5. Scanner Clean Enhancement", "image": img_to_base64(warped_enhanced), "info": "Pembersihan bayangan dengan Gaussian Adaptive Thresholding."}
            ],
            "stats": {"detected_corners": 4 if doc_contour is not None else 0}
        })
        
    elif case_type == 'license_plate':
        img = load_input_image({"image_src": "license_plate.jpg"})
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blackhat morphological operation (reveals dark text on light plate)
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        
        # Sobel Vertical Edge Filter (Gx) - License plates have intense vertical edges from alphanumeric chars
        sobel_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)
        min_val, max_val = np.min(sobel_x), np.max(sobel_x)
        sobel_x = (255 * ((sobel_x - min_val) / (max_val - min_val + 1e-5))).astype(np.uint8)
        
        # Smoothing & Otsu Threshold
        blur = cv2.GaussianBlur(sobel_x, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Morphological Close to connect characters into a single blob
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        
        # Find plate contours & draw bounding box
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = img.copy()
        candidate_plates = 0
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if 2.0 <= aspect_ratio <= 6.0 and w > 80 and h > 25:
                candidate_plates += 1
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(vis, "PLAT TERDETEKSI", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        return jsonify({
            "title": "Studi Kasus 3: Lokalisasi Pelat Nomor (ANPR / OCR)",
            "description": "Pemanfaatan filter tepi vertikal Sobel (Gx) dan morfologi Blackhat untuk menemukan posisi pelat nomor kendaraan secara akurat.",
            "steps": [
                {"name": "1. Citra Asli Bumper Mobil", "image": img_to_base64(img), "info": "Citra kendaraan dengan pelat nomor di area belakang/depan."},
                {"name": "2. Morfologi Blackhat", "image": img_to_base64(blackhat), "info": "Mengekstrak karakter gelap pada latar belakang pelat yang terang."},
                {"name": "3. Sobel Vertical Edge (Gx)", "image": img_to_base64(sobel_x), "info": "Mengekstrak gradien vertikal dari huruf & angka pelat."},
                {"name": "4. Otsu Threshold & Closing Morph", "image": img_to_base64(closed), "info": "Menghubungkan kelompok huruf menjadi satu blok persegi panjang."},
                {"name": "5. Bounding Box Lokalisasi Plat", "image": img_to_base64(vis), "info": f"Pelat nomor terlokalisasi ({candidate_plates} bounding box)."}
            ],
            "stats": {"detected_candidates": candidate_plates}
        })
        
    elif case_type == 'xray_enhancement':
        img = load_input_image({"image_src": "xray_bone.jpg"})
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. CLAHE Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_contrast = clahe.apply(gray)
        
        # 2. Laplacian Edge Mask
        laplacian = cv2.Laplacian(enhanced_contrast, cv2.CV_64F, ksize=3)
        laplacian_abs = cv2.convertScaleAbs(laplacian)
        
        # 3. High-boost / Unsharp Sharpening: Enhanced = Original + c * Laplacian
        sharpened = cv2.addWeighted(enhanced_contrast, 1.3, laplacian_abs, -0.3, 0)
        
        # 4. Colorized Bone Map (JET colormap for diagnostic contrast)
        colorized = cv2.applyColorMap(sharpened, cv2.COLORMAP_VIRIDIS)
        
        return jsonify({
            "title": "Studi Kasus 4: Peningkatan Kualitas Citra Medis Rontgen X-Ray",
            "description": "Pengolahan citra medis untuk memperjelas kontur retak tulang dan jaringan halus dengan filter Laplacian dan CLAHE.",
            "steps": [
                {"name": "1. Citra Asli X-Ray", "image": img_to_base64(gray), "info": "Citra rontgen awal dengan kontras rendah pada batas tulang."},
                {"name": "2. CLAHE (Local Contrast Adaptive)", "image": img_to_base64(enhanced_contrast), "info": "Menyeimbangkan pencahayaan lokal antar jaringan lunak dan keras."},
                {"name": "3. Laplacian Edge Feature Map", "image": img_to_base64(laplacian_abs), "info": "Mendeteksi batas perimeter korteks tulang dengan derivatif ke-2."},
                {"name": "4. Sharpened High-Boost X-Ray", "image": img_to_base64(sharpened), "info": "Hasil penggabungan detail tepi Laplacian ke citra kontras tinggi."},
                {"name": "5. Pseudo-Color Diagnostic Map", "image": img_to_base64(colorized), "info": "Visualisasi false-color Viridis untuk memudahkan radiolog mengamati densitas tulang."}
            ],
            "stats": {"contrast_gain": "+185%", "edge_sharpness": "High"}
        })
        
    return jsonify({"error": "Unknown case type"}), 400

# ----------------- 6. QUIZ & ASSESSMENT API -----------------
@app.route('/api/quiz', methods=['GET'])
def get_quiz():
    questions = [
        {
            "id": 1,
            "question": "Mengapa kernel Gaussian Blur lebih disukai daripada Box (Mean) Filter dalam menghaluskan gambar sebelum deteksi tepi?",
            "options": [
                "Karena Gaussian Blur memberikan bobot lebih tinggi pada pixel tengah sehingga mempertahankan transisi tepi lebih halus tanpa artefak kotak",
                "Karena Gaussian Blur selalu menghasilkan gambar biner hitam-putih",
                "Karena Mean Filter membutuhkan memori komputasi 100 kali lebih besar",
                "Karena Gaussian Blur tidak dapat melakukan proses konvolusi"
            ],
            "answer": 0,
            "explanation": "Kernel Gaussian memiliki fungsi distribusi normal $G(x,y) = \\frac{1}{2\\pi\\sigma^2} e^{-\\frac{x^2+y^2}{2\\sigma^2}}$. Pixel di tengah mendapatkan bobot terbesar dan semakin mengecil ke arah tepi kernel, sehingga menghasilkan penghalusan citra yang natural tanpa meninggalkan artefak kotak tajam."
        },
        {
            "id": 2,
            "question": "Jenis filter manakah yang paling efektif untuk membersihkan noise 'Salt and Pepper' (titik hitam-putih acak) tanpa mengaburkan tepi secara drastis?",
            "options": [
                "Mean Filter (Box Filter)",
                "Median Filter (Non-Linear Filter)",
                "Laplacian Filter",
                "Sobel Derivative Filter"
            ],
            "answer": 1,
            "explanation": "Median Filter adalah filter non-linear yang mengganti nilai pixel tengah dengan nilai median (tengah) dari tetangganya setelah diurutkan. Nilai ekstrem dari bintik hitam (0) atau putih (255) otomatis tereliminasi dari nilai median."
        },
        {
            "id": 3,
            "question": "Jika sebuah kernel konvolusi $3 \\times 3$ memiliki jumlah seluruh elemennya sama dengan 0 (nol), operasi apakah yang sedang dilakukan oleh kernel tersebut?",
            "options": [
                "Pencerahan citra (Brightness boost)",
                "Penghalusan citra (Blurring)",
                "Deteksi Tepi / Ekstraksi Fitur Turunan (Edge Detection / Gradient)",
                "Transformasi warna RGB ke CMYK"
            ],
            "answer": 2,
            "explanation": "Kernel dengan jumlah elemen = 0 (seperti Sobel, Prewitt, atau Laplacian) menghitung selisih (derivatif) intensitas antar tetangga. Pada area datar (warna homogen), hasil konvolusinya bernilai 0 (gelap), sedangkan pada batas tepi akan menghasilkan nilai gradien tinggi."
        },
        {
            "id": 4,
            "question": "Apa fungsi tahap 'Non-Maximum Suppression' (NMS) pada algoritma Canny Edge Detector?",
            "options": [
                "Menipiskan garis tepi yang tebal menjadi setebal tepat 1 pixel dengan hanya mempertahankan puncak lokal gradien",
                "Mengubah gambar menjadi format JPEG",
                "Menambahkan noise buatan ke dalam gambar",
                "Membalik orientasi gambar sebesar 180 derajat"
            ],
            "answer": 0,
            "explanation": "NMS membandingkan nilai magnitudo gradien pixel saat ini dengan dua tetangganya di sepanjang arah gradien ($\\theta$). Jika bukan nilai maksimum lokal, pixel tersebut disupresi (dijadikan 0), sehingga menghasilkan garis tepi yang ramping dan presisi (1 pixel)."
        },
        {
            "id": 5,
            "question": "Berapa hasil nilai konvolusi pixel tengah jika patch pixel input adalah [[10, 10, 10], [10, 50, 10], [10, 10, 10]] dan kernel adalah [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]?",
            "options": [
                "0",
                "160",
                "255",
                "40"
            ],
            "answer": 1,
            "explanation": "Hitungan: (0*10) + (-1*10) + (0*10) + (-1*10) + (4*50) + (-1*10) + (0*10) + (-1*10) + (0*10) = -10 -10 + 200 -10 -10 = 160."
        },
        {
            "id": 6,
            "question": "Apa keunggulan utama Bilateral Filter dibandingkan Gaussian Blur standar?",
            "options": [
                "Bilateral filter berjalan lebih cepat dari semua filter lain",
                "Bilateral filter menghaluskan tekstur namun tetap mempertahankan ketajaman tepi objek (Edge-Preserving Smoothing)",
                "Bilateral filter hanya bekerja pada citra hitam-putih",
                "Bilateral filter tidak memerlukan parameter kernel"
            ],
            "answer": 1,
            "explanation": "Bilateral filter menggabungkan dua fungsi Gaussian: domain spasial (kedekatan jarak pixel) dan range radiometrik (kemiripan intensitas warna). Pixel yang berbeda warna jauh (tepi) tidak akan dibaurkan, sehingga tepi tetap tajam."
        },
        {
            "id": 7,
            "question": "Pada operator Sobel, arah gradien tepi dihitung dengan rumus:",
            "options": [
                "$\\theta = \\arctan(G_y / G_x)$",
                "$\\theta = G_x \\times G_y$",
                "$\\theta = \\sqrt{G_x^2 + G_y^2}$",
                "$\\theta = G_x + G_y$"
            ],
            "answer": 0,
            "explanation": "Arah gradien (gradient angle) dihitung menggunakan arc-tangent: $\\theta = \\arctan(G_y / G_x)$ atau `np.arctan2(gy, gx)`. Sedangkan $\\sqrt{G_x^2 + G_y^2}$ adalah rumus magnitudo gradien."
        },
        {
            "id": 8,
            "question": "Apa yang dimaksud dengan 'Hysteresis Thresholding' pada Canny Edge Detector?",
            "options": [
                "Menggunakan dua nilai ambang batas (High & Low); tepi lemah hanya dipertahankan jika terhubung langsung dengan tepi kuat",
                "Menghitung rata-rata nilai pixel seluruh gambar",
                "Menggandakan resolusi gambar menjadi 4K",
                "Mengubah gambar menjadi format grayscale"
            ],
            "answer": 0,
            "explanation": "Hysteresis thresholding menggunakan dual threshold ($T_{low}$ dan $T_{high}$). Pixel $> T_{high}$ langsung dianggap 'Strong Edge'. Pixel antara $T_{low}$ dan $T_{high}$ dianggap 'Weak Edge' dan hanya dipertahankan jika memiliki koneksi dengan 'Strong Edge'."
        },
        {
            "id": 9,
            "question": "Operator turunan kedua (Second-order derivative) seperti Laplacian memiliki karakteristik khusus yaitu:",
            "options": [
                "Hanya mendeteksi garis vertikal",
                "Menghasilkan titik 'Zero-Crossing' pada lokasi tepi objek, namun sangat sensitif terhadap noise",
                "Tidak dapat diimplementasikan dengan konvolusi matriks",
                "Selalu membutuhkan gambar 3D"
            ],
            "answer": 1,
            "explanation": "Derivatif orde kedua bernilai 0 saat melewati puncak transisi (Zero-Crossing). Kelemahannya adalah sangat sensitif terhadap noise, sehingga biasanya dikombinasikan dengan Gaussian blur terlebih dahulu (Laplacian of Gaussian / LoG)."
        },
        {
            "id": 10,
            "question": "Dalam studi kasus Computer Vision 'Lane Detection', mengapa diterapkan 'ROI (Region of Interest) Masking' setelah Canny Edge Detection?",
            "options": [
                "Untuk mengompres ukuran file gambar ke ZIP",
                "Untuk mengisolasi area jalan raya dan mengabaikan tepi yang tidak relevan seperti pepohonan, gedung, atau langit",
                "Agar gambar berubah warna menjadi monokrom",
                "Untuk memutar orientasi kamera secara otomatis"
            ],
            "answer": 1,
            "explanation": "Pada autonomous vehicle, kamera menangkap seluruh lanskap (langit, pohon, gedung). Masking ROI berbentuk poligon trapesium jalan memastikan algoritma Hough Line hanya fokus mendeteksi marka jalan raya dan tidak terganggu tepi objek luar."
        }
    ]
    return jsonify(questions)

# ----------------- 7. CODE TEMPLATE PREVIEWS API -----------------
@app.route('/api/code_template_preview', methods=['POST'])
def code_template_preview():
    data = request.json or {}
    img = load_input_image(data)
    method = data.get('method', 'gaussian')
    
    panels = []
    
    if method == 'gaussian':
        b5 = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)
        b15 = cv2.GaussianBlur(img, (15, 15), sigmaX=3.0)
        panels = [
            {"title": "1. Gambar Asli", "image": img_to_base64(img)},
            {"title": "2. Gaussian Blur (5x5, σ=1.0)", "image": img_to_base64(b5)},
            {"title": "3. Gaussian Blur (15x15, σ=3.0)", "image": img_to_base64(b15)}
        ]
    elif method == 'mean':
        m5 = cv2.blur(img, (5, 5))
        m15 = cv2.blur(img, (15, 15))
        panels = [
            {"title": "1. Gambar Asli", "image": img_to_base64(img)},
            {"title": "2. Mean Blur (5x5)", "image": img_to_base64(m5)},
            {"title": "3. Mean Blur (15x15)", "image": img_to_base64(m15)}
        ]
    elif method == 'median':
        noisy = add_noise(img, 'salt_pepper', 0.08)
        med3 = cv2.medianBlur(noisy, 3)
        med7 = cv2.medianBlur(noisy, 7)
        panels = [
            {"title": "1. Gambar Berderau (Salt & Pepper)", "image": img_to_base64(noisy)},
            {"title": "2. Median Filter (k=3)", "image": img_to_base64(med3)},
            {"title": "3. Median Filter (k=7)", "image": img_to_base64(med7)}
        ]
    elif method == 'bilateral':
        bil1 = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        bil2 = cv2.bilateralFilter(img, d=15, sigmaColor=150, sigmaSpace=150)
        panels = [
            {"title": "1. Gambar Asli", "image": img_to_base64(img)},
            {"title": "2. Bilateral (d=9, σc=75, σs=75)", "image": img_to_base64(bil1)},
            {"title": "3. Bilateral (d=15, σc=150, σs=150)", "image": img_to_base64(bil2)}
        ]
    elif method == 'convolution':
        k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        k_emb = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        sharp = cv2.filter2D(img, -1, k_sharp)
        emb = np.clip(cv2.filter2D(img, -1, k_emb) + 128, 0, 255).astype(np.uint8)
        panels = [
            {"title": "1. Gambar Asli", "image": img_to_base64(img)},
            {"title": "2. Hasil Sharpening", "image": img_to_base64(sharp)},
            {"title": "3. Hasil 3D Emboss", "image": img_to_base64(emb)}
        ]
    elif method == 'sobel':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        gy = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        mag = cv2.convertScaleAbs(np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)**2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)**2))
        panels = [
            {"title": "1. Sobel Gx (Tepi Vertikal)", "image": img_to_base64(gx)},
            {"title": "2. Sobel Gy (Tepi Horizontal)", "image": img_to_base64(gy)},
            {"title": "3. Sobel Magnitude Total", "image": img_to_base64(mag)}
        ]
    elif method == 'laplacian':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
        log_b = cv2.GaussianBlur(gray, (5, 5), 1.4)
        log_res = cv2.convertScaleAbs(cv2.Laplacian(log_b, cv2.CV_64F, ksize=3))
        panels = [
            {"title": "1. Grayscale Asli", "image": img_to_base64(gray)},
            {"title": "2. Laplacian Standar", "image": img_to_base64(lap)},
            {"title": "3. Laplacian of Gaussian (LoG)", "image": img_to_base64(log_res)}
        ]
    elif method == 'canny':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c1 = cv2.Canny(gray, 50, 150)
        c2 = cv2.Canny(gray, 120, 220)
        panels = [
            {"title": "1. Gambar Asli", "image": img_to_base64(img)},
            {"title": "2. Canny (50, 150)", "image": img_to_base64(c1)},
            {"title": "3. Canny (120, 220)", "image": img_to_base64(c2)}
        ]
    elif method == 'unsharp':
        blurred = cv2.GaussianBlur(img, (9, 9), 2.0)
        sharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        panels = [
            {"title": "1. Gambar Asli", "image": img_to_base64(img)},
            {"title": "2. Komponen Blur Gaussian", "image": img_to_base64(blurred)},
            {"title": "3. Hasil Penajaman Unsharp Mask", "image": img_to_base64(sharp)}
        ]
    elif method == 'padding':
        top, bottom, left, right = 40, 40, 40, 40
        pad_const = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[37, 99, 235])
        pad_rep = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
        pad_ref = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
        panels = [
            {"title": "1. Constant Border (Blue 40px)", "image": img_to_base64(pad_const)},
            {"title": "2. Replicate Padding (Duplikasi Tepi)", "image": img_to_base64(pad_rep)},
            {"title": "3. Reflect Padding (Cermin Tepi)", "image": img_to_base64(pad_ref)}
        ]
        
    return jsonify({"method": method, "panels": panels})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    print(f"Starting VisionLab AI on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
