/**
 * VisionLab AI - Real-time Live Camera Processing Studio
 * High-performance browser-side Canvas & TypedArray image filtering engine
 */

const LiveCamState = {
    stream: null,
    isRunning: false,
    filterMode: 'passthrough',
    fps: 0,
    frameCount: 0,
    lastFpsUpdate: Date.now()
};

document.addEventListener('DOMContentLoaded', () => {
    initLiveCameraModule();
});

function initLiveCameraModule() {
    const btnStart = document.getElementById('btnStartCam');
    const btnCapture = document.getElementById('btnCaptureCam');
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    const placeholder = document.getElementById('camPlaceholder');
    
    btnStart.addEventListener('click', async () => {
        if (!LiveCamState.isRunning) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                    audio: false
                });
                video.srcObject = stream;
                LiveCamState.stream = stream;
                LiveCamState.isRunning = true;
                
                placeholder.style.display = 'none';
                btnStart.innerHTML = '<i class="fa-solid fa-stop"></i> Matikan Kamera';
                btnStart.classList.replace('btn-primary', 'btn-secondary');
                btnCapture.disabled = false;
                
                video.play();
                requestAnimationFrame(processCamLoop);
                showToast('Kamera aktif! Pilih filter real-time.', 'fa-video', 'var(--emerald)');
            } catch (err) {
                console.error('Camera access error:', err);
                showToast('Izin kamera ditolak atau tidak ditemukan perangkat webcam.', 'fa-triangle-exclamation', 'var(--rose)');
            }
        } else {
            // Stop Camera
            if (LiveCamState.stream) {
                LiveCamState.stream.getTracks().forEach(track => track.stop());
            }
            LiveCamState.isRunning = false;
            placeholder.style.display = 'flex';
            btnStart.innerHTML = '<i class="fa-solid fa-power-off"></i> Hidupkan Kamera';
            btnStart.classList.replace('btn-secondary', 'btn-primary');
            btnCapture.disabled = true;
            document.getElementById('liveCamFps').textContent = '0 FPS';
            showToast('Kamera dinonaktifkan.');
        }
    });
    
    // Capture Frame to Global Sample
    btnCapture.addEventListener('click', () => {
        if (!LiveCamState.isRunning) return;
        const canvas = document.getElementById('webcamCanvas');
        const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
        
        VisionState.currentImageSrc = dataUrl;
        
        // Add to global select
        const select = document.getElementById('globalImageSelect');
        const opt = document.createElement('option');
        opt.value = dataUrl;
        opt.textContent = `Snapshot Kamera (${new Date().toLocaleTimeString()})`;
        opt.selected = true;
        select.appendChild(opt);
        
        showToast('Snapshot kamera disimpan! Beralih ke modul lab...', 'fa-camera-retro', 'var(--cyan)');
        refreshActiveTab();
    });
    
    // Live filter pill buttons
    const pills = document.querySelectorAll('#camFilterGrid .cam-pill');
    pills.forEach(pill => {
        pill.addEventListener('click', () => {
            pills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            LiveCamState.filterMode = pill.getAttribute('data-camfilter');
            showToast(`Filter Kamera: ${pill.textContent.trim()}`, 'fa-wand-magic-sparkles', 'var(--rose)');
        });
    });
}

function processCamLoop() {
    if (!LiveCamState.isRunning) return;
    
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    if (!video || !canvas || video.readyState < 2) {
        requestAnimationFrame(processCamLoop);
        return;
    }
    
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const width = canvas.width;
    const height = canvas.height;
    
    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, width, height);
    
    // Read raw pixels
    const frame = ctx.getImageData(0, 0, width, height);
    const data = frame.data;
    
    // Apply real-time filter algorithm
    applyLiveFilter(data, width, height, LiveCamState.filterMode);
    
    // Put filtered pixels back
    ctx.putImageData(frame, 0, 0);
    
    // Calculate FPS
    LiveCamState.frameCount++;
    const now = Date.now();
    if (now - LiveCamState.lastFpsUpdate >= 1000) {
        LiveCamState.fps = LiveCamState.frameCount;
        LiveCamState.frameCount = 0;
        LiveCamState.lastFpsUpdate = now;
        document.getElementById('liveCamFps').textContent = `${LiveCamState.fps} FPS`;
    }
    
    requestAnimationFrame(processCamLoop);
}

// Fast Canvas Pixel Manipulation Engine
function applyLiveFilter(data, width, height, mode) {
    if (mode === 'passthrough') return;
    
    const len = data.length;
    const copy = new Uint8ClampedArray(data);
    
    if (mode === 'sobel_edges') {
        // Fast Sobel gradient with neon glow
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                
                // Neighborhood Grayscale
                const p00 = (copy[((y-1)*width + (x-1))*4] + copy[((y-1)*width + (x-1))*4 + 1] + copy[((y-1)*width + (x-1))*4 + 2]) / 3;
                const p01 = (copy[((y-1)*width + x)*4] + copy[((y-1)*width + x)*4 + 1] + copy[((y-1)*width + x)*4 + 2]) / 3;
                const p02 = (copy[((y-1)*width + (x+1))*4] + copy[((y-1)*width + (x+1))*4 + 1] + copy[((y-1)*width + (x+1))*4 + 2]) / 3;
                
                const p10 = (copy[(y*width + (x-1))*4] + copy[(y*width + (x-1))*4 + 1] + copy[(y*width + (x-1))*4 + 2]) / 3;
                const p12 = (copy[(y*width + (x+1))*4] + copy[(y*width + (x+1))*4 + 1] + copy[(y*width + (x+1))*4 + 2]) / 3;
                
                const p20 = (copy[((y+1)*width + (x-1))*4] + copy[((y+1)*width + (x-1))*4 + 1] + copy[((y+1)*width + (x-1))*4 + 2]) / 3;
                const p21 = (copy[((y+1)*width + x)*4] + copy[((y+1)*width + x)*4 + 1] + copy[((y+1)*width + x)*4 + 2]) / 3;
                const p22 = (copy[((y+1)*width + (x+1))*4] + copy[((y+1)*width + (x+1))*4 + 1] + copy[((y+1)*width + (x+1))*4 + 2]) / 3;
                
                const gx = (-1 * p00) + (1 * p02) + (-2 * p10) + (2 * p12) + (-1 * p20) + (1 * p22);
                const gy = (-1 * p00) + (-2 * p01) + (-1 * p02) + (1 * p20) + (2 * p21) + (1 * p22);
                
                const mag = Math.min(255, Math.sqrt(gx * gx + gy * gy) * 1.5);
                
                // Cyan edge glow on dark background
                data[idx] = 0;
                data[idx + 1] = mag; // Green / Cyan
                data[idx + 2] = mag; // Blue / Cyan
            }
        }
    } else if (mode === 'canny_live') {
        // Fast Binarized Edge Detection
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                const p00 = copy[((y-1)*width + (x-1))*4];
                const p02 = copy[((y-1)*width + (x+1))*4];
                const p10 = copy[(y*width + (x-1))*4];
                const p12 = copy[(y*width + (x+1))*4];
                const p20 = copy[((y+1)*width + (x-1))*4];
                const p22 = copy[((y+1)*width + (x+1))*4];
                const p01 = copy[((y-1)*width + x)*4];
                const p21 = copy[((y+1)*width + x)*4];
                
                const gx = (-1 * p00) + (1 * p02) + (-2 * p10) + (2 * p12) + (-1 * p20) + (1 * p22);
                const gy = (-1 * p00) + (-2 * p01) + (-1 * p02) + (1 * p20) + (2 * p21) + (1 * p22);
                const mag = Math.sqrt(gx * gx + gy * gy);
                
                const edgeVal = (mag > 65) ? 255 : 0;
                data[idx] = edgeVal;
                data[idx + 1] = edgeVal;
                data[idx + 2] = edgeVal;
            }
        }
    } else if (mode === 'pencil_sketch') {
        // Inverted Pencil Sketch (White paper with dark graphite edges)
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                const p00 = copy[((y-1)*width + (x-1))*4];
                const p02 = copy[((y-1)*width + (x+1))*4];
                const p10 = copy[(y*width + (x-1))*4];
                const p12 = copy[(y*width + (x+1))*4];
                const p20 = copy[((y+1)*width + (x-1))*4];
                const p22 = copy[((y+1)*width + (x+1))*4];
                const p01 = copy[((y-1)*width + x)*4];
                const p21 = copy[((y+1)*width + x)*4];
                
                const gx = (-1 * p00) + (1 * p02) + (-2 * p10) + (2 * p12) + (-1 * p20) + (1 * p22);
                const gy = (-1 * p00) + (-2 * p01) + (-1 * p02) + (1 * p20) + (2 * p21) + (1 * p22);
                const mag = Math.min(255, Math.sqrt(gx * gx + gy * gy) * 1.6);
                
                const sketchVal = Math.max(0, 255 - mag);
                data[idx] = sketchVal;
                data[idx + 1] = sketchVal;
                data[idx + 2] = sketchVal;
            }
        }
    } else if (mode === 'gaussian_blur') {
        // 3x3 Box/Gaussian fast blur
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                for (let c = 0; c < 3; c++) {
                    const sum = copy[((y-1)*width + (x-1))*4 + c] +
                                copy[((y-1)*width + x)*4 + c] * 2 +
                                copy[((y-1)*width + (x+1))*4 + c] +
                                copy[(y*width + (x-1))*4 + c] * 2 +
                                copy[(y*width + x)*4 + c] * 4 +
                                copy[(y*width + (x+1))*4 + c] * 2 +
                                copy[((y+1)*width + (x-1))*4 + c] +
                                copy[((y+1)*width + x)*4 + c] * 2 +
                                copy[((y+1)*width + (x+1))*4 + c];
                    data[idx + c] = sum / 16;
                }
            }
        }
    } else if (mode === 'sharpen_live') {
        // Fast Sharpen Kernel: [0, -1, 0, -1, 5, -1, 0, -1, 0]
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                for (let c = 0; c < 3; c++) {
                    const center = copy[idx + c] * 5;
                    const up = copy[((y-1)*width + x)*4 + c];
                    const down = copy[((y+1)*width + x)*4 + c];
                    const left = copy[(y*width + (x-1))*4 + c];
                    const right = copy[(y*width + (x+1))*4 + c];
                    data[idx + c] = Math.min(255, Math.max(0, center - up - down - left - right));
                }
            }
        }
    } else if (mode === 'emboss_live') {
        // 3D Emboss Relief
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                for (let c = 0; c < 3; c++) {
                    const val = (-2 * copy[((y-1)*width + (x-1))*4 + c]) +
                                (-1 * copy[((y-1)*width + x)*4 + c]) +
                                (1 * copy[((y+1)*width + x)*4 + c]) +
                                (2 * copy[((y+1)*width + (x+1))*4 + c]) + 128;
                    data[idx + c] = Math.min(255, Math.max(0, val));
                }
            }
        }
    } else if (mode === 'cartoon') {
        // Cartoonifier: Posterization (Color quantization) + Edge overlay
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                
                // Edge detection
                const p00 = copy[((y-1)*width + (x-1))*4];
                const p02 = copy[((y-1)*width + (x+1))*4];
                const p10 = copy[(y*width + (x-1))*4];
                const p12 = copy[(y*width + (x+1))*4];
                const p20 = copy[((y+1)*width + (x-1))*4];
                const p22 = copy[((y+1)*width + (x+1))*4];
                const gx = (-1 * p00) + (1 * p02) + (-2 * p10) + (2 * p12) + (-1 * p20) + (1 * p22);
                const gy = (-1 * p00) + (-2 * copy[((y-1)*width + x)*4]) + (-1 * p02) + (1 * p20) + (2 * copy[((y+1)*width + x)*4]) + (1 * p22);
                const isEdge = Math.sqrt(gx * gx + gy * gy) > 55;
                
                if (isEdge) {
                    data[idx] = 10;
                    data[idx + 1] = 10;
                    data[idx + 2] = 10;
                } else {
                    // Quantize colors into 4 bins
                    data[idx] = Math.floor(copy[idx] / 64) * 64 + 32;
                    data[idx + 1] = Math.floor(copy[idx + 1] / 64) * 64 + 32;
                    data[idx + 2] = Math.floor(copy[idx + 2] / 64) * 64 + 32;
                }
            }
        }
    }
}
