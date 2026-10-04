/**
 * VisionLab AI - 2D Convolution Studio & Pixel Math Probe
 * Interactive 3x3 / 5x5 matrix editor, presets, and hover/click pixel probe
 */

const ConvState = {
    gridSize: 3, // 3 or 5
    kernel: [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ],
    divisor: 1,
    bias: 0,
    borderMode: 'replicate',
    probeX: 150,
    probeY: 120
};

const KernelPresets = {
    sharpen: {
        size: 3,
        matrix: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        divisor: 1,
        bias: 0
    },
    box_blur: {
        size: 3,
        matrix: [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        divisor: 9,
        bias: 0
    },
    gaussian: {
        size: 3,
        matrix: [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        divisor: 16,
        bias: 0
    },
    laplacian: {
        size: 3,
        matrix: [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        divisor: 1,
        bias: 128
    },
    sobel_h: {
        size: 3,
        matrix: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        divisor: 1,
        bias: 128
    },
    sobel_v: {
        size: 3,
        matrix: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        divisor: 1,
        bias: 128
    },
    emboss: {
        size: 3,
        matrix: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
        divisor: 1,
        bias: 128
    },
    ridge: {
        size: 3,
        matrix: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        divisor: 1,
        bias: 0
    },
    identity: {
        size: 3,
        matrix: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        divisor: 1,
        bias: 0
    }
};

document.addEventListener('DOMContentLoaded', () => {
    initConvolutionModule();
});

function initConvolutionModule() {
    renderKernelMatrixGrid();
    initKernelPresets();
    initGridSizeToggle();
    initProbeInteraction();
    
    // Apply Button
    document.getElementById('btnApplyCustomConv').addEventListener('click', () => {
        readKernelFromUI();
        applyConvolutionAPI();
        showToast('Konvolusi kustom berhasil diterapkan!', 'fa-border-all', 'var(--purple)');
    });
    
    // Normalize Button
    document.getElementById('btnNormalizeKernel').addEventListener('click', () => {
        readKernelFromUI();
        let sum = 0;
        ConvState.kernel.forEach(row => row.forEach(val => sum += val));
        if (sum !== 0) {
            document.getElementById('kernelDivisor').value = sum;
            ConvState.divisor = sum;
            showToast(`Divisor diatur ke jumlah elemen kernel (${sum})`, 'fa-divide', 'var(--cyan)');
        } else {
            showToast('Jumlah elemen = 0 (Filter Turunan / Tepi), Divisor dibiarkan 1', 'fa-info-circle', 'var(--yellow)');
        }
        applyConvolutionAPI();
    });
    
    // Copy Code Button
    document.getElementById('btnCopyConvCode').addEventListener('click', () => {
        const code = document.getElementById('convCodeSnippet').textContent;
        navigator.clipboard.writeText(code);
        showToast('Kode cv2.filter2D berhasil disalin!', 'fa-copy', 'var(--yellow)');
    });
}

function initKernelPresets() {
    const pills = document.querySelectorAll('#kernelPresetPills .pill');
    pills.forEach(pill => {
        pill.addEventListener('click', () => {
            pills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            
            const presetKey = pill.getAttribute('data-preset');
            const preset = KernelPresets[presetKey];
            if (!preset) return;
            
            ConvState.gridSize = preset.size;
            ConvState.kernel = JSON.parse(JSON.stringify(preset.matrix));
            ConvState.divisor = preset.divisor;
            ConvState.bias = preset.bias;
            
            document.getElementById('kernelDivisor').value = ConvState.divisor;
            document.getElementById('kernelBias').value = ConvState.bias;
            
            // Update size button state
            document.getElementById('btnKernel3x3').classList.toggle('active', preset.size === 3);
            document.getElementById('btnKernel5x5').classList.toggle('active', preset.size === 5);
            
            renderKernelMatrixGrid();
            applyConvolutionAPI();
        });
    });
}

function initGridSizeToggle() {
    const btn3 = document.getElementById('btnKernel3x3');
    const btn5 = document.getElementById('btnKernel5x5');
    
    btn3.addEventListener('click', () => {
        btn3.classList.add('active');
        btn5.classList.remove('active');
        ConvState.gridSize = 3;
        ConvState.kernel = [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ];
        renderKernelMatrixGrid();
        applyConvolutionAPI();
    });
    
    btn5.addEventListener('click', () => {
        btn5.classList.add('active');
        btn3.classList.remove('active');
        ConvState.gridSize = 5;
        ConvState.kernel = [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ];
        ConvState.divisor = 256;
        document.getElementById('kernelDivisor').value = 256;
        renderKernelMatrixGrid();
        applyConvolutionAPI();
    });
}

function renderKernelMatrixGrid() {
    const grid = document.getElementById('kernelMatrixGrid');
    grid.innerHTML = '';
    grid.className = `kernel-matrix-grid grid-${ConvState.gridSize}x${ConvState.gridSize}`;
    
    for (let r = 0; r < ConvState.gridSize; r++) {
        for (let c = 0; c < ConvState.gridSize; c++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.className = 'matrix-cell-input';
            input.value = ConvState.kernel[r] && ConvState.kernel[r][c] !== undefined ? ConvState.kernel[r][c] : 0;
            input.setAttribute('data-r', r);
            input.setAttribute('data-c', c);
            
            input.addEventListener('change', () => {
                readKernelFromUI();
                applyConvolutionAPI();
            });
            grid.appendChild(input);
        }
    }
}

function readKernelFromUI() {
    const size = ConvState.gridSize;
    const inputs = document.querySelectorAll('#kernelMatrixGrid .matrix-cell-input');
    const newKernel = [];
    
    for (let r = 0; r < size; r++) {
        const row = [];
        for (let c = 0; c < size; c++) {
            const idx = r * size + c;
            const val = parseFloat(inputs[idx]?.value || 0);
            row.push(val);
        }
        newKernel.push(row);
    }
    
    ConvState.kernel = newKernel;
    ConvState.divisor = parseFloat(document.getElementById('kernelDivisor').value || 1);
    ConvState.bias = parseFloat(document.getElementById('kernelBias').value || 0);
    ConvState.borderMode = document.getElementById('kernelBorderMode').value;
}

// Call Convolution API
async function applyConvolutionAPI() {
    try {
        readKernelFromUI();
        const res = await fetch('/api/convolve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_src: VisionState.currentImageSrc,
                kernel: ConvState.kernel,
                divisor: ConvState.divisor,
                bias: ConvState.bias,
                border_mode: ConvState.borderMode,
                grayscale: false
            })
        });
        
        const data = await res.json();
        const convImg = document.getElementById('convResultImg');
        convImg.src = data.processed_image;
        
        const codeEl = document.getElementById('convCodeSnippet');
        codeEl.textContent = data.code_snippet;
        hljs.highlightElement(codeEl);
        
        // Trigger pixel probe update at current coordinate
        updatePixelProbe(ConvState.probeX, ConvState.probeY);
        
    } catch (err) {
        console.error('Convolution API error:', err);
    }
}

// Probe Pixel Hover / Click
function initProbeInteraction() {
    const wrap = document.getElementById('convCanvasWrap');
    const img = document.getElementById('convResultImg');
    const crosshair = document.getElementById('probeCrosshair');
    
    function handleProbe(e) {
        const rect = img.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        
        const clientX = e.clientX || (e.touches && e.touches[0]?.clientX);
        const clientY = e.clientY || (e.touches && e.touches[0]?.clientY);
        if (clientX === undefined) return;
        
        const relX = clientX - rect.left;
        const relY = clientY - rect.top;
        
        if (relX < 0 || relX > rect.width || relY < 0 || relY > rect.height) return;
        
        // Map to natural image coordinates
        const scaleX = (img.naturalWidth || 600) / rect.width;
        const scaleY = (img.naturalHeight || 400) / rect.height;
        
        const imgX = Math.floor(relX * scaleX);
        const imgY = Math.floor(relY * scaleY);
        
        crosshair.style.display = 'block';
        crosshair.style.left = `${relX + (img.offsetLeft || 0)}px`;
        crosshair.style.top = `${relY + (img.offsetTop || 0)}px`;
        
        ConvState.probeX = imgX;
        ConvState.probeY = imgY;
        
        updatePixelProbe(imgX, imgY);
    }
    
    wrap.addEventListener('mousemove', handleProbe);
    wrap.addEventListener('click', handleProbe);
}

let probeDebounceTimer = null;
async function updatePixelProbe(x, y) {
    clearTimeout(probeDebounceTimer);
    probeDebounceTimer = setTimeout(async () => {
        try {
            const res = await fetch('/api/convolve_pixel_probe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_src: VisionState.currentImageSrc,
                    x: x,
                    y: y,
                    kernel: ConvState.kernel,
                    divisor: ConvState.divisor,
                    bias: ConvState.bias
                })
            });
            
            const data = await res.json();
            
            // Update Header Coordinate
            document.getElementById('probeCoord').textContent = `(x: ${data.x}, y: ${data.y})`;
            
            // Render 3x3 (or 5x5) patch grid
            renderSubMatrix('patchMatrixDisplay', data.patch, false);
            renderSubMatrix('kernelMatrixDisplay', data.kernel, true);
            
            // Render products matrix
            const productMatrix = data.multiplications.map(row => row.map(cell => cell.product));
            renderSubMatrix('productMatrixDisplay', productMatrix, false);
            
            // Render Calculation Summary
            document.getElementById('calcSumProducts').textContent = data.sum_products;
            document.getElementById('calcDivided').textContent = data.divided_sum;
            document.getElementById('calcBiased').textContent = data.final_raw;
            document.getElementById('calcFinalPixel').textContent = `${data.clamped_pixel}`;
            
            const colorBox = document.getElementById('calcColorBox');
            colorBox.style.background = `rgb(${data.clamped_pixel}, ${data.clamped_pixel}, ${data.clamped_pixel})`;
            
        } catch (err) {
            console.error('Pixel probe error:', err);
        }
    }, 60);
}

function renderSubMatrix(elementId, matrix, isKernel = false) {
    const el = document.getElementById(elementId);
    if (!el || !matrix) return;
    el.innerHTML = '';
    
    const rows = matrix.length;
    const cols = matrix[0]?.length || 3;
    el.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const cell = document.createElement('div');
            cell.className = 'sub-cell';
            if (r === Math.floor(rows / 2) && c === Math.floor(cols / 2)) {
                cell.classList.add('highlight');
            }
            const val = matrix[r][c];
            cell.textContent = typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(1)) : val;
            el.appendChild(cell);
        }
    }
}
