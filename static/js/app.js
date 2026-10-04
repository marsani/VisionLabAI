/**
 * VisionLab AI - Core Application Logic
 * Tab handling, Global Image management, Filtering Lab, Histograms & Metrics
 */

// Global State
const VisionState = {
    currentImageSrc: 'road_lane.jpg',
    currentTab: 'tab-filtering',
    samples: [],
    
    // Filtering State
    filterType: 'gaussian',
    noiseType: 'none',
    noiseIntensity: 0.05,
    ksize: 5,
    sigma: 1.5,
    sigmaColor: 75,
    sigmaSpace: 75,
    
    // View mode
    viewMode: 'split', // 'split' or 'side'
};

// Toast Notification
function showToast(message, icon = 'fa-circle-check', color = 'var(--cyan)') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<i class="fa-solid ${icon}" style="color: ${color}"></i> <span>${message}</span>`;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
    initTabs();
    await loadSamples();
    initGlobalImageSelector();
    initFilteringControls();
    initSplitSlider();
    
    // Initial run
    applyFilterAPI();
});

// Tab Switching
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            const targetPane = document.getElementById(targetTab);
            if (targetPane) targetPane.classList.add('active');
            
            VisionState.currentTab = targetTab;
            
            // Trigger tab specific refreshes
            if (targetTab === 'tab-filtering') applyFilterAPI();
            else if (targetTab === 'tab-convolution') applyConvolutionAPI();
            else if (targetTab === 'tab-edges') applyEdgeDetectionAPI();
            else if (targetTab === 'tab-canny') applyCannyPipelineAPI();
            else if (targetTab === 'tab-casestudies') loadCaseStudy('lane_detection');
            else if (targetTab === 'tab-quiz') loadQuizData();
        });
    });
}

// Load Image Samples from Backend
async function loadSamples() {
    try {
        const res = await fetch('/api/samples');
        VisionState.samples = await res.json();
        
        const select = document.getElementById('globalImageSelect');
        select.innerHTML = '';
        VisionState.samples.forEach(sample => {
            const opt = document.createElement('option');
            opt.value = sample.id;
            opt.textContent = `${sample.name} (${sample.category})`;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error('Error loading samples:', err);
    }
}

// Global Image Controls
function initGlobalImageSelector() {
    const select = document.getElementById('globalImageSelect');
    const fileUpload = document.getElementById('globalImageUpload');
    
    select.addEventListener('change', (e) => {
        VisionState.currentImageSrc = e.target.value;
        showToast(`Citra sampel diganti ke: ${select.options[select.selectedIndex].text}`);
        refreshActiveTab();
    });
    
    fileUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.success) {
                VisionState.currentImageSrc = data.image_src;
                
                // Add custom option to dropdown
                const customOpt = document.createElement('option');
                customOpt.value = data.image_src;
                customOpt.textContent = `Custom: ${file.name}`;
                customOpt.selected = true;
                select.appendChild(customOpt);
                
                showToast(`Citra custom "${file.name}" berhasil diunggah!`, 'fa-cloud-arrow-up', 'var(--emerald)');
                refreshActiveTab();
            } else {
                showToast(data.error || 'Gagal upload citra', 'fa-triangle-exclamation', 'var(--rose)');
            }
        } catch (err) {
            showToast('Error uploading file', 'fa-triangle-exclamation', 'var(--rose)');
        }
    });
}

function refreshActiveTab() {
    if (VisionState.currentTab === 'tab-filtering') applyFilterAPI();
    else if (VisionState.currentTab === 'tab-convolution') applyConvolutionAPI();
    else if (VisionState.currentTab === 'tab-edges') applyEdgeDetectionAPI();
    else if (VisionState.currentTab === 'tab-canny') applyCannyPipelineAPI();
}

// Filtering Lab Controls
function initFilteringControls() {
    // Filter type pills
    const filterPills = document.querySelectorAll('#filterTypePills .pill');
    filterPills.forEach(pill => {
        pill.addEventListener('click', () => {
            filterPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            VisionState.filterType = pill.getAttribute('data-filter');
            
            // Adjust parameter visibility
            const sigmaRow = document.getElementById('sigmaRow');
            const sigmaColorRow = document.getElementById('sigmaColorRow');
            const sigmaSpaceRow = document.getElementById('sigmaSpaceRow');
            
            if (VisionState.filterType === 'bilateral') {
                sigmaRow.style.display = 'none';
                sigmaColorRow.style.display = 'block';
                sigmaSpaceRow.style.display = 'block';
            } else if (VisionState.filterType === 'gaussian' || VisionState.filterType === 'unsharp_mask') {
                sigmaRow.style.display = 'block';
                sigmaColorRow.style.display = 'none';
                sigmaSpaceRow.style.display = 'none';
            } else {
                sigmaRow.style.display = 'none';
                sigmaColorRow.style.display = 'none';
                sigmaSpaceRow.style.display = 'none';
            }
            
            applyFilterAPI();
        });
    });
    
    // Noise radio buttons
    const noiseRadios = document.querySelectorAll('input[name="noiseType"]');
    noiseRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            VisionState.noiseType = e.target.value;
            const intensityRow = document.getElementById('noiseIntensityRow');
            intensityRow.style.display = (VisionState.noiseType !== 'none') ? 'flex' : 'none';
            applyFilterAPI();
        });
    });
    
    // Noise Intensity Slider
    const noiseSlider = document.getElementById('noiseIntensity');
    const noiseVal = document.getElementById('noiseIntensityVal');
    noiseSlider.addEventListener('input', (e) => {
        VisionState.noiseIntensity = parseFloat(e.target.value);
        noiseVal.textContent = `${Math.round(VisionState.noiseIntensity * 100)}%`;
        applyFilterAPI();
    });
    
    // Ksize Slider
    const ksizeSlider = document.getElementById('filterKsize');
    const ksizeVal = document.getElementById('filterKsizeVal');
    ksizeSlider.addEventListener('input', (e) => {
        let val = parseInt(e.target.value);
        if (val % 2 === 0) val += 1; // Ensure odd
        VisionState.ksize = val;
        ksizeVal.textContent = `${val} × ${val}`;
        applyFilterAPI();
    });
    
    // Sigma Slider
    const sigmaSlider = document.getElementById('filterSigma');
    const sigmaVal = document.getElementById('filterSigmaVal');
    sigmaSlider.addEventListener('input', (e) => {
        VisionState.sigma = parseFloat(e.target.value);
        sigmaVal.textContent = VisionState.sigma.toFixed(1);
        applyFilterAPI();
    });
    
    // Bilateral Sigmas
    const sigmaColorSlider = document.getElementById('filterSigmaColor');
    const sigmaColorVal = document.getElementById('filterSigmaColorVal');
    sigmaColorSlider.addEventListener('input', (e) => {
        VisionState.sigmaColor = parseFloat(e.target.value);
        sigmaColorVal.textContent = VisionState.sigmaColor;
        applyFilterAPI();
    });
    
    const sigmaSpaceSlider = document.getElementById('filterSigmaSpace');
    const sigmaSpaceVal = document.getElementById('filterSigmaSpaceVal');
    sigmaSpaceSlider.addEventListener('input', (e) => {
        VisionState.sigmaSpace = parseFloat(e.target.value);
        sigmaSpaceVal.textContent = VisionState.sigmaSpace;
        applyFilterAPI();
    });
    
    // View Toggle
    const btnViewSplit = document.getElementById('btnViewSplit');
    const btnViewSide = document.getElementById('btnViewSide');
    const splitWrap = document.getElementById('splitViewerWrap');
    const sideWrap = document.getElementById('sideViewerWrap');
    
    btnViewSplit.addEventListener('click', () => {
        btnViewSplit.classList.add('active');
        btnViewSide.classList.remove('active');
        splitWrap.style.display = 'block';
        sideWrap.style.display = 'none';
        VisionState.viewMode = 'split';
    });
    
    btnViewSide.addEventListener('click', () => {
        btnViewSide.classList.add('active');
        btnViewSplit.classList.remove('active');
        splitWrap.style.display = 'none';
        sideWrap.style.display = 'grid';
        VisionState.viewMode = 'side';
    });
    
    // Copy Code Button
    document.getElementById('btnCopyFilterCode').addEventListener('click', () => {
        const code = document.getElementById('filterCodeSnippet').textContent;
        navigator.clipboard.writeText(code);
        showToast('Kode Python berhasil disalin!', 'fa-copy', 'var(--yellow)');
    });
}

// Call Filter API
let filterDebounceTimer = null;
function applyFilterAPI() {
    clearTimeout(filterDebounceTimer);
    filterDebounceTimer = setTimeout(async () => {
        try {
            const res = await fetch('/api/filter', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_src: VisionState.currentImageSrc,
                    filter_type: VisionState.filterType,
                    noise_type: VisionState.noiseType,
                    noise_intensity: VisionState.noiseIntensity,
                    ksize: VisionState.ksize,
                    sigma: VisionState.sigma,
                    sigma_color: VisionState.sigmaColor,
                    sigma_space: VisionState.sigmaSpace
                })
            });
            
            const data = await res.json();
            
            // Update Images
            const beforeSrc = (VisionState.noiseType !== 'none') ? data.noisy_image : data.original_image;
            document.getElementById('splitImgOriginal').src = beforeSrc;
            document.getElementById('splitImgProcessed').src = data.processed_image;
            document.getElementById('sideImgOriginal').src = beforeSrc;
            document.getElementById('sideImgProcessed').src = data.processed_image;
            
            // Update Metrics
            document.getElementById('metricPsnr').textContent = `${data.metrics.psnr} dB`;
            document.getElementById('metricSsim').textContent = data.metrics.ssim;
            document.getElementById('metricMse').textContent = data.metrics.mse;
            document.getElementById('metricLatency').textContent = `${data.execution_time_ms} ms`;
            
            // Update Code Snippet
            const codeEl = document.getElementById('filterCodeSnippet');
            codeEl.textContent = data.code_snippet;
            hljs.highlightElement(codeEl);
            
            // Draw Histogram
            drawHistogram(data.orig_histogram, data.proc_histogram);
            
        } catch (err) {
            console.error('Filter API error:', err);
        }
    }, 80);
}

// Interactive Split Slider Viewer
function initSplitSlider() {
    const viewer = document.getElementById('splitViewer');
    const overlay = document.getElementById('splitOverlay');
    const handle = document.getElementById('splitHandle');
    let isDragging = false;
    
    function setPosition(x) {
        const rect = viewer.getBoundingClientRect();
        let offsetX = x - rect.left;
        offsetX = Math.max(0, Math.min(offsetX, rect.width));
        const pct = (offsetX / rect.width) * 100;
        
        overlay.style.width = `${pct}%`;
        handle.style.left = `${pct}%`;
    }
    
    handle.addEventListener('mousedown', () => isDragging = true);
    window.addEventListener('mouseup', () => isDragging = false);
    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        setPosition(e.clientX);
    });
    
    // Touch support for mobile/tablets
    handle.addEventListener('touchstart', () => isDragging = true);
    window.addEventListener('touchend', () => isDragging = false);
    window.addEventListener('touchmove', (e) => {
        if (!isDragging || !e.touches[0]) return;
        setPosition(e.touches[0].clientX);
    });
}

// Draw Canvas Histogram
function drawHistogram(histBefore, histAfter) {
    const canvas = document.getElementById('histCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    
    ctx.clearRect(0, 0, w, h);
    
    const maxVal = Math.max(...histBefore, ...histAfter, 1);
    const barWidth = w / 256;
    
    // Draw Input/Before (Cyan with opacity)
    ctx.fillStyle = 'rgba(6, 182, 212, 0.4)';
    for (let i = 0; i < 256; i++) {
        const val = (histBefore[i] / maxVal) * (h - 10);
        ctx.fillRect(i * barWidth, h - val, barWidth, val);
    }
    
    // Draw Processed/After (Emerald stroke/fill)
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < 256; i++) {
        const val = (histAfter[i] / maxVal) * (h - 10);
        const x = i * barWidth;
        const y = h - val;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}
