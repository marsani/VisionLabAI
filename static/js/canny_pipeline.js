/**
 * VisionLab AI - Edge Detection & Canny 5-Stage Pipeline
 * Module 3 & Module 4 interactive logic
 */

const EdgeState = {
    operator: 'sobel_mag',
    preBlur: 3,
    ksize: 3,
    scale: 1.0,
    threshold: 0
};

const CannyState = {
    lowThresh: 50,
    highThresh: 150,
    gaussKsize: 5,
    gaussSigma: 1.4
};

document.addEventListener('DOMContentLoaded', () => {
    initEdgeDetectionModule();
    initCannyPipelineModule();
});

/* ================= MODULE 3: EDGE DETECTION ================= */
function initEdgeDetectionModule() {
    // Operator Pills
    const pills = document.querySelectorAll('#edgeOperatorPills .pill');
    pills.forEach(pill => {
        pill.addEventListener('click', () => {
            pills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            EdgeState.operator = pill.getAttribute('data-op');
            
            // Show/hide angle map for non-Sobel operators
            const angleCard = document.getElementById('angleMapCard');
            angleCard.style.display = (EdgeState.operator === 'sobel_mag') ? 'block' : 'none';
            
            applyEdgeDetectionAPI();
        });
    });
    
    // Pre-Blur Slider
    const preBlurSlider = document.getElementById('edgePreBlur');
    const preBlurVal = document.getElementById('edgePreBlurVal');
    preBlurSlider.addEventListener('input', (e) => {
        let val = parseInt(e.target.value);
        if (val % 2 === 0) val += 1;
        EdgeState.preBlur = val;
        preBlurVal.textContent = `${val} × ${val}`;
        applyEdgeDetectionAPI();
    });
    
    // Ksize Slider
    const ksizeSlider = document.getElementById('edgeKsize');
    const ksizeVal = document.getElementById('edgeKsizeVal');
    ksizeSlider.addEventListener('input', (e) => {
        let val = parseInt(e.target.value);
        if (val % 2 === 0) val += 1;
        EdgeState.ksize = val;
        ksizeVal.textContent = val;
        applyEdgeDetectionAPI();
    });
    
    // Scale Slider
    const scaleSlider = document.getElementById('edgeScale');
    const scaleVal = document.getElementById('edgeScaleVal');
    scaleSlider.addEventListener('input', (e) => {
        EdgeState.scale = parseFloat(e.target.value);
        scaleVal.textContent = EdgeState.scale.toFixed(1);
        applyEdgeDetectionAPI();
    });
    
    // Threshold Slider
    const threshSlider = document.getElementById('edgeThreshold');
    const threshVal = document.getElementById('edgeThresholdVal');
    threshSlider.addEventListener('input', (e) => {
        EdgeState.threshold = parseInt(e.target.value);
        threshVal.textContent = (EdgeState.threshold === 0) ? '0 (Off)' : EdgeState.threshold;
        applyEdgeDetectionAPI();
    });
    
    // Copy Code Button
    document.getElementById('btnCopyEdgeCode').addEventListener('click', () => {
        const code = document.getElementById('edgeCodeSnippet').textContent;
        navigator.clipboard.writeText(code);
        showToast('Kode Deteksi Tepi berhasil disalin!', 'fa-copy', 'var(--yellow)');
    });
}

let edgeDebounceTimer = null;
async function applyEdgeDetectionAPI() {
    clearTimeout(edgeDebounceTimer);
    edgeDebounceTimer = setTimeout(async () => {
        try {
            const res = await fetch('/api/edge_detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_src: VisionState.currentImageSrc,
                    operator: EdgeState.operator,
                    blur_ksize: EdgeState.preBlur,
                    ksize: EdgeState.ksize,
                    scale: EdgeState.scale,
                    threshold: EdgeState.threshold
                })
            });
            
            const data = await res.json();
            document.getElementById('edgeResultImg').src = data.processed_image;
            document.getElementById('edgeDensityVal').textContent = `${data.edge_density}%`;
            
            if (data.angle_map) {
                document.getElementById('edgeAngleMapImg').src = data.angle_map;
            }
            
            const codeEl = document.getElementById('edgeCodeSnippet');
            codeEl.textContent = data.code_snippet;
            hljs.highlightElement(codeEl);
            
        } catch (err) {
            console.error('Edge Detection error:', err);
        }
    }, 80);
}

/* ================= MODULE 4: CANNY 5-STAGE PIPELINE ================= */
function initCannyPipelineModule() {
    // Low Threshold Slider
    const lowSlider = document.getElementById('cannyLowThresh');
    const lowVal = document.getElementById('cannyLowVal');
    lowSlider.addEventListener('input', (e) => {
        CannyState.lowThresh = parseInt(e.target.value);
        lowVal.textContent = CannyState.lowThresh;
        
        // Ensure low <= high
        if (CannyState.lowThresh > CannyState.highThresh) {
            CannyState.highThresh = CannyState.lowThresh;
            document.getElementById('cannyHighThresh').value = CannyState.highThresh;
            document.getElementById('cannyHighVal').textContent = CannyState.highThresh;
        }
        applyCannyPipelineAPI();
    });
    
    // High Threshold Slider
    const highSlider = document.getElementById('cannyHighThresh');
    const highVal = document.getElementById('cannyHighVal');
    highSlider.addEventListener('input', (e) => {
        CannyState.highThresh = parseInt(e.target.value);
        highVal.textContent = CannyState.highThresh;
        
        // Ensure high >= low
        if (CannyState.highThresh < CannyState.lowThresh) {
            CannyState.lowThresh = CannyState.highThresh;
            document.getElementById('cannyLowThresh').value = CannyState.lowThresh;
            document.getElementById('cannyLowVal').textContent = CannyState.lowThresh;
        }
        applyCannyPipelineAPI();
    });
    
    // Gauss Ksize
    const ksizeSlider = document.getElementById('cannyGaussKsize');
    const ksizeVal = document.getElementById('cannyGaussKsizeVal');
    ksizeSlider.addEventListener('input', (e) => {
        let val = parseInt(e.target.value);
        if (val % 2 === 0) val += 1;
        CannyState.gaussKsize = val;
        ksizeVal.textContent = `${val} × ${val}`;
        applyCannyPipelineAPI();
    });
    
    // Gauss Sigma
    const sigmaSlider = document.getElementById('cannyGaussSigma');
    const sigmaVal = document.getElementById('cannyGaussSigmaVal');
    sigmaSlider.addEventListener('input', (e) => {
        CannyState.gaussSigma = parseFloat(e.target.value);
        sigmaVal.textContent = CannyState.gaussSigma.toFixed(1);
        applyCannyPipelineAPI();
    });
    
    // Copy Code Button
    document.getElementById('btnCopyCannyCode').addEventListener('click', () => {
        const code = document.getElementById('cannyCodeSnippet').textContent;
        navigator.clipboard.writeText(code);
        showToast('Kode Canny Edge Detection berhasil disalin!', 'fa-copy', 'var(--yellow)');
    });
}

let cannyDebounceTimer = null;
async function applyCannyPipelineAPI() {
    clearTimeout(cannyDebounceTimer);
    cannyDebounceTimer = setTimeout(async () => {
        try {
            const res = await fetch('/api/canny_pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_src: VisionState.currentImageSrc,
                    low_threshold: CannyState.lowThresh,
                    high_threshold: CannyState.highThresh,
                    gauss_ksize: CannyState.gaussKsize,
                    gauss_sigma: CannyState.gaussSigma
                })
            });
            
            const data = await res.json();
            
            // Populate 5-stages visual cards
            document.getElementById('cannyStage1').src = data.stage1_gray;
            document.getElementById('cannyStage2').src = data.stage2_blur;
            document.getElementById('cannyStage3').src = data.stage3_mag;
            document.getElementById('cannyStage4').src = data.stage4_nms;
            document.getElementById('cannyStage5').src = data.stage5_final;
            
            // Populate statistics
            document.getElementById('statStrongEdges').textContent = `${data.strong_edges_count} px`;
            document.getElementById('statWeakEdges').textContent = `${data.weak_edges_count} px`;
            document.getElementById('statFinalEdges').textContent = `${data.final_edge_pixels} px`;
            
            const codeEl = document.getElementById('cannyCodeSnippet');
            codeEl.textContent = data.code_snippet;
            hljs.highlightElement(codeEl);
            
        } catch (err) {
            console.error('Canny pipeline error:', err);
        }
    }, 80);
}
