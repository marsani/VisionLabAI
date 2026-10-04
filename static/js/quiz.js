/**
 * VisionLab AI - Case Studies & Quiz Evaluation
 * Module 5 & Module 7 interactive client logic
 */

let QuizQuestions = [];
let UserAnswers = {};

document.addEventListener('DOMContentLoaded', () => {
    initCaseStudiesNav();
    initQuizModule();
});

/* ================= MODULE 5: COMPUTER VISION CASE STUDIES ================= */
function initCaseStudiesNav() {
    const caseButtons = document.querySelectorAll('.case-tab-btn');
    caseButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            caseButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const caseType = btn.getAttribute('data-case');
            loadCaseStudy(caseType);
        });
    });
}

async function loadCaseStudy(caseType) {
    try {
        const res = await fetch('/api/cv_case_study', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ case_type: caseType })
        });
        
        const data = await res.json();
        
        document.getElementById('caseTitle').textContent = data.title;
        document.getElementById('caseDescription').textContent = data.description;
        
        const flowContainer = document.getElementById('caseStepsFlow');
        flowContainer.innerHTML = '';
        
        data.steps.forEach(step => {
            const card = document.createElement('div');
            card.className = 'case-step-card';
            card.innerHTML = `
                <h5>${step.name}</h5>
                <p>${step.info}</p>
                <div class="case-step-img-wrap">
                    <img src="${step.image}" alt="${step.name}">
                </div>
            `;
            flowContainer.appendChild(card);
        });
        
    } catch (err) {
        console.error('Case study load error:', err);
    }
}

/* ================= MODULE 7: QUIZ & ASSESSMENT ================= */
function initQuizModule() {
    document.getElementById('btnSubmitQuiz').addEventListener('click', evaluateQuiz);
    document.getElementById('btnResetQuiz').addEventListener('click', resetQuiz);
}

async function loadQuizData() {
    if (QuizQuestions.length > 0) return; // already loaded
    
    try {
        const res = await fetch('/api/quiz');
        QuizQuestions = await res.json();
        renderQuizQuestions();
    } catch (err) {
        console.error('Quiz loading error:', err);
    }
}

function renderQuizQuestions() {
    const container = document.getElementById('quizQuestionsContainer');
    container.innerHTML = '';
    
    QuizQuestions.forEach((q, idx) => {
        const item = document.createElement('div');
        item.className = 'quiz-item';
        item.id = `quiz-item-${q.id}`;
        
        const optionsHtml = q.options.map((opt, optIdx) => `
            <label class="quiz-opt-label" data-qid="${q.id}" data-opt="${optIdx}">
                <input type="radio" name="question_${q.id}" value="${optIdx}">
                <span>${opt}</span>
            </label>
        `).join('');
        
        item.innerHTML = `
            <div class="quiz-item-header">
                <span class="q-num">Soal ${idx + 1}</span>
                <span class="q-text">${q.question}</span>
            </div>
            <div class="quiz-options">
                ${optionsHtml}
            </div>
            <div class="quiz-explanation" id="quiz-exp-${q.id}">
                <strong><i class="fa-solid fa-lightbulb"></i> Pembahasan:</strong>
                <p>${q.explanation}</p>
            </div>
        `;
        
        container.appendChild(item);
    });
    
    // Add option select listeners
    const optLabels = document.querySelectorAll('.quiz-opt-label');
    optLabels.forEach(label => {
        label.addEventListener('click', () => {
            const qid = parseInt(label.getAttribute('data-qid'));
            const optIdx = parseInt(label.getAttribute('data-opt'));
            
            // Remove selected class from sibling labels
            const siblingLabels = document.querySelectorAll(`.quiz-opt-label[data-qid="${qid}"]`);
            siblingLabels.forEach(l => l.classList.remove('selected'));
            
            label.classList.add('selected');
            const radio = label.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
            
            UserAnswers[qid] = optIdx;
        });
    });
    
    // Re-render MathJax formulas if available
    if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise();
    }
}

function evaluateQuiz() {
    if (Object.keys(UserAnswers).length === 0) {
        showToast('Silakan pilih jawaban Anda sebelum mengumpulkan kuis.', 'fa-triangle-exclamation', 'var(--rose)');
        return;
    }
    
    let score = 0;
    
    QuizQuestions.forEach(q => {
        const userChoice = UserAnswers[q.id];
        const isCorrect = (userChoice === q.answer);
        
        if (isCorrect) score++;
        
        // Highlight options
        const labels = document.querySelectorAll(`.quiz-opt-label[data-qid="${q.id}"]`);
        labels.forEach(l => {
            const optIdx = parseInt(l.getAttribute('data-opt'));
            l.classList.remove('correct', 'wrong');
            
            if (optIdx === q.answer) {
                l.classList.add('correct');
            } else if (optIdx === userChoice && !isCorrect) {
                l.classList.add('wrong');
            }
        });
        
        // Reveal explanation
        const expEl = document.getElementById(`quiz-exp-${q.id}`);
        if (expEl) expEl.style.display = 'block';
    });
    
    const badge = document.getElementById('quizScoreBadge');
    badge.textContent = `Skor: ${score} / ${QuizQuestions.length} (${Math.round((score / QuizQuestions.length) * 100)}%)`;
    
    if (score >= 8) {
        showToast(`Hebat! Skor Anda: ${score}/${QuizQuestions.length} (Sangat Memuaskan)`, 'fa-trophy', 'var(--yellow)');
    } else {
        showToast(`Skor Anda: ${score}/${QuizQuestions.length}. Pelajari kembali pembahasan di setiap soal.`, 'fa-circle-info', 'var(--cyan)');
    }
    
    // Re-render MathJax
    if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise();
    }
}

function resetQuiz() {
    UserAnswers = {};
    const radios = document.querySelectorAll('.quiz-opt-label input[type="radio"]');
    radios.forEach(r => r.checked = false);
    
    const labels = document.querySelectorAll('.quiz-opt-label');
    labels.forEach(l => l.classList.remove('selected', 'correct', 'wrong'));
    
    const exps = document.querySelectorAll('.quiz-explanation');
    exps.forEach(e => e.style.display = 'none');
    
    const badge = document.getElementById('quizScoreBadge');
    badge.textContent = `Skor: 0 / ${QuizQuestions.length}`;
    
    showToast('Kuis direset. Silakan jawab kembali!');
}
