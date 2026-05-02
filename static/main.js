/**
 * SignBridge Pro - ML Edition
 * Frontend with Backend ML Integration
 * Same UI, Real Predictions
 */

console.log('%c SignBridge Pro ', 'background: linear-gradient(135deg, #6366f1, #ec4899); color: white; font-size: 24px; font-weight: bold; padding: 10px 20px; border-radius: 10px;');
console.log('%c ML Edition - Real-time ASL Translation ', 'color: #fbbf24; font-size: 14px; font-weight: 600;');

document.addEventListener('DOMContentLoaded', function() {

    // ===== PARTICLE SYSTEM =====
    function initParticles() {
        const container = document.getElementById('particlesContainer');
        if (!container) return;

        const colors = ['#6366f1', '#ec4899', '#06b6d4', '#10b981', '#f59e0b', '#8b5cf6'];
        const particleCount = 30;

        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            const size = Math.random() * 6 + 2;
            const color = colors[Math.floor(Math.random() * colors.length)];
            const left = Math.random() * 100;
            const duration = Math.random() * 15 + 10;
            const delay = Math.random() * 10;

            particle.style.cssText = `
                width: ${size}px;
                height: ${size}px;
                background: ${color};
                left: ${left}%;
                animation-duration: ${duration}s;
                animation-delay: ${delay}s;
                box-shadow: 0 0 ${size * 2}px ${color};
            `;
            container.appendChild(particle);
        }
    }

    // ===== CURSOR GLOW EFFECT =====
    function initCursorGlow() {
        const cursor = document.getElementById('cursorGlow');
        if (!cursor || window.matchMedia('(pointer: coarse)').matches) {
            if (cursor) cursor.style.display = 'none';
            return;
        }

        let mouseX = 0, mouseY = 0;
        let currentX = 0, currentY = 0;

        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
        });

        function animateCursor() {
            currentX += (mouseX - currentX) * 0.1;
            currentY += (mouseY - currentY) * 0.1;
            cursor.style.left = currentX + 'px';
            cursor.style.top = currentY + 'px';
            requestAnimationFrame(animateCursor);
        }
        animateCursor();
    }

    // ===== SCROLL REVEAL =====
    function initScrollReveal() {
        const reveals = document.querySelectorAll('.reveal');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('active');
                }
            });
        }, { threshold: 0.1 });

        reveals.forEach(el => observer.observe(el));
    }

    // ===== SMOOTH SCROLL FOR NAV LINKS =====
    function initSmoothScroll() {
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
                    link.classList.add('active');
                }
            });
        });
    }

    // ===== DOM ELEMENTS =====
    const els = {
        video: document.getElementById('inputVideo'),
        canvas: document.getElementById('outputCanvas'),
        startBtn: document.getElementById('startBtn'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        progressFill: document.getElementById('progressFill'),
        startOverlay: document.getElementById('startOverlay'),
        currentPrediction: document.getElementById('currentPrediction'),
        confidenceFill: document.getElementById('confidenceFill'),
        confidenceText: document.getElementById('confidenceText'),
        topPredictions: document.getElementById('topPredictions'),
        translationOutput: document.getElementById('translationOutput'),
        detectionValue: document.getElementById('detectionValue'),
        alphabetGrid: document.getElementById('alphabetGrid'),
        sentencesGrid: document.getElementById('sentencesGrid'),
        textInput: document.getElementById('textInput'),
        convertBtn: document.getElementById('convertBtn'),
        speakTextBtn: document.getElementById('speakTextBtn'),
        signDisplay: document.getElementById('signDisplay'),
        settingsModal: document.getElementById('settingsModal'),
        thresholdRange: document.getElementById('thresholdRange'),
        thresholdValue: document.getElementById('thresholdValue'),
        cooldownRange: document.getElementById('cooldownRange'),
        cooldownValue: document.getElementById('cooldownValue'),
        toastContainer: document.getElementById('toastContainer'),
        skeletonToggle: document.getElementById('skeletonToggle'),
        autoSpeakToggle: document.getElementById('autoSpeakToggle')
    };

    if (!els.video || !els.canvas) {
        console.error('Critical elements missing!');
        showToast('Application error: Critical elements missing', 'error');
        return;
    }

    // ===== APPLICATION STATE =====
    const state = {
        isRunning: false,
        hands: null,
        currentMode: 'letters',
        currentWord: '',
        detectedLetters: [],
        lastLetter: '',
        lastDetectionTime: 0,
        stableCount: 0,
        confidenceThreshold: 0.7,
        showSkeleton: true,
        autoSpeak: false,
        letterCooldown: 1500,
        isSpeaking: false,
        lastDetectedSentence: '',
        sentenceCooldown: 3000,
        // ML Backend state
        mlBackend: true,
        apiUrl: window.location.origin,
        modelLoaded: false,
        frameBuffer: [],
        bufferSize: 5,
        isProcessing: false
    };

    const speechSynth = window.speechSynthesis;
    let voices = [];

    // ===== ENHANCED LETTER DATA =====
    const letterData = {
        'A': { name: 'Fist', desc: 'Close all fingers into a fist, thumb rests on side', emoji: '✊', color: '#6366f1' },
        'B': { name: 'Open Hand', desc: 'All fingers extended upward, thumb tucked', emoji: '🖐️', color: '#ec4899' },
        'C': { name: 'C Shape', desc: 'Curved hand forming letter C', emoji: 'C', color: '#06b6d4' },
        'D': { name: 'Point Up', desc: 'Index finger pointing up, others closed', emoji: '☝️', color: '#10b981' },
        'E': { name: 'Claw', desc: 'All fingers curled down like a claw', emoji: '🦅', color: '#f59e0b' },
        'F': { name: 'F', desc: 'Index and thumb touch, others extended', emoji: '🤙', color: '#8b5cf6' },
        'G': { name: 'Point Side', desc: 'Index finger pointing sideways', emoji: '👉', color: '#ef4444' },
        'H': { name: 'Two Fingers', desc: 'Index and middle fingers side by side', emoji: '✌️', color: '#14b8a6' },
        'I': { name: 'Pinky', desc: 'Pinky finger raised alone', emoji: '🤙', color: '#f97316' },
        'J': { name: 'J Motion', desc: 'I shape with J writing motion', emoji: 'J', color: '#84cc16' },
        'K': { name: 'V Up', desc: 'Index and middle in V, thumb out', emoji: '✌️', color: '#06b6d4' },
        'L': { name: 'L Shape', desc: 'Index up, thumb out forming L', emoji: 'L', color: '#eab308' },
        'M': { name: 'M', desc: 'Three fingers over thumb', emoji: 'M', color: '#a855f7' },
        'N': { name: 'N', desc: 'Two fingers over thumb', emoji: 'N', color: '#3b82f6' },
        'O': { name: 'O Shape', desc: 'Fingers form circle shape', emoji: 'O', color: '#ef4444' },
        'P': { name: 'P', desc: 'K shape pointing downward', emoji: 'P', color: '#10b981' },
        'Q': { name: 'Q', desc: 'G shape pointing down', emoji: 'Q', color: '#f59e0b' },
        'R': { name: 'Crossed', desc: 'Index and middle fingers crossed', emoji: '🤞', color: '#ec4899' },
        'S': { name: 'S', desc: 'Fist with thumb over fingers', emoji: '✊', color: '#6366f1' },
        'T': { name: 'T', desc: 'Thumb between index and middle', emoji: 'T', color: '#14b8a6' },
        'U': { name: 'U', desc: 'Index and middle together up', emoji: 'U', color: '#8b5cf6' },
        'V': { name: 'Peace', desc: 'Index and middle spread in V', emoji: '✌️', color: '#ec4899' },
        'W': { name: 'W', desc: 'Index, middle, ring fingers up', emoji: 'W', color: '#3b82f6' },
        'X': { name: 'Hook', desc: 'Index bent in hook shape', emoji: 'X', color: '#f97316' },
        'Y': { name: 'Hang Loose', desc: 'Thumb and pinky out, others closed', emoji: '🤙', color: '#eab308' },
        'Z': { name: 'Z', desc: 'Index draws Z in air', emoji: 'Z', color: '#10b981' },
        'space': { name: 'Space', desc: 'Open palm facing forward', emoji: '␣', color: '#06b6d4' }
    };

    // ===== ENHANCED SENTENCES DATABASE =====
    const sentencesDB = [
        { id: 'hello', text: "Hello", category: "Greetings", emoji: "👋", desc: "Wave your open hand side to side near your face", gesture: "Open hand, all fingers extended, wave motion" },
        { id: 'love', text: "I Love You", category: "Feelings", emoji: "🤟", desc: "Index, pinky and thumb extended (rock sign)", gesture: "I-L-Y hand shape: thumb, index, and pinky out" },
        { id: 'awesome', text: "This is Awesome", category: "Reactions", emoji: "👍", desc: "Thumbs up with other fingers closed", gesture: "Thumb pointing up, all other fingers closed into palm" },
        { id: 'you', text: "You", category: "Pointing", emoji: "👉", desc: "Point directly at the person with index finger", gesture: "Index finger extended pointing forward, others closed" },
        { id: 'ok', text: "OK", category: "Agreement", emoji: "👌", desc: "Thumb and index finger form circle, others extended", gesture: "Thumb and index touch forming O, other fingers up" },
        { id: 'goodjob', text: "Good Job", category: "Praise", emoji: "👍", desc: "Thumbs up gesture with smile", gesture: "Thumb up, move slightly up and down" }
    ];

    // ===== INITIALIZATION =====
    function init() {
        console.log('Initializing SignBridge Pro ML Edition...');
        initParticles();
        initCursorGlow();
        initScrollReveal();
        initSmoothScroll();

        let progress = 0;
        const loadInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 100) {
                progress = 100;
                clearInterval(loadInterval);
                setTimeout(() => {
                    els.loadingOverlay.classList.add('hidden');
                }, 500);
            }
            if (els.progressFill) els.progressFill.style.width = progress + '%';
        }, 200);

        setupGuide();
        setupSentences();
        setupEventListeners();
        initSpeech();
        checkBackendStatus();

        console.log('Initialization complete!');
        showToast('Welcome to SignBridge Pro ML! 🎉', 'success');
    }

    function initSpeech() {
        if (!speechSynth) {
            console.warn('Speech synthesis not supported');
            return;
        }
        const loadVoices = () => {
            voices = speechSynth.getVoices();
            console.log(`Loaded ${voices.length} voices`);
        };
        loadVoices();
        if (speechSynthesis.onvoiceschanged !== undefined) {
            speechSynthesis.onvoiceschanged = loadVoices;
        }
    }

    // ===== CHECK BACKEND STATUS =====
    async function checkBackendStatus() {
        try {
            const response = await fetch(`${state.apiUrl}/api/status`);
            const data = await response.json();
            state.modelLoaded = data.model_loaded;

            if (data.model_loaded) {
                showToast('🤖 ML Model loaded! Real-time predictions active.', 'success');
                console.log('✅ ML Backend connected - Model loaded');
            } else {
                showToast('⚠️ Running in demo mode. Train model for real predictions.', 'warning');
                console.log('⚠️ ML Backend connected - Model not loaded');
            }
        } catch (error) {
            console.warn('Backend not available:', error);
            state.mlBackend = false;
            showToast('⚠️ Backend not available. Using local detection.', 'warning');
        }
    }

    // ===== MEDIAPIPE SETUP =====
    function initMediaPipe() {
        if (typeof Hands === 'undefined') {
            console.log('Waiting for MediaPipe...');
            setTimeout(initMediaPipe, 500);
            return;
        }
        try {
            state.hands = new Hands({locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }});
            state.hands.setOptions({
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            state.hands.onResults(onResults);
            console.log('✅ MediaPipe Hands initialized');
        } catch (e) {
            console.error('MediaPipe initialization error:', e);
            showToast('Failed to initialize hand tracking', 'error');
        }
    }

    // ===== CAPTURE FRAME FOR ML =====
    function captureFrame() {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const ctx = tempCanvas.getContext('2d');

        // Draw video frame (grayscale, centered on hand)
        ctx.filter = 'grayscale(100%)';
        ctx.drawImage(els.video, 0, 0, 28, 28);

        return tempCanvas.toDataURL('image/jpeg', 0.8);
    }

    // ===== SEND TO BACKEND =====
    async function predictWithBackend(imageData) {
        try {
            const response = await fetch(`${state.apiUrl}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Backend prediction error:', error);
            return null;
        }
    }

    // ===== BATCH PREDICTION =====
    async function predictBatch(frames) {
        try {
            const response = await fetch(`${state.apiUrl}/api/batch_predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ images: frames })
            });

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Batch prediction error:', error);
            return null;
        }
    }

    function onResults(results) {
        const ctx = els.canvas.getContext('2d');
        if (els.video.videoWidth && els.canvas.width !== els.video.videoWidth) {
            els.canvas.width = els.video.videoWidth;
            els.canvas.height = els.video.videoHeight;
        }
        ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
        ctx.save();
        ctx.translate(els.canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            if (state.showSkeleton) drawSkeleton(ctx, landmarks);

            // ML Backend Prediction
            if (state.mlBackend && state.modelLoaded && !state.isProcessing) {
                // Capture frame and add to buffer
                const frame = captureFrame();
                state.frameBuffer.push(frame);

                if (state.frameBuffer.length >= state.bufferSize) {
                    processFrameBuffer();
                }
            } else {
                // Fallback to local detection
                if (state.currentMode === 'letters') {
                    const detection = detectLetter(landmarks);
                    handleDetection(detection);
                } else {
                    const sentenceDetection = detectSentence(landmarks);
                    handleSentenceDetection(sentenceDetection);
                }
            }
        } else {
            updateUI('-', 0, []);
            if (els.detectionValue) els.detectionValue.textContent = '-';
            state.frameBuffer = []; // Clear buffer when no hand
        }
        ctx.restore();
    }

    async function processFrameBuffer() {
        state.isProcessing = true;

        // Use batch prediction for stability
        const result = await predictBatch([...state.frameBuffer]);
        state.frameBuffer = []; // Clear buffer

        if (result && result.success) {
            const letter = result.prediction;
            const confidence = result.confidence;
            const top3 = result.all_results ? 
                result.all_results[0]?.top3 || [] : [];

            // Create detection object compatible with existing handlers
            const detection = {
                letter: letter,
                confidence: confidence,
                alternatives: top3.slice(1).map(t => ({ letter: t.letter, conf: t.confidence })),
                type: 'letter'
            };

            handleDetection(detection);
        }

        state.isProcessing = false;
    }

    function drawSkeleton(ctx, landmarks) {
        const w = els.canvas.width;
        const h = els.canvas.height;
        const connections = [
            [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
            [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
            [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]
        ];
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#00ff88';
        for (const [start, end] of connections) {
            const p1 = landmarks[start];
            const p2 = landmarks[end];
            ctx.beginPath();
            ctx.moveTo(p1.x * w, p1.y * h);
            ctx.lineTo(p2.x * w, p2.y * h);
            ctx.stroke();
        }
        ctx.shadowBlur = 15;
        for (let i = 0; i < landmarks.length; i++) {
            const p = landmarks[i];
            ctx.beginPath();
            ctx.arc(p.x * w, p.y * h, i % 4 === 0 ? 8 : 5, 0, 2 * Math.PI);
            ctx.fillStyle = i === 0 ? '#00ffff' : '#ff0066';
            ctx.fill();
        }
        ctx.shadowBlur = 0;
    }

    // ===== LETTER DETECTION (Local Fallback) =====
    function detectLetter(landmarks) {
        const wrist = landmarks[0];
        function getFinger(tipIdx, pipIdx, mcpIdx) {
            const tip = landmarks[tipIdx];
            const pip = landmarks[pipIdx];
            const mcp = landmarks[mcpIdx];
            const tipDist = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
            const pipDist = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);
            return {
                extended: tipDist > pipDist * 1.15,
                tip: tip, pip: pip, mcp: mcp,
                x: tip.x, y: tip.y
            };
        }
        const thumb = getFinger(4, 3, 2);
        const index = getFinger(8, 6, 5);
        const middle = getFinger(12, 10, 9);
        const ring = getFinger(16, 14, 13);
        const pinky = getFinger(20, 18, 17);
        const fingers = [index, middle, ring, pinky];
        const extendedCount = fingers.filter(f => f.extended).length;
        const upCount = fingers.filter(f => f.extended && f.y < wrist.y - 0.05).length;
        const indexMiddleSpread = Math.hypot(index.x - middle.x, index.y - middle.y);

        let letter = '-';
        let confidence = 0;

        if (extendedCount === 0 && !thumb.extended) { letter = 'A'; confidence = 0.95; }
        else if (upCount === 4 && !thumb.extended) { letter = 'B'; confidence = 0.95; }
        else if (extendedCount === 4 && thumb.extended && index.y > wrist.y - 0.1 && index.y < wrist.y + 0.1) { letter = 'C'; confidence = 0.85; }
        else if (index.extended && !middle.extended && !ring.extended && !pinky.extended) {
            if (index.y < wrist.y) { letter = 'D'; confidence = 0.92; }
        }
        else if (extendedCount === 0 && thumb.extended) {
            const avgTipY = (index.tip.y + middle.tip.y + ring.tip.y + pinky.tip.y) / 4;
            if (avgTipY > wrist.y) { letter = 'E'; confidence = 0.88; }
        }
        else if (!index.extended && middle.extended && ring.extended && pinky.extended) {
            const thumbIndexDist = Math.hypot(thumb.x - index.x, thumb.y - index.y);
            if (thumbIndexDist < 0.1) { letter = 'F'; confidence = 0.85; }
        }
        else if (index.extended && Math.abs(index.y - wrist.y) < 0.1 && !middle.extended && !ring.extended && !pinky.extended) { letter = 'G'; confidence = 0.88; }
        else if (index.extended && middle.extended && !ring.extended && !pinky.extended && indexMiddleSpread < 0.08) { letter = 'H'; confidence = 0.87; }
        else if (!index.extended && !middle.extended && !ring.extended && pinky.extended) { letter = 'I'; confidence = 0.92; }
        else if (index.extended && middle.extended && !ring.extended && !pinky.extended && thumb.extended) {
            if (indexMiddleSpread > 0.06) { letter = 'K'; confidence = 0.88; }
        }
        else if (index.extended && !middle.extended && !ring.extended && !pinky.extended && thumb.extended) {
            if (Math.abs(thumb.x - index.x) > 0.05) { letter = 'L'; confidence = 0.90; }
        }
        else if (!index.extended && !middle.extended && !ring.extended && !pinky.extended && thumb.extended) {
            const tipsToThumb = [index, middle, ring, pinky].every(f => {
                const d = Math.hypot(f.tip.x - thumb.x, f.tip.y - thumb.y);
                return d < 0.15;
            });
            if (tipsToThumb) { letter = 'O'; confidence = 0.87; }
        }
        else if (index.extended && middle.extended && !ring.extended && !pinky.extended) {
            if (Math.abs(index.x - middle.x) < 0.03) { letter = 'R'; confidence = 0.85; }
        }
        else if (extendedCount === 0 && thumb.extended && thumb.x > wrist.x) { letter = 'S'; confidence = 0.88; }
        else if (upCount === 2 && !ring.extended && !pinky.extended) {
            if (indexMiddleSpread < 0.05) { letter = 'U'; confidence = 0.90; }
            else { letter = 'V'; confidence = 0.92; }
        }
        else if (upCount === 3 && !pinky.extended) { letter = 'W'; confidence = 0.90; }
        else if (!index.extended && !middle.extended && !ring.extended && pinky.extended && thumb.extended) { letter = 'Y'; confidence = 0.90; }

        const alternatives = Object.keys(letterData)
            .filter(l => l !== letter)
            .map(l => ({ letter: l, conf: Math.random() * 0.4 + 0.2 }))
            .sort((a, b) => b.conf - a.conf)
            .slice(0, 2);

        return { letter, confidence, alternatives, type: 'letter' };
    }

    // ===== SENTENCE DETECTION =====
    function detectSentence(landmarks) {
        const wrist = landmarks[0];
        function getFinger(tipIdx, pipIdx) {
            const tip = landmarks[tipIdx];
            const pip = landmarks[pipIdx];
            const tipDist = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
            const pipDist = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);
            return { extended: tipDist > pipDist * 1.2, tip: tip, x: tip.x, y: tip.y };
        }
        const thumb = getFinger(4, 2);
        const index = getFinger(8, 6);
        const middle = getFinger(12, 10);
        const ring = getFinger(16, 14);
        const pinky = getFinger(20, 18);
        const fingers = [index, middle, ring, pinky];
        const extendedFingers = fingers.filter(f => f.extended);
        const extendedCount = extendedFingers.length;
        const pointingUp = extendedFingers.every(f => f.y < wrist.y);

        let detectedId = null;
        let confidence = 0;
        let sentenceText = '';

        if (extendedCount === 4 && thumb.extended && pointingUp) { detectedId = 'hello'; confidence = 0.90; sentenceText = 'Hello'; }
        else if (index.extended && !middle.extended && !ring.extended && pinky.extended && thumb.extended) { detectedId = 'love'; confidence = 0.95; sentenceText = 'I Love You'; }
        else if (!index.extended && !middle.extended && !ring.extended && !pinky.extended && thumb.extended && thumb.y < wrist.y) { detectedId = 'awesome'; confidence = 0.92; sentenceText = 'This is Awesome'; }
        else if (index.extended && !middle.extended && !ring.extended && !pinky.extended && !thumb.extended) { detectedId = 'you'; confidence = 0.93; sentenceText = 'You'; }
        else if (index.extended && middle.extended && ring.extended && pinky.extended && thumb.extended) {
            const thumbIndexDist = Math.hypot(thumb.x - index.x, thumb.y - index.y);
            if (thumbIndexDist < 0.08) { detectedId = 'ok'; confidence = 0.88; sentenceText = 'OK'; }
        }

        return { id: detectedId, text: sentenceText, confidence: confidence, type: 'sentence' };
    }

    // ===== DETECTION HANDLING =====
    function handleDetection(detection) {
        const { letter, confidence } = detection;
        const now = Date.now();
        updateUI(letter, confidence, detection.alternatives);
        if (els.detectionValue) els.detectionValue.textContent = letter;

        if (letter !== '-' && letter === state.lastLetter) {
            if (now - state.lastDetectionTime < state.letterCooldown) return;
        }
        if (letter !== '-') {
            if (letter === state.lastLetter) { state.stableCount++; }
            else { state.stableCount = 1; state.lastLetter = letter; }
            if (state.stableCount >= 3 && confidence > state.confidenceThreshold) {
                addLetter(letter);
                state.lastDetectionTime = now;
                state.stableCount = 0;
            }
        }
    }

    function handleSentenceDetection(detection) {
        const { id, text, confidence } = detection;
        const now = Date.now();

        if (id) {
            if (els.currentPrediction) {
                els.currentPrediction.textContent = text;
                els.currentPrediction.style.color = '#10b981';
            }
            if (els.confidenceFill) els.confidenceFill.style.width = Math.round(confidence * 100) + '%';
            if (els.confidenceText) els.confidenceText.textContent = `${Math.round(confidence * 100)}% confidence`;
            const sentence = sentencesDB.find(s => s.id === id);
            if (els.detectionValue) els.detectionValue.textContent = sentence?.emoji || '📝';

            if (id === state.lastDetectedSentence) {
                if (now - state.lastDetectionTime < state.sentenceCooldown) return;
            }
            if (id === state.lastDetectedSentence) { state.stableCount++; }
            else { state.stableCount = 1; state.lastDetectedSentence = id; }

            if (state.stableCount >= 3 && confidence > 0.75) {
                state.currentWord = text;
                updateTranslation();
                selectSentenceById(id);
                state.lastDetectionTime = now;
                state.stableCount = 0;
                if (state.autoSpeak && !state.isSpeaking) speakText(text);
                showToast(`Detected: "${text}" ${sentence?.emoji || ''}`, 'success');
            }
        } else {
            if (els.currentPrediction) { els.currentPrediction.textContent = '-'; els.currentPrediction.style.color = '#6366f1'; }
            if (els.confidenceFill) els.confidenceFill.style.width = '0%';
            if (els.confidenceText) els.confidenceText.textContent = '0% confidence';
            if (els.detectionValue) els.detectionValue.textContent = '-';
            state.lastDetectedSentence = '';
            state.stableCount = 0;
        }
    }

    function addLetter(letter) {
        state.currentWord += letter;
        state.detectedLetters.push({ letter: letter, time: Date.now() });
        updateTranslation();
        if (els.currentPrediction) {
            els.currentPrediction.classList.add('detected');
            setTimeout(() => els.currentPrediction.classList.remove('detected'), 500);
        }
        if (els.detectionValue) {
            els.detectionValue.classList.add('detected');
            setTimeout(() => els.detectionValue.classList.remove('detected'), 500);
        }
        if (state.autoSpeak && !state.isSpeaking) speakLetter(letter);
        showToast(`Detected: ${letter}`, 'success');
    }

    function updateTranslation() {
        const text = state.currentWord;
        if (text) {
            const html = text.split('').map((char, i) => {
                if (i === text.length - 1) return `<span class="char-new">${char}</span>`;
                return `<span>${char}</span>`;
            }).join('');
            if (els.translationOutput) els.translationOutput.innerHTML = html;
        } else {
            if (els.translationOutput) els.translationOutput.innerHTML = '<span class="placeholder">Start signing to translate...</span>';
        }
    }

    // ===== UI UPDATES =====
    function updateUI(letter, confidence, alternatives) {
        if (els.currentPrediction) els.currentPrediction.textContent = letter;
        const confPercent = Math.round(confidence * 100);
        if (els.confidenceFill) els.confidenceFill.style.width = Math.max(confPercent, 5) + '%';
        if (els.confidenceText) els.confidenceText.textContent = `${confPercent}% confidence`;

        if (confidence > 0.85) {
            if (els.confidenceFill) els.confidenceFill.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
            if (els.currentPrediction) els.currentPrediction.style.color = '#10b981';
        } else if (confidence > 0.6) {
            if (els.confidenceFill) els.confidenceFill.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
            if (els.currentPrediction) els.currentPrediction.style.color = '#f59e0b';
        } else {
            if (els.confidenceFill) els.confidenceFill.style.background = 'linear-gradient(90deg, #6366f1, #8b5cf6)';
            if (els.currentPrediction) els.currentPrediction.style.color = '#6366f1';
        }

        const allPredictions = [{ letter, conf: confidence }, ...alternatives]
            .filter(p => p.letter !== '-').slice(0, 3);

        if (els.topPredictions) {
            els.topPredictions.innerHTML = allPredictions.map((p, i) => `
                <div class="prediction-row ${i === 0 ? 'winner' : ''}">
                    <div class="rank-badge">${i + 1}</div>
                    <div class="prediction-name">${p.letter}</div>
                    <div class="prediction-prob">${Math.round(p.conf * 100)}%</div>
                </div>
            `).join('');
        }
    }

    // ===== TEXT TO SPEECH =====
    function speakText(text) {
        if (!speechSynth) { showToast('Speech not supported', 'error'); return; }
        speechSynth.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9; utterance.pitch = 1; utterance.volume = 1;
        const preferredVoice = voices.find(v => v.name.includes('Google US English')) ||
                              voices.find(v => v.lang === 'en-US' && v.name.includes('Female')) ||
                              voices.find(v => v.lang === 'en-US') || voices[0];
        if (preferredVoice) utterance.voice = preferredVoice;
        utterance.onstart = () => { state.isSpeaking = true; showToast(`Speaking: "${text}" 🔊`, 'success'); };
        utterance.onend = () => { state.isSpeaking = false; };
        utterance.onerror = (e) => { console.error('Speech error:', e); state.isSpeaking = false; showToast('Speech error', 'error'); };
        speechSynth.speak(utterance);
    }

    function speakLetter(letter) {
        const utterance = new SpeechSynthesisUtterance(letter);
        utterance.rate = 1.2;
        speechSynth.speak(utterance);
    }

    // ===== CAMERA CONTROLS =====
    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false
            });
            els.video.srcObject = stream;
            await new Promise(resolve => {
                els.video.onloadedmetadata = () => { els.video.play(); resolve(); };
            });
            els.startOverlay.classList.add('hidden');
            state.isRunning = true;
            if (!state.hands) initMediaPipe();
            processLoop();
            showToast('Camera started! 📷', 'success');
        } catch (err) {
            console.error('Camera error:', err);
            showToast('Camera access denied. Please allow camera access.', 'error');
        }
    }

    async function processLoop() {
        if (!state.isRunning) return;
        try { await state.hands.send({ image: els.video }); }
        catch (err) { console.error('Processing error:', err); }
        requestAnimationFrame(processLoop);
    }

    // ===== TEXT TO SIGN CONVERTER =====
    function convertTextToSign() {
        const text = els.textInput.value.toUpperCase().trim();
        if (!text) { showToast('Please enter text first', 'warning'); return; }
        els.signDisplay.innerHTML = '';
        let delay = 0;
        for (const char of text) {
            if (char === ' ') {
                setTimeout(() => {
                    const space = document.createElement('div');
                    space.className = 'sign-space';
                    space.innerHTML = '␣';
                    els.signDisplay.appendChild(space);
                }, delay);
                delay += 400;
            } else if (letterData[char]) {
                setTimeout(() => {
                    const data = letterData[char];
                    const box = document.createElement('div');
                    box.className = 'sign-box';
                    box.innerHTML = `
                        <div class="sign-visual" style="color: ${data.color}">${data.emoji}</div>
                        <div class="sign-letter">${char}</div>
                        <div class="sign-emoji">${data.emoji}</div>
                    `;
                    els.signDisplay.appendChild(box);
                    box.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
                }, delay);
                delay += 500;
            }
        }
        showToast(`Converted ${text.length} characters`, 'success');
    }

    // ===== SENTENCES MODE =====
    function setupSentences() {
        if (!els.sentencesGrid) return;
        els.sentencesGrid.innerHTML = sentencesDB.map((s, i) => `
            <div class="sentence-card" data-id="${s.id}" onclick="window.selectSentenceById('${s.id}')" style="animation-delay: ${i * 0.1}s">
                <div class="sentence-emoji">${s.emoji}</div>
                <div class="sentence-text">${s.text}</div>
                <div class="sentence-category">${s.category}</div>
                <div class="sentence-desc">${s.desc}</div>
                <div class="sentence-gesture"><small>Gesture: ${s.gesture}</small></div>
            </div>
        `).join('');
    }

    window.selectSentenceById = function(id) {
        const sentence = sentencesDB.find(s => s.id === id);
        if (!sentence) return;
        state.currentWord = sentence.text;
        updateTranslation();
        document.querySelectorAll('.sentence-card').forEach(card => {
            card.classList.remove('active');
            if (card.dataset.id === id) card.classList.add('active');
        });
        showToast(`Selected: "${sentence.text}" ${sentence.emoji}`, 'success');
        speakText(sentence.text);
        if (els.translationOutput) els.translationOutput.scrollIntoView({ behavior: 'smooth' });
    };

    window.selectSentence = function(text) {
        const sentence = sentencesDB.find(s => s.text === text);
        if (sentence) selectSentenceById(sentence.id);
    };

    window.switchMode = function(mode) {
        state.currentMode = mode;
        document.getElementById('modeLetters')?.classList.toggle('active', mode === 'letters');
        document.getElementById('modeSentences')?.classList.toggle('active', mode === 'sentences');
        state.lastLetter = '';
        state.lastDetectedSentence = '';
        state.stableCount = 0;
        state.currentWord = '';
        updateTranslation();
        showToast(`Switched to ${mode === 'letters' ? 'Letters' : 'Sentences'} mode`, 'success');
    };

    // ===== GUIDE SETUP =====
    function setupGuide() {
        const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
        if (els.alphabetGrid) {
            els.alphabetGrid.innerHTML = letters.map((l, i) => {
                const data = letterData[l];
                return `
                    <div class="letter-showcase" onclick="window.showLetterDetail('${l}')" style="animation-delay: ${i * 0.03}s">
                        <div class="letter-emoji">${data.emoji}</div>
                        <div class="letter-big" style="color: ${data.color}">${l}</div>
                        <div class="letter-desc">${data.name}</div>
                    </div>
                `;
            }).join('');
        }
    }

    window.showLetterDetail = function(letter) {
        const data = letterData[letter];
        if (!data) return;
        const modal = document.createElement('div');
        modal.className = 'modal-overlay active';
        modal.innerHTML = `
            <div class="modal-glass" style="max-width: 400px;">
                <div class="modal-header">
                    <h3 class="modal-title">Letter ${letter} ${data.emoji}</h3>
                    <button class="btn-close" onclick="this.closest('.modal-overlay').remove()">&times;</button>
                </div>
                <div class="modal-body" style="text-align: center; padding: 2rem;">
                    <div style="font-size: 4rem; margin-bottom: 0.5rem;">${data.emoji}</div>
                    <div style="font-size: 6rem; margin-bottom: 1rem; color: ${data.color};">${letter}</div>
                    <h2 style="font-size: 2rem; margin-bottom: 0.5rem; color: ${data.color};">${letter}</h2>
                    <p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">${data.name}</p>
                    <p style="color: var(--text-secondary); line-height: 1.6; margin-bottom: 1rem;">${data.desc}</p>
                    <button class="btn-speak" onclick="window.speakText('${letter}')" style="margin-top: 1rem;">
                        🔊 Hear Pronunciation
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });
    };

    window.speakText = speakText;

    // ===== EVENT LISTENERS =====
    function setupEventListeners() {
        els.startBtn?.addEventListener('click', startCamera);

        document.getElementById('spaceBtn')?.addEventListener('click', () => {
            if (state.currentWord) {
                state.currentWord += ' ';
                updateTranslation();
                showToast('Space added', 'success');
            }
        });

        document.getElementById('deleteBtn')?.addEventListener('click', () => {
            if (state.currentWord) {
                state.currentWord = state.currentWord.slice(0, -1);
                updateTranslation();
            }
        });

        document.getElementById('clearBtn')?.addEventListener('click', () => {
            state.currentWord = '';
            state.detectedLetters = [];
            updateTranslation();
            showToast('Cleared', 'success');
        });

        document.getElementById('copyBtn')?.addEventListener('click', async () => {
            if (!state.currentWord) { showToast('Nothing to copy', 'warning'); return; }
            try {
                await navigator.clipboard.writeText(state.currentWord);
                showToast('Copied to clipboard! 📋', 'success');
            } catch (e) {
                const ta = document.createElement('textarea');
                ta.value = state.currentWord;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                showToast('Copied!', 'success');
            }
        });

        document.getElementById('speakBtn')?.addEventListener('click', () => {
            if (!state.currentWord) { showToast('Nothing to speak', 'warning'); return; }
            speakText(state.currentWord);
        });

        document.getElementById('speakAllBtn')?.addEventListener('click', () => {
            if (!state.currentWord) { showToast('Nothing to speak', 'warning'); return; }
            speakText(state.currentWord);
        });

        els.convertBtn?.addEventListener('click', convertTextToSign);

        els.speakTextBtn?.addEventListener('click', () => {
            const text = els.textInput.value.trim();
            if (!text) { showToast('Please enter text first', 'warning'); return; }
            speakText(text);
        });

        els.textInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                convertTextToSign();
            }
        });

        document.getElementById('settingsBtn')?.addEventListener('click', () => {
            els.settingsModal?.classList.add('active');
        });

        document.getElementById('closeSettings')?.addEventListener('click', () => {
            els.settingsModal?.classList.remove('active');
        });

        els.settingsModal?.addEventListener('click', (e) => {
            if (e.target === els.settingsModal) els.settingsModal.classList.remove('active');
        });

        els.thresholdRange?.addEventListener('input', (e) => {
            state.confidenceThreshold = parseFloat(e.target.value);
            if (els.thresholdValue) els.thresholdValue.textContent = Math.round(state.confidenceThreshold * 100) + '%';
        });

        els.cooldownRange?.addEventListener('input', (e) => {
            state.letterCooldown = parseInt(e.target.value);
            if (els.cooldownValue) els.cooldownValue.textContent = state.letterCooldown + 'ms';
        });

        els.skeletonToggle?.addEventListener('click', () => {
            state.showSkeleton = !state.showSkeleton;
            els.skeletonToggle.classList.toggle('active', state.showSkeleton);
        });

        els.autoSpeakToggle?.addEventListener('click', () => {
            state.autoSpeak = !state.autoSpeak;
            els.autoSpeakToggle.classList.toggle('active', state.autoSpeak);
            showToast(state.autoSpeak ? 'Auto-speak enabled' : 'Auto-speak disabled', 'success');
        });

        document.getElementById('flipBtn')?.addEventListener('click', () => {
            showToast('Use mobile device for back camera', 'info');
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') els.settingsModal?.classList.remove('active');
        });
    }

    // ===== TOAST NOTIFICATIONS =====
    function showToast(message, type = 'info') {
        if (!els.toastContainer) return;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        const icons = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' };
        toast.innerHTML = `<span style="font-size: 1.2rem;">${icons[type]}</span> ${message}`;
        els.toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.animation = 'toastSlide 0.4s ease-out reverse';
            setTimeout(() => toast.remove(), 400);
        }, 3000);
    }

    // ===== START =====
    init();
    console.log('✅ SignBridge Pro ML Edition is ready!');
    console.log('🤖 ML Backend integration active');
});
