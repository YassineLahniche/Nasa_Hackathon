// Enhanced Exo-Explorer Dashboard Script
// Premium Edition with Smooth Animations (Final Version)

document.addEventListener('DOMContentLoaded', () => {
    // === CONFIGURATION ===
    const API_BASE_URL = 'http://127.0.0.1:5001';
    
    // === STATE ===
    let currentParams = {};
    let importanceChart = null;
    
    // === DOM ELEMENTS ===
    const modeExplorerBtn = document.getElementById('mode-explorer');
    const modeResearcherBtn = document.getElementById('mode-researcher');
    const explorerView = document.getElementById('explorer-view');
    const researcherView = document.getElementById('researcher-view');
    const explorerControls = document.getElementById('explorer-controls');
    const researcherControls = document.getElementById('researcher-controls');
    const predictionOutput = document.getElementById('prediction-output');
    const probabilityDisplay = document.getElementById('probability-display');
    const explanationText = document.getElementById('explanation-text');
    const sliderContainer = document.getElementById('slider-container');
    const loadRandomBtn = document.getElementById('load-random-btn');
    const manualForm = document.getElementById('manual-entry-form');
    const classifyManualBtn = document.getElementById('classify-manual-btn');
    const fileInput = document.getElementById('csv-file-input');
    const resultsTableContainer = document.getElementById('results-table-container');
    
    // === UTILITY FUNCTIONS ===
    const showNotification = (message, type = 'info') => {
        // Simple console notification for now. Can be replaced with a UI element.
        console.log(`[${type.toUpperCase()}] ${message}`);
    };
    
    const animateValue = (element, start, end, duration) => {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const currentValue = progress * (end - start) + start;
            element.textContent = `Confidence: ${currentValue.toFixed(1)}%`;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    };
    
    // === API FUNCTIONS ===
    async function fetchModelStats() {
        try {
            const response = await fetch(`${API_BASE_URL}/model_stats`);
            if (!response.ok) throw new Error('Failed to fetch stats');
            const stats = await response.json();
            
            const container = document.getElementById('model-stats-container');
            container.innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Accuracy</span>
                    <span class="stat-value">${stats.accuracy}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">F1-Score</span>
                    <span class="stat-value">${stats.f1Score}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Precision</span>
                    <span class="stat-value">${stats.precision}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Recall</span>
                    <span class="stat-value">${stats.recall}</span>
                </div>
            `;
        } catch (error) {
            console.error('Error fetching model stats:', error);
            showNotification('Failed to load model statistics', 'error');
        }
    }
    
    async function getPrediction(params) {
        try {
            predictionOutput.parentElement.style.opacity = '0.6';
            
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            
            if (!response.ok) { throw new Error(`API Error: ${response.statusText}`); }
            const result = await response.json();
            if (result.error) { throw new Error(result.error); }
            
            setTimeout(() => {
                updateExplorerUI(result);
                predictionOutput.parentElement.style.opacity = '1';
            }, 200);
        } catch (error) {
            console.error('Error getting prediction:', error);
            showNotification(`Prediction failed: ${error.message}`, 'error');
            predictionOutput.parentElement.style.opacity = '1';
        }
    }
    
    // === UI UPDATE FUNCTIONS ===
    function updateExplorerUI(result) {
        const resultElement = predictionOutput;
        resultElement.querySelector('.prediction-text').textContent = result.prediction;
        resultElement.setAttribute('data-prediction', result.prediction);
        
        const currentConfidence = parseFloat(probabilityDisplay.textContent.match(/[\d.]+/)?.[0] || 0);
        const newConfidence = result.probability * 100;
        animateValue(probabilityDisplay, currentConfidence, newConfidence, 500);
        
        explanationText.textContent = generateExplanation(result);
        
        if (result.featureImportance) {
            updateChart(result.featureImportance);
        }
        
        resultElement.style.animation = 'none';
        setTimeout(() => { resultElement.style.animation = ''; }, 10);
    }
    
    function generateExplanation(result) {
        const confidence = (result.probability * 100).toFixed(1);
        const prediction = result.prediction;
        let explanation = `The AI model predicts this candidate is a ${prediction} with ${confidence}% confidence. `;
        
        if (prediction === 'CONFIRMED') {
            explanation += 'The orbital and physical parameters strongly match known exoplanet characteristics.';
        } else if (prediction === 'FALSE POSITIVE') {
            explanation += 'The signal characteristics suggest this is likely not a genuine exoplanet transit.';
        } else {
            explanation += 'This candidate shows promising features but requires further validation.';
        }
        return explanation;
    }
    
    function createSliders() {
        const sliderDefs = {
            'orbital_period': { label: 'Orbital Period (days)', min: 0.1, max: 200, value: 10, unit: 'd' },
            'transit_duration': { label: 'Transit Duration (hours)', min: 0.01, max: 0.5, value: 0.125, unit: 'h' },
            'transit_depth': { label: 'Transit Depth (ppm)', min: 1, max: 20000, value: 500, unit: '' },
            'planet_radius': { label: 'Planetary Radius (R⊕)', min: 0.1, max: 25, value: 2, unit: 'R⊕' },
            'insolation_flux': { label: 'Insolation Flux (F⊕)', min: 0.1, max: 1000, value: 10, unit: 'F⊕' },
            'stellar_temp': { label: 'Stellar Temp (K)', min: 2000, max: 10000, value: 5500, unit: 'K' },
            'stellar_radius': { label: 'Stellar Radius (R☉)', min: 0.1, max: 20, value: 1, unit: 'R☉' },
            'stellar_log_g': { label: 'Stellar Gravity (log g)', min: 1, max: 6, value: 4.5, unit: '' }
        };
        
        sliderContainer.innerHTML = '';
        manualForm.innerHTML = '';
        
        for (const key in sliderDefs) {
            const def = sliderDefs[key];
            currentParams[key] = def.value;
            
            const sliderGroup = document.createElement('div');
            sliderGroup.className = 'slider-group';
            
            const label = document.createElement('label');
            label.innerHTML = `<span>${def.label}</span><span id="${key}-val">${def.value.toFixed(2)}${def.unit}</span>`;
            
            const slider = document.createElement('input');
            slider.type = 'range'; slider.min = def.min; slider.max = def.max; slider.value = def.value;
            slider.step = (def.max - def.min) / 200;
            slider.setAttribute('data-key', key);
            slider.setAttribute('data-unit', def.unit || '');
            
            let predictionTimeout;
            slider.oninput = (e) => {
                const val = parseFloat(e.target.value);
                currentParams[key] = val;
                document.getElementById(`${key}-val`).textContent = val.toFixed(2) + def.unit;
                clearTimeout(predictionTimeout);
                predictionTimeout = setTimeout(() => getPrediction(currentParams), 300);
            };
            
            sliderGroup.appendChild(label);
            sliderGroup.appendChild(slider);
            sliderContainer.appendChild(sliderGroup);
            
            const input = document.createElement('input');
            input.type = 'number'; input.name = key; input.placeholder = def.label;
            input.step = 'any'; input.required = true;
            manualForm.appendChild(input);
        }
    }
    
    function updateChart(importanceData) {
        const ctx = document.getElementById('importance-chart').getContext('2d');
        if (!importanceChart) {
            importanceChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Importance', data: [],
                        backgroundColor: (context) => {
                            const gradient = context.chart.ctx.createLinearGradient(0, 0, 0, context.chart.height);
                            gradient.addColorStop(0, '#38bdf8'); gradient.addColorStop(1, '#8b5cf6');
                            return gradient;
                        },
                        borderRadius: 8, borderSkipped: false
                    }]
                },
                options: {
                    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(26, 31, 53, 0.95)', titleColor: '#fff', bodyColor: '#b8c5d6',
                            borderColor: 'rgba(56, 189, 248, 0.5)', borderWidth: 1, padding: 12, displayColors: false,
                            callbacks: { label: (context) => `Importance: ${context.parsed.x.toFixed(4)}` }
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#6b7a94' } },
                        y: { grid: { display: false }, ticks: { color: '#b8c5d6' } }
                    },
                    animation: { duration: 750, easing: 'easeInOutQuart' }
                }
            });
        }
        importanceChart.data.labels = importanceData.map(d => d.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
        importanceChart.data.datasets[0].data = importanceData.map(d => d.value);
        importanceChart.update('active');
    }
    
    // === EVENT LISTENERS ===
    
    modeExplorerBtn.onclick = () => {
        modeExplorerBtn.classList.add('active');
        modeResearcherBtn.classList.remove('active');
        explorerView.classList.remove('hidden');
        researcherView.classList.add('hidden');
        explorerControls.classList.remove('hidden');
        researcherControls.classList.add('hidden');
    };
    
    modeResearcherBtn.onclick = () => {
        modeResearcherBtn.classList.add('active');
        modeExplorerBtn.classList.remove('active');
        researcherView.classList.remove('hidden');
        explorerView.classList.add('hidden');
        researcherControls.classList.remove('hidden');
        explorerControls.classList.add('hidden');
    };
    
    loadRandomBtn.onclick = async () => {
        loadRandomBtn.disabled = true;
        loadRandomBtn.innerHTML = '<span class="loading"></span><span>Generating...</span>';
        try {
            const response = await fetch(`${API_BASE_URL}/random_candidate`);
            if (!response.ok) { throw new Error('Network error'); }
            const randomParams = await response.json();

            for (const key in randomParams) {
                const slider = sliderContainer.querySelector(`input[data-key="${key}"]`);
                if (slider) {
                    const newValue = parseFloat(randomParams[key]);
                    slider.value = newValue;
                    const unit = slider.getAttribute('data-unit') || '';
                    document.getElementById(`${key}-val`).textContent = newValue.toFixed(2) + unit;
                    currentParams[key] = newValue; // Update current state
                }
            }
            await getPrediction(currentParams);
        } catch (error) {
            console.error('Error loading random candidate:', error);
            showNotification('Failed to load a random candidate.', 'error');
        } finally {
            setTimeout(() => {
                loadRandomBtn.disabled = false;
                loadRandomBtn.innerHTML = `<span>Generate Random</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>`;
            }, 500);
        }
    };
    
    classifyManualBtn.onclick = async (e) => {
        e.preventDefault();
        const formData = new FormData(manualForm);
        const data = {};
        let isValid = true;
        
        for (let [key, value] of formData.entries()) {
            const numValue = parseFloat(value);
            if (isNaN(numValue)) { isValid = false; break; }
            data[key] = numValue;
        }
        
        if (!isValid) {
            showNotification('Please fill all fields with valid numbers', 'error');
            return;
        }
        
        classifyManualBtn.disabled = true;
        classifyManualBtn.innerHTML = '<span class="loading"></span><span>Classifying...</span>';
        
        try {
            const response = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
            const result = await response.json();
            if (result.error) { throw new Error(result.error); }
            resultsTableContainer.innerHTML = createResultsTable([{...data, ...result}]);
            showNotification('Classification completed successfully', 'success');
        } catch (error) {
            console.error('Error classifying:', error);
            showNotification(`Classification failed: ${error.message}`, 'error');
        } finally {
            classifyManualBtn.disabled = false;
            classifyManualBtn.innerHTML = `<span>Classify Candidate</span><svg width="16" height="16" viewBox="0 0 24 24"><polyline points="9 11 12 14 22 4"></polyline><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path></svg>`;
        }
    };
    
    fileInput.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        resultsTableContainer.innerHTML = `<div class="empty-state"><span class="loading" style="width: 40px; height: 40px; margin-bottom: 20px;"></span><p>Processing ${file.name}...</p></div>`;
        
        try {
            const response = await fetch(`${API_BASE_URL}/predict_batch`, { method: 'POST', body: formData });
            const results = await response.json();
            
            if (results.error) {
                resultsTableContainer.innerHTML = `<div class="empty-state"><p>Error: ${results.error}</p></div>`;
                showNotification(results.error, 'error');
                return;
            }
            resultsTableContainer.innerHTML = createResultsTable(results);
            showNotification(`Successfully processed ${results.length} candidates`, 'success');
        } catch (error) {
            console.error('Error uploading file:', error);
            resultsTableContainer.innerHTML = `<div class="empty-state"><p>An error occurred during file upload.</p></div>`;
            showNotification('File upload failed', 'error');
        } finally {
            fileInput.value = '';
        }
    };
    
    const uploadLabel = document.querySelector('.upload-label');
    uploadLabel.addEventListener('dragover', (e) => { e.preventDefault(); uploadLabel.classList.add('active'); });
    uploadLabel.addEventListener('dragleave', () => { uploadLabel.classList.remove('active'); });
    uploadLabel.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadLabel.classList.remove('active');
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.csv')) {
            fileInput.files = e.dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        } else {
            showNotification('Please upload a CSV file', 'error');
        }
    });
    
    function createResultsTable(data) {
        if (!data || data.length === 0) { return `<div class="empty-state"><p>No results to display</p></div>`; }
        
        const headers = ['prediction', 'probability', 'planet_radius', 'orbital_period', 'transit_depth'];
        const headerLabels = { 'prediction': 'Classification', 'probability': 'Confidence', 'planet_radius': 'Radius (R⊕)', 'orbital_period': 'Period (days)', 'transit_depth': 'Depth (ppm)' };
        
        let table = '<table><thead><tr>';
        headers.forEach(h => { table += `<th>${headerLabels[h] || h}</th>`; });
        table += '</tr></thead><tbody>';
        
        data.forEach(row => {
            table += '<tr>';
            headers.forEach(h => {
                let val = row[h];
                if (h === 'prediction') {
                    let cellClass = val === 'CONFIRMED' ? 'text-success' : val === 'FALSE POSITIVE' ? 'text-error' : 'text-warning';
                    table += `<td class="${cellClass}"><strong>${val}</strong></td>`;
                } else if (h === 'probability') {
                    table += `<td><strong>${(val * 100).toFixed(1)}%</strong></td>`;
                } else {
                    table += `<td>${(typeof val === 'number') ? val.toFixed(2) : val}</td>`;
                }
            });
            table += '</tr>';
        });
        
        table += '</tbody></table>';
        
        const summary = `
            <div class="results-summary">
                <div><span>Total</span><strong class="text-blue">${data.length}</strong></div>
                <div><span>Confirmed</span><strong class="text-success">${data.filter(r => r.prediction === 'CONFIRMED').length}</strong></div>
                <div><span>Candidates</span><strong class="text-warning">${data.filter(r => r.prediction === 'CANDIDATE').length}</strong></div>
                <div><span>False Positives</span><strong class="text-error">${data.filter(r => r.prediction === 'FALSE POSITIVE').length}</strong></div>
            </div>`;
        
        return table + summary;
    }
    
    const style = document.createElement('style');
    style.textContent = `
        .text-success { color: var(--success) !important; } .text-error { color: var(--error) !important; } .text-warning { color: var(--warning) !important; } .text-blue { color: var(--accent-blue) !important; }
        .upload-label.active { border-color: var(--accent-blue); background: var(--bg-light); }
        .results-summary { margin-top: 20px; padding: 15px; background: var(--bg-lighter); border-radius: 10px; display: flex; justify-content: space-around; flex-wrap: wrap; gap: 15px; }
        .results-summary > div { text-align: center; }
        .results-summary > div > span { font-size: 0.85rem; color: var(--text-muted); display: block; }
        .results-summary > div > strong { font-size: 1.5rem; font-weight: 700; }
    `;
    document.head.appendChild(style);
    
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === '1') { e.preventDefault(); modeExplorerBtn.click(); }
        if ((e.ctrlKey || e.metaKey) && e.key === '2') { e.preventDefault(); modeResearcherBtn.click(); }
        if ((e.ctrlKey || e.metaKey) && e.key === 'r' && !explorerView.classList.contains('hidden')) { e.preventDefault(); loadRandomBtn.click(); }
    });
    
    // === INITIALIZATION ===
    createSliders();
    fetchModelStats();
    getPrediction(currentParams);
    
    setTimeout(() => {
        document.querySelectorAll('.glass-card').forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
        });
    }, 100);
    
    console.log('%c🚀 Exo-Explorer Dashboard Initialized', 'color: #38bdf8; font-size: 16px; font-weight: bold;');
    console.log('%c⌨️  Keyboard Shortcuts:', 'color: #8b5cf6; font-weight: bold;');
    console.log('  Ctrl/Cmd + 1: Explorer Mode');
    console.log('  Ctrl/Cmd + 2: Researcher Mode');
    console.log('  Ctrl/Cmd + R: Load Random Candidate');
});