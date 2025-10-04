// Enhanced Exo-Explorer Dashboard Script
// Premium Edition with Smooth Animations

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
        // Simple notification system (you can enhance this)
        console.log(`[${type.toUpperCase()}] ${message}`);
    };
    
    const animateValue = (element, start, end, duration) => {
        const range = end - start;
        const increment = range / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                current = end;
                clearInterval(timer);
            }
            element.textContent = current.toFixed(1) + '%';
        }, 16);
    };
    
    // === API FUNCTIONS ===
    async function fetchModelStats() {
        try {
            const response = await fetch(`${API_BASE_URL}/model_stats`);
            const stats = await response.json();
            
            const container = document.getElementById('model-stats-container');
            container.innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Accuracy</span>
                    <span class="stat-value shimmer">${stats.accuracy}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">F1-Score</span>
                    <span class="stat-value shimmer">${stats.f1Score}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Precision</span>
                    <span class="stat-value shimmer">${stats.precision}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Recall</span>
                    <span class="stat-value shimmer">${stats.recall}</span>
                </div>
            `;
        } catch (error) {
            console.error('Error fetching model stats:', error);
            showNotification('Failed to load model statistics', 'error');
        }
    }
    
    async function getPrediction(params) {
        try {
            // Add loading state
            predictionOutput.parentElement.style.opacity = '0.6';
            
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            
            const result = await response.json();
            
            // Animate the update
            setTimeout(() => {
                updateExplorerUI(result);
                predictionOutput.parentElement.style.opacity = '1';
            }, 200);
        } catch (error) {
            console.error('Error getting prediction:', error);
            showNotification('Prediction failed', 'error');
            predictionOutput.parentElement.style.opacity = '1';
        }
    }
    
    // === UI UPDATE FUNCTIONS ===
    function updateExplorerUI(result) {
        const resultElement = predictionOutput.parentElement;
        
        // Update prediction text
        predictionOutput.querySelector('.prediction-text').textContent = result.prediction;
        
        // Set data attribute for styling
        resultElement.setAttribute('data-prediction', result.prediction);
        
        // Animate confidence
        const currentConfidence = parseFloat(probabilityDisplay.textContent.match(/[\d.]+/)?.[0] || 0);
        const newConfidence = result.probability * 100;
        animateValue(probabilityDisplay, currentConfidence, newConfidence, 500);
        
        // Update explanation
        explanationText.textContent = generateExplanation(result);
        
        // Update chart
        if (result.featureImportance) {
            updateChart(result.featureImportance);
        }
        
        // Add pulse animation
        resultElement.style.animation = 'none';
        setTimeout(() => {
            resultElement.style.animation = '';
        }, 10);
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
            koi_period: { label: 'Orbital Period (days)', min: 0.1, max: 200, value: 10, unit: 'd' },
            koi_depth: { label: 'Transit Depth (ppm)', min: 1, max: 20000, value: 500, unit: '' },
            koi_duration: { label: 'Transit Duration (hrs)', min: 0.1, max: 10, value: 3, unit: 'h' },
            koi_prad: { label: 'Planetary Radius (R⊕)', min: 0.1, max: 25, value: 2, unit: 'R⊕' },
            koi_insol: { label: 'Insolation Flux (F⊕)', min: 0.1, max: 1000, value: 10, unit: 'F⊕' },
            koi_steff: { label: 'Stellar Temp (K)', min: 2000, max: 10000, value: 5500, unit: 'K' },
            koi_srad: { label: 'Stellar Radius (R☉)', min: 0.1, max: 20, value: 1, unit: 'R☉' },
            koi_slogg: { label: 'Stellar Gravity (log g)', min: 1, max: 6, value: 4.5, unit: '' }
        };
        
        sliderContainer.innerHTML = '';
        manualForm.innerHTML = '';
        
        for (const key in sliderDefs) {
            const def = sliderDefs[key];
            currentParams[key] = def.value;
            
            // Create slider for Explorer mode
            const sliderGroup = document.createElement('div');
            sliderGroup.className = 'slider-group';
            
            const label = document.createElement('label');
            label.innerHTML = `
                <span>${def.label}</span>
                <span id="${key}-val">${def.value.toFixed(2)}${def.unit}</span>
            `;
            
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = def.min;
            slider.max = def.max;
            slider.value = def.value;
            slider.step = (def.max - def.min) / 200;
            
            // Debounce prediction calls
            let predictionTimeout;
            slider.oninput = (e) => {
                const val = parseFloat(e.target.value);
                currentParams[key] = val;
                document.getElementById(`${key}-val`).textContent = val.toFixed(2) + def.unit;
                
                clearTimeout(predictionTimeout);
                predictionTimeout = setTimeout(() => {
                    getPrediction(currentParams);
                }, 300);
            };
            
            sliderGroup.appendChild(label);
            sliderGroup.appendChild(slider);
            sliderContainer.appendChild(sliderGroup);
            
            // Create input for Researcher mode
            const input = document.createElement('input');
            input.type = 'number';
            input.name = key;
            input.placeholder = def.label;
            input.step = 'any';
            input.required = true;
            manualForm.appendChild(input);
        }
    }
    
    function updateChart(importanceData) {
        const ctx = document.getElementById('importance-chart');
        
        if (!importanceChart) {
            importanceChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Importance',
                        data: [],
                        backgroundColor: (context) => {
                            const gradient = context.chart.ctx.createLinearGradient(0, 0, context.chart.width, 0);
                            gradient.addColorStop(0, '#38bdf8');
                            gradient.addColorStop(1, '#8b5cf6');
                            return gradient;
                        },
                        borderRadius: 8,
                        borderSkipped: false,
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(26, 31, 53, 0.95)',
                            titleColor: '#fff',
                            bodyColor: '#b8c5d6',
                            borderColor: 'rgba(56, 189, 248, 0.5)',
                            borderWidth: 1,
                            padding: 12,
                            displayColors: false,
                            callbacks: {
                                label: (context) => `Importance: ${context.parsed.x.toFixed(4)}`
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.05)',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#6b7a94',
                                font: { size: 11 }
                            }
                        },
                        y: {
                            grid: { display: false },
                            ticks: {
                                color: '#b8c5d6',
                                font: { size: 12, weight: '500' }
                            }
                        }
                    },
                    animation: {
                        duration: 750,
                        easing: 'easeInOutQuart'
                    }
                }
            });
        }
        
        // Update data
        importanceChart.data.labels = importanceData.map(d => d.feature);
        importanceChart.data.datasets[0].data = importanceData.map(d => d.value);
        importanceChart.update('active');
    }
    
    // === EVENT LISTENERS ===
    
    // Mode Switching
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
    
    // Load Random Button
    loadRandomBtn.onclick = () => {
        loadRandomBtn.disabled = true;
        loadRandomBtn.innerHTML = '<span class="loading"></span><span>Generating...</span>';
        
        const sliders = sliderContainer.querySelectorAll('input[type="range"]');
        sliders.forEach(slider => {
            const min = parseFloat(slider.min);
            const max = parseFloat(slider.max);
            const randomValue = Math.random() * (max - min) + min;
            slider.value = randomValue;
            slider.dispatchEvent(new Event('input'));
        });
        
        setTimeout(() => {
            loadRandomBtn.disabled = false;
            loadRandomBtn.innerHTML = `
                <span>Generate Random</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="23 4 23 10 17 10"></polyline>
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                </svg>
            `;
        }, 1000);
    };
    
    // Manual Classification
    classifyManualBtn.onclick = async (e) => {
        e.preventDefault();
        
        const formData = new FormData(manualForm);
        const data = {};
        let isValid = true;
        
        for (let [key, value] of formData.entries()) {
            const numValue = parseFloat(value);
            if (isNaN(numValue)) {
                isValid = false;
                break;
            }
            data[key] = numValue;
        }
        
        if (!isValid) {
            showNotification('Please fill all fields with valid numbers', 'error');
            return;
        }
        
        // Show loading state
        classifyManualBtn.disabled = true;
        classifyManualBtn.innerHTML = '<span class="loading"></span><span>Classifying...</span>';
        
        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (result.error) {
                showNotification(result.error, 'error');
                return;
            }
            
            resultsTableContainer.innerHTML = createResultsTable([{...data, ...result}]);
            showNotification('Classification completed successfully', 'success');
        } catch (error) {
            console.error('Error classifying:', error);
            showNotification('Classification failed', 'error');
        } finally {
            classifyManualBtn.disabled = false;
            classifyManualBtn.innerHTML = `
                <span>Classify Candidate</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="9 11 12 14 22 4"></polyline>
                    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
                </svg>
            `;
        }
    };
    
    // CSV File Upload
    fileInput.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading in results container
        resultsTableContainer.innerHTML = `
            <div class="empty-state">
                <span class="loading" style="width: 40px; height: 40px; margin-bottom: 20px;"></span>
                <p>Processing ${file.name}...</p>
            </div>
        `;
        
        try {
            const response = await fetch(`${API_BASE_URL}/predict_batch`, {
                method: 'POST',
                body: formData
            });
            
            const results = await response.json();
            
            if (results.error) {
                resultsTableContainer.innerHTML = `
                    <div class="empty-state">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="15" y1="9" x2="9" y2="15"></line>
                            <line x1="9" y1="9" x2="15" y2="15"></line>
                        </svg>
                        <p>Error: ${results.error}</p>
                    </div>
                `;
                showNotification(results.error, 'error');
                return;
            }
            
            resultsTableContainer.innerHTML = createResultsTable(results);
            showNotification(`Successfully processed ${results.length} candidates`, 'success');
        } catch (error) {
            console.error('Error uploading file:', error);
            resultsTableContainer.innerHTML = `
                <div class="empty-state">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="15" y1="9" x2="9" y2="15"></line>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                    <p>An error occurred during file upload.</p>
                </div>
            `;
            showNotification('File upload failed', 'error');
        } finally {
            // Reset file input
            fileInput.value = '';
        }
    };
    
    // Drag and drop for file upload
    const uploadLabel = document.querySelector('.upload-label');
    
    uploadLabel.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadLabel.style.borderColor = 'var(--accent-blue)';
        uploadLabel.style.background = 'var(--bg-light)';
    });
    
    uploadLabel.addEventListener('dragleave', () => {
        uploadLabel.style.borderColor = '';
        uploadLabel.style.background = '';
    });
    
    uploadLabel.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadLabel.style.borderColor = '';
        uploadLabel.style.background = '';
        
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.csv')) {
            fileInput.files = e.dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        } else {
            showNotification('Please upload a CSV file', 'error');
        }
    });
    
    // === HELPER FUNCTIONS ===
    
    function createResultsTable(data) {
        if (!data || data.length === 0) {
            return `
                <div class="empty-state">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M9 11l3 3L22 4"></path>
                        <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
                    </svg>
                    <p>No results to display</p>
                </div>
            `;
        }
        
        const headers = ['prediction', 'probability', 'koi_prad', 'koi_period', 'koi_depth'];
        const headerLabels = {
            'prediction': 'Classification',
            'probability': 'Confidence',
            'koi_prad': 'Radius (R⊕)',
            'koi_period': 'Period (days)',
            'koi_depth': 'Depth (ppm)'
        };
        
        let table = '<table><thead><tr>';
        headers.forEach(h => {
            table += `<th>${headerLabels[h] || h}</th>`;
        });
        table += '</tr></thead><tbody>';
        
        data.forEach(row => {
            table += '<tr>';
            headers.forEach(h => {
                let val = row[h];
                let cellClass = '';
                
                if (h === 'prediction') {
                    // Add color coding for predictions
                    if (val === 'CONFIRMED') cellClass = 'text-success';
                    else if (val === 'FALSE POSITIVE') cellClass = 'text-error';
                    else cellClass = 'text-warning';
                    table += `<td class="${cellClass}"><strong>${val}</strong></td>`;
                } else if (h === 'probability') {
                    const percent = (val * 100).toFixed(1);
                    table += `<td><strong>${percent}%</strong></td>`;
                } else {
                    if (typeof val === 'number') val = val.toFixed(2);
                    table += `<td>${val}</td>`;
                }
            });
            table += '</tr>';
        });
        
        table += '</tbody></table>';
        
        // Add summary statistics
        const summary = `
            <div style="margin-top: 20px; padding: 15px; background: var(--bg-lighter); border-radius: 10px; display: flex; justify-content: space-around; flex-wrap: wrap; gap: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: var(--text-muted);">Total Candidates</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent-blue);">${data.length}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: var(--text-muted);">Confirmed</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--success);">${data.filter(r => r.prediction === 'CONFIRMED').length}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: var(--text-muted);">Candidates</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--warning);">${data.filter(r => r.prediction === 'CANDIDATE').length}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: var(--text-muted);">False Positives</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--error);">${data.filter(r => r.prediction === 'FALSE POSITIVE').length}</div>
                </div>
            </div>
        `;
        
        return table + summary;
    }
    
    // Add some CSS dynamically for text colors
    const style = document.createElement('style');
    style.textContent = `
        .text-success { color: var(--success) !important; }
        .text-error { color: var(--error) !important; }
        .text-warning { color: var(--warning) !important; }
    `;
    document.head.appendChild(style);
    
    // === KEYBOARD SHORTCUTS ===
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + 1: Switch to Explorer Mode
        if ((e.ctrlKey || e.metaKey) && e.key === '1') {
            e.preventDefault();
            modeExplorerBtn.click();
        }
        
        // Ctrl/Cmd + 2: Switch to Researcher Mode
        if ((e.ctrlKey || e.metaKey) && e.key === '2') {
            e.preventDefault();
            modeResearcherBtn.click();
        }
        
        // Ctrl/Cmd + R: Load Random (in Explorer mode)
        if ((e.ctrlKey || e.metaKey) && e.key === 'r' && !explorerView.classList.contains('hidden')) {
            e.preventDefault();
            loadRandomBtn.click();
        }
    });
    
    // === INITIALIZATION ===
    
    // Create sliders and form inputs
    createSliders();
    
    // Fetch initial model stats
    fetchModelStats();
    
    // Get initial prediction with default values
    getPrediction(currentParams);
    
    // Add welcome animation
    setTimeout(() => {
        const cards = document.querySelectorAll('.glass-card');
        cards.forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
        });
    }, 100);
    
    // Log initialization complete
    console.log('%c🚀 Exo-Explorer Dashboard Initialized', 'color: #38bdf8; font-size: 16px; font-weight: bold;');
    console.log('%c⌨️  Keyboard Shortcuts:', 'color: #8b5cf6; font-weight: bold;');
    console.log('  Ctrl/Cmd + 1: Explorer Mode');
    console.log('  Ctrl/Cmd + 2: Researcher Mode');
    console.log('  Ctrl/Cmd + R: Load Random Candidate');
});