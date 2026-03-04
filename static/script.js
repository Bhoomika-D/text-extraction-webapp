// Global variables
let currentFilename = null;
let currentResults = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    loadFilters();
});

function setupEventListeners() {
    // File input change
    const fileInput = document.getElementById('imageInput');
    fileInput.addEventListener('change', handleFileSelect);
    
    // Upload area click
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    
    // Process button
    document.getElementById('processBtn').addEventListener('click', processImage);
    
    // Confidence slider
    const confidenceSlider = document.getElementById('confidenceInput');
    confidenceSlider.addEventListener('input', function() {
        document.getElementById('confidenceValue').textContent = this.value;
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        uploadImage(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.style.borderColor = '#764ba2';
    event.currentTarget.style.background = '#f0f2ff';
}

function handleDragLeave(event) {
    event.currentTarget.style.borderColor = '#667eea';
    event.currentTarget.style.background = '#f8f9ff';
}

function handleDrop(event) {
    event.preventDefault();
    handleDragLeave(event);
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        uploadImage(file);
    } else {
        alert('Please drop an image file');
    }
}

function uploadImage(file) {
    const formData = new FormData();
    formData.append('image', file);
    
    showLoading();
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            currentFilename = data.filename;
            displayUploadedImage(data.image);
            showFilterSection();
            showNotification('Image uploaded successfully!', 'success');
        } else {
            showNotification(data.error || 'Upload failed', 'error');
        }
    })
    .catch(error => {
        hideLoading();
        showNotification('Error uploading image: ' + error.message, 'error');
    });
}

function displayUploadedImage(imageData) {
    const img = document.getElementById('uploadedImage');
    img.src = imageData;
    img.style.display = 'block';
    
    const placeholder = document.querySelector('.upload-placeholder');
    placeholder.style.display = 'none';
}

function showFilterSection() {
    document.getElementById('filterSection').style.display = 'block';
}

function processImage() {
    if (!currentFilename) {
        showNotification('Please upload an image first', 'error');
        return;
    }
    
    // Get image type (simple or complex)
    const imageType = document.querySelector('input[name="imageType"]:checked').value;
    const filterMethod = document.getElementById('filterSelect').value;
    const confidence = document.getElementById('confidenceInput').value;
    
    showLoading();
    
    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: currentFilename,
            image_type: imageType,
            filter: filterMethod,
            confidence: confidence
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            currentResults = data.text_results;
            displayResults(data);
            showNotification('Processing completed!', 'success');
        } else {
            showNotification(data.error || 'Processing failed', 'error');
        }
    })
    .catch(error => {
        hideLoading();
        showNotification('Error processing image: ' + error.message, 'error');
    });
}

function displayResults(data) {
    // Display filtered image
    const filteredImg = document.getElementById('filteredImage');
    filteredImg.src = data.filtered_image;
    
    document.getElementById('filterInfo').textContent = 
        `Filter: ${data.filter_method} | Confidence Threshold: ${document.getElementById('confidenceInput').value}`;
    
    // Display OCR method used
    document.getElementById('ocrMethodInfo').textContent = 
        `OCR Method: ${data.ocr_method} | Image Type: ${data.image_type === 'simple' ? 'Simple' : 'Complex'}`;
    
    // Display text statistics
    const stats = data.text_results;
    document.getElementById('textStats').innerHTML = `
        <div class="stat-item">
            <span class="stat-label">Total Detections</span>
            <span class="stat-value">${stats.num_detections}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">English</span>
            <span class="stat-value">${stats.english_text ? stats.english_text.split(' ').length : 0}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Persian</span>
            <span class="stat-value">${stats.persian_text ? stats.persian_text.split(' ').length : 0}</span>
        </div>
    `;
    
    // Display text
    document.getElementById('combinedText').textContent = 
        stats.combined_text || 'No text extracted.';
    
    document.getElementById('englishText').textContent = 
        stats.english_text || 'No English text found.';
    
    const persianTextEl = document.getElementById('persianText');
    persianTextEl.textContent = stats.persian_text || 'No Persian text found.';
    if (stats.persian_text) {
        persianTextEl.classList.add('persian-text');
    } else {
        persianTextEl.classList.remove('persian-text');
    }
    
    // Display detections
    displayDetections(stats.detections);
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function displayDetections(detections) {
    const detectionsList = document.getElementById('detectionsList');
    
    if (!detections || detections.length === 0) {
        detectionsList.innerHTML = '<p style="color: #999; font-style: italic;">No detections found.</p>';
        return;
    }
    
    detectionsList.innerHTML = detections.map(det => `
        <div class="detection-item">
            <div class="detection-header">
                <span class="detection-id">#${det.id}</span>
                <span class="detection-confidence">${(det.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="detection-text ${det.language === 'persian' ? 'persian-text' : ''}">${det.text}</div>
            <div class="detection-language">${det.language.toUpperCase()}</div>
        </div>
    `).join('');
}

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + 'Tab').classList.add('active');
    
    // Activate button
    event.target.classList.add('active');
}

function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showNotification(message, type) {
    // Simple notification (you can enhance this with a toast library)
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        background: ${type === 'success' ? '#38ef7d' : '#ff6b6b'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        z-index: 1001;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function loadFilters() {
    fetch('/filters')
        .then(response => response.json())
        .then(data => {
            const select = document.getElementById('filterSelect');
            select.innerHTML = data.filters.map(filter => 
                `<option value="${filter.value}">${filter.name}</option>`
            ).join('');
        })
        .catch(error => console.error('Error loading filters:', error));
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

