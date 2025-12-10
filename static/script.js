/**
 * FaceArt® - Frontend JavaScript
 */

// ============================
// Global Variables
// ============================

let selectedFile = null;
let selectedFiles = [];
let selectedFolderFile = null;

// ============================
// Home Page Functions
// ============================

function clearHome() {
    // Clear any stored data
    selectedFile = null;
    selectedFiles = [];
}

// ============================
// Features Page Functions
// ============================

function handlePersonFiles(event) {
    const files = Array.from(event.target.files);
    selectedFiles = files;
    
    const previewContainer = document.getElementById('personFilesPreview');
    previewContainer.innerHTML = '';
    
    files.forEach((file, index) => {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const div = document.createElement('div');
                div.className = 'file-preview-item';
                div.innerHTML = `<img src="${e.target.result}" alt="${file.name}">`;
                previewContainer.appendChild(div);
            };
            reader.readAsDataURL(file);
        }
    });
}

function handleFolderFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.name.endsWith('.zip')) {
        alert('Please upload a ZIP file');
        event.target.value = '';
        return;
    }
    
    selectedFolderFile = file;
    
    const fileInfo = document.getElementById('folderFileInfo');
    const fileName = document.getElementById('folderFileName');
    fileName.textContent = file.name;
    fileInfo.style.display = 'flex';
}

function clearFolderFile() {
    selectedFolderFile = null;
    document.getElementById('folderFile').value = '';
    document.getElementById('folderFileInfo').style.display = 'none';
}

// Handle Add Person Form
document.addEventListener('DOMContentLoaded', function() {
    const addPersonForm = document.getElementById('addPersonForm');
    if (addPersonForm) {
        addPersonForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const personName = document.getElementById('personName').value.trim();
            if (!personName) {
                showResult('addPersonResult', 'Please enter a person name', 'error');
                return;
            }
            
            if (selectedFiles.length === 0) {
                showResult('addPersonResult', 'Please upload at least one image', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('name', personName);
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            showLoading(true);
            showResult('addPersonResult', 'Registering person...', 'success');
            
            try {
                const response = await fetch('/register-person', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                showLoading(false);
                
                if (data.success) {
                    showResult('addPersonResult', `Successfully registered ${data.person_name} with ${data.successful || data.images_count || selectedFiles.length} images!`, 'success');
                    // Reset form
                    addPersonForm.reset();
                    selectedFiles = [];
                    document.getElementById('personFilesPreview').innerHTML = '';
                } else {
                    showResult('addPersonResult', `Error: ${data.error || 'Registration failed'}`, 'error');
                }
            } catch (error) {
                showLoading(false);
                showResult('addPersonResult', `Error: ${error.message}`, 'error');
            }
        });
    }
    
    // Handle Add Folder Form
    const addFolderForm = document.getElementById('addFolderForm');
    if (addFolderForm) {
        addFolderForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (!selectedFolderFile) {
                showResult('addFolderResult', 'Please upload a ZIP file', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('folder', selectedFolderFile);
            
            showLoading(true);
            showResult('addFolderResult', 'Processing folder...', 'success');
            
            try {
                const response = await fetch('/register-folder', {
                    method: 'POST',
                    body: formData
                });
                
                let data;
                try {
                    data = await response.json();
                } catch (jsonError) {
                    // If response is not JSON, get text instead
                    const text = await response.text();
                    showLoading(false);
                    showResult('addFolderResult', `Error: Server returned: ${text || response.statusText}`, 'error');
                    return;
                }
                
                showLoading(false);
                
                if (response.ok && data.success) {
                    let message = `Successfully processed folder! Registered ${data.persons_registered || 0} person(s) with ${data.total_images || 0} total images.`;
                    if (data.warnings && data.warnings.length > 0) {
                        message += `\n\nWarnings: ${data.warnings.slice(0, 3).join('; ')}`;
                        if (data.warnings.length > 3) {
                            message += ` (and ${data.warnings.length - 3} more)`;
                        }
                    }
                    showResult('addFolderResult', message, 'success');
                    // Reset form
                    addFolderForm.reset();
                    clearFolderFile();
                } else {
                    // Handle HTTPException responses (detail field) or regular error responses
                    const errorMsg = data.detail || data.error || `Processing failed (Status: ${response.status})`;
                    showResult('addFolderResult', `Error: ${errorMsg}`, 'error');
                }
            } catch (error) {
                showLoading(false);
                showResult('addFolderResult', `Error: ${error.message}`, 'error');
            }
        });
    }
    
    // Setup drag and drop for upload areas
    setupDragAndDrop();
});

function showResult(elementId, message, type) {
    const resultElement = document.getElementById(elementId);
    if (resultElement) {
        resultElement.textContent = message;
        resultElement.className = `result-message ${type}`;
        resultElement.style.display = 'block';
    }
}

// ============================
// Try Now Page Functions
// ============================

function handleImageFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        event.target.value = '';
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewSection = document.getElementById('previewSection');
        const previewImage = document.getElementById('previewImage');
        const recognizeBtn = document.getElementById('recognizeBtn');
        
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        recognizeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearPreview() {
    selectedFile = null;
    document.getElementById('imageFile').value = '';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('recognizeBtn').disabled = true;
    document.getElementById('resultsSection').style.display = 'none';
}

function clearAll() {
    clearPreview();
}

async function recognizeFace() {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    showLoading(true);
    document.getElementById('resultsSection').style.display = 'none';
    
    try {
        const response = await fetch('/recognize', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        showLoading(false);
        
        displayResults(data);
    } catch (error) {
        showLoading(false);
        displayError(`Error: ${error.message}`);
    }
}

function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const resultContent = document.getElementById('resultContent');
    
    if (!data.success) {
        displayError(data.error || 'Recognition failed');
        return;
    }
    
    const numFacesDetected = data.num_faces_detected || 0;
    const numFacesRecognized = data.num_faces_recognized || 0;
    const faces = data.faces || [];
    
    if (numFacesDetected === 0) {
        let html = `
            <div class="result-item">
                <div class="result-identity unknown">No Faces Detected</div>
                <p style="color: var(--text-secondary); margin-top: 1rem;">No faces were detected in the uploaded image. Please try a different image with clear, visible faces.</p>
            </div>
        `;
        resultContent.innerHTML = html;
        resultsSection.style.display = 'block';
        return;
    }
    
    // Summary section
    let html = `
        <div class="result-summary">
            <div class="summary-item">
                <span class="summary-label">Total Faces Detected:</span>
                <span class="summary-value">${numFacesDetected}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Faces Recognized:</span>
                <span class="summary-value recognized">${numFacesRecognized}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Faces Not Recognized:</span>
                <span class="summary-value unknown">${numFacesDetected - numFacesRecognized}</span>
            </div>
        </div>
    `;
    
    // Individual face results
    html += '<div class="faces-list">';
    
    faces.forEach((face, index) => {
        const isRecognized = face.is_recognized;
        const identity = isRecognized ? face.identity : 'Unknown';
        
        html += `
            <div class="face-result-item">
                <div class="face-header">
                    <span class="face-number">Face ${face.face_id}</span>
                    <span class="face-status ${isRecognized ? 'recognized' : 'unknown'}">
                        ${isRecognized ? 'Recognized' : 'Not Recognized'}
                    </span>
                </div>
                ${isRecognized ? `
                    <div class="face-details">
                        <div class="face-identity recognized">${identity}</div>
                        ${face.confidence !== null ? `
                            <div class="face-confidence">
                                <span class="confidence-label">Confidence:</span>
                                <span class="confidence-value">${face.confidence.toFixed(2)}%</span>
                            </div>
                        ` : ''}
                        ${face.reference_image_url ? `
                            <div class="reference-photo">
                                <div class="reference-label">Database Match</div>
                                <div class="reference-thumb">
                                    <img src="${face.reference_image_url}" alt="${identity} reference image">
                                </div>
                            </div>
                        ` : ''}
                    </div>
                ` : `
                    <div class="face-details">
                        <div class="face-identity unknown">Unknown Person</div>
                    </div>
                `}
                ${face.error ? `
                    <div class="face-error">Error: ${face.error}</div>
                ` : ''}
            </div>
        `;
    });
    
    html += '</div>';
    
    resultContent.innerHTML = html;
    resultsSection.style.display = 'block';
}

function displayError(message) {
    const resultsSection = document.getElementById('resultsSection');
    const resultContent = document.getElementById('resultContent');
    
    resultContent.innerHTML = `
        <div class="result-item">
            <div class="result-identity unknown">Error</div>
            <p style="color: #dc3545; margin-top: 1rem;">${message}</p>
        </div>
    `;
    resultsSection.style.display = 'block';
}

// ============================
// Utility Functions
// ============================

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

function setupDragAndDrop() {
    // Person upload area
    const personUploadArea = document.getElementById('personUploadArea');
    if (personUploadArea) {
        setupDragDropArea(personUploadArea, (files) => {
            const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
            if (imageFiles.length > 0) {
                selectedFiles = imageFiles;
                handlePersonFiles({ target: { files: imageFiles } });
            }
        });
    }
    
    // Folder upload area
    const folderUploadArea = document.getElementById('folderUploadArea');
    if (folderUploadArea) {
        setupDragDropArea(folderUploadArea, (files) => {
            const zipFile = Array.from(files).find(f => f.name.endsWith('.zip'));
            if (zipFile) {
                selectedFolderFile = zipFile;
                handleFolderFile({ target: { files: [zipFile] } });
            }
        });
    }
    
    // Try Now upload area
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        setupDragDropArea(uploadArea, (files) => {
            const imageFile = Array.from(files).find(f => f.type.startsWith('image/'));
            if (imageFile) {
                selectedFile = imageFile;
                handleImageFile({ target: { files: [imageFile] } });
            }
        });
    }
}

function setupDragDropArea(area, callback) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        area.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        area.addEventListener(eventName, () => {
            area.style.borderColor = '#0066ff';
            area.style.background = 'rgba(0, 102, 255, 0.1)';
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        area.addEventListener(eventName, () => {
            area.style.borderColor = '';
            area.style.background = '';
        }, false);
    });
    
    area.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            callback(files);
        }
    }, false);
}