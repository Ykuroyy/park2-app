
document.addEventListener('DOMContentLoaded', function () {
    // Camera-related elements
    const cameraPreview = document.getElementById('camera-preview');
    const captureCanvas = document.getElementById('capture-canvas');
    const guideFrame = document.getElementById('guide-frame');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    const cameraControls = document.getElementById('camera-controls');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const captureBtn = document.getElementById('capture-btn');
    const retakeBtn = document.getElementById('retake-btn');
    const usePhotoBtn = document.getElementById('use-photo-btn');
    const stopCameraBtn = document.getElementById('stop-camera-btn');
    const cameraModeBtn = document.getElementById('camera-mode-btn');
    const fileModeBtn = document.getElementById('file-mode-btn');
    const cameraInterface = document.getElementById('camera-interface');
    const fileInterface = document.getElementById('file-interface');
    
    let stream = null;
    let capturedImageData = null;
    
    const ocrForm = document.getElementById('ocr-form');
    const imageInput = document.getElementById('image-input');
    const ocrSpinner = document.getElementById('ocr-spinner');
    const ocrError = document.getElementById('ocr-error');
    const licensePlateInput = document.getElementById('license-plate-input');
    const checkinForm = document.getElementById('checkin-form');
    const checkinSuccess = document.getElementById('checkin-success');
    const parkingTableBody = document.getElementById('parking-table-body');

    // --- 1. OCR Processing ---
    ocrForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!imageInput.files || imageInput.files.length === 0) {
            showError('まず画像を選択してください。');
            return;
        }

        const formData = new FormData();
        formData.append('file', imageInput.files[0]);

        // Show spinner and hide previous results
        ocrSpinner.classList.remove('d-none');
        ocrError.classList.add('d-none');
        licensePlateInput.value = '';

        try {
            const response = await fetch('/upload-image/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('サーバーでエラーが発生しました。');
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            if (result.license_plate && result.confidence !== 'failed') {
                licensePlateInput.value = result.license_plate;
                
                // Display full license plate information if available
                let plateInfo = result.license_plate;
                if (result.plate_parts) {
                    const parts = result.plate_parts;
                    let fullPlate = '';
                    if (parts.area) fullPlate += parts.area + ' ';
                    if (parts.classification) fullPlate += parts.classification + ' ';
                    if (parts.hiragana) fullPlate += parts.hiragana + ' ';
                    if (parts.number) fullPlate += parts.number;
                    
                    if (fullPlate.trim()) {
                        plateInfo = fullPlate.trim();
                        // Update the input with full information
                        licensePlateInput.value = plateInfo;
                    }
                }
                
                // Show confidence level and confirm if low
                if (result.confidence === 'low') {
                    if (confirm(`認識結果: ${plateInfo}\nこの情報で正しいですか？\n（違う場合は「キャンセル」を押して手動で修正してください）`)) {
                        // Auto submit check-in
                        checkinForm.dispatchEvent(new Event('submit'));
                    } else {
                        // Focus on input for manual correction
                        licensePlateInput.focus();
                        licensePlateInput.select();
                    }
                } else if (result.confidence === 'high') {
                    // High confidence - show success and auto submit
                    showSuccess(`認識成功: ${plateInfo}`);
                    checkinForm.dispatchEvent(new Event('submit'));
                } else if (result.confidence === 'medium') {
                    // Medium confidence - show result and ask for confirmation
                    showSuccess(`認識結果: ${plateInfo} (中程度の信頼度)`);
                    licensePlateInput.focus();
                }
            } else {
                // OCR failed - prompt for manual input
                showError(result.message || 'ナンバープレートを認識できませんでした。手動で入力してください。');
                licensePlateInput.focus();
            }

        } catch (error) {
            showError(error.message);
        } finally {
            ocrSpinner.classList.add('d-none');
        }
    });

    // --- 2. Check-in Processing ---
    checkinForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const plate = licensePlateInput.value.trim();
        if (!plate) {
            alert('ナンバープレートが入力されていません。');
            return;
        }

        try {
            const response = await fetch('/api/check-in', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ license_plate: plate }),
            });

            const result = await response.json();

            if (response.ok) {
                showSuccess(`車両 (ナンバー: ${result.license_plate}) の入庫を記録しました。`);
                licensePlateInput.value = ''; // Clear input after success
                loadParkingData(); // Refresh table
            } else {
                alert(result.detail || '入庫処理に失敗しました。');
            }
        } catch (error) {
            alert('エラーが発生しました: ' + error.message);
        }
    });

    // --- 3. Load Parking Data ---
    async function loadParkingData() {
        try {
            const response = await fetch('/api/parking-data');
            const data = await response.json();

            parkingTableBody.innerHTML = ''; // Clear existing data

            if (data.length === 0) {
                parkingTableBody.innerHTML = '<tr><td colspan="3" class="text-center">現在、駐車中の車両はありません。</td></tr>';
            }

            data.forEach(item => {
                const checkInTime = new Date(item.check_in).toLocaleString('ja-JP');
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.license_plate}</td>
                    <td>${checkInTime}</td>
                    <td>
                        <button class="btn btn-danger btn-sm checkout-btn" data-id="${item.id}">出庫</button>
                    </td>
                `;
                parkingTableBody.appendChild(row);
            });

        } catch (error) {
            console.error('駐車データの読み込みに失敗しました:', error);
        }
    }

    // --- 4. Check-out Processing ---
    parkingTableBody.addEventListener('click', async (e) => {
        if (e.target && e.target.classList.contains('checkout-btn')) {
            const logId = e.target.getAttribute('data-id');
            if (confirm('この車両を出庫させますか？')) {
                try {
                    const response = await fetch(`/api/check-out/${logId}`, {
                        method: 'POST',
                    });
                    if (response.ok) {
                        showSuccess('出庫処理が完了しました。');
                        loadParkingData(); // Refresh the list
                    } else {
                        const result = await response.json();
                        alert(result.detail || '出庫処理に失敗しました。');
                    }
                } catch (error) {
                     alert('エラーが発生しました: ' + error.message);
                }
            }
        }
    });

    // --- Utility Functions ---
    function showError(message) {
        ocrError.textContent = message;
        ocrError.classList.remove('d-none');
    }

    function showSuccess(message) {
        checkinSuccess.textContent = message;
        checkinSuccess.classList.remove('d-none');
        // Hide after 3 seconds
        setTimeout(() => {
            checkinSuccess.classList.add('d-none');
        }, 3000);
    }

    // Initial load of parking data
    loadParkingData();
    
    // --- Camera Functions ---
    async function startCamera() {
        try {
            // Request camera permission
            const constraints = {
                video: {
                    facingMode: 'environment', // Use back camera
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            };
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraPreview.srcObject = stream;
            
            // Show camera interface
            cameraPlaceholder.style.display = 'none';
            cameraPreview.style.display = 'block';
            guideFrame.style.display = 'block';
            cameraControls.style.display = 'block';
            captureBtn.style.display = 'inline-block';
            retakeBtn.style.display = 'none';
            usePhotoBtn.style.display = 'none';
            
        } catch (error) {
            console.error('Camera access error:', error);
            showError('カメラへのアクセスが拒否されました。ブラウザの設定を確認してください。');
        }
    }
    
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        cameraPreview.style.display = 'none';
        captureCanvas.style.display = 'none';
        guideFrame.style.display = 'none';
        cameraControls.style.display = 'none';
        cameraPlaceholder.style.display = 'block';
        capturedImageData = null;
    }
    
    function capturePhoto() {
        const ctx = captureCanvas.getContext('2d');
        captureCanvas.width = cameraPreview.videoWidth;
        captureCanvas.height = cameraPreview.videoHeight;
        ctx.drawImage(cameraPreview, 0, 0);
        
        // Convert to blob for upload
        captureCanvas.toBlob((blob) => {
            capturedImageData = blob;
        }, 'image/jpeg', 0.95);
        
        // Show captured image
        cameraPreview.style.display = 'none';
        captureCanvas.style.display = 'block';
        
        // Update buttons
        captureBtn.style.display = 'none';
        retakeBtn.style.display = 'inline-block';
        usePhotoBtn.style.display = 'inline-block';
    }
    
    function retakePhoto() {
        captureCanvas.style.display = 'none';
        cameraPreview.style.display = 'block';
        captureBtn.style.display = 'inline-block';
        retakeBtn.style.display = 'none';
        usePhotoBtn.style.display = 'none';
        capturedImageData = null;
    }
    
    async function usePhoto() {
        if (!capturedImageData) {
            showError('写真が撮影されていません。');
            return;
        }
        
        // Show loading spinner
        ocrSpinner.classList.remove('d-none');
        ocrError.classList.add('d-none');
        
        const formData = new FormData();
        formData.append('file', capturedImageData, 'capture.jpg');
        
        try {
            const response = await fetch('/upload-image/', {
                method: 'POST',
                body: formData,
            });
            
            if (!response.ok) {
                throw new Error('サーバーでエラーが発生しました。');
            }
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            if (result.license_plate && result.confidence !== 'failed') {
                licensePlateInput.value = result.license_plate;
                
                // Display full license plate information if available
                let plateInfo = result.license_plate;
                if (result.plate_parts) {
                    const parts = result.plate_parts;
                    let fullPlate = '';
                    if (parts.area) fullPlate += parts.area + ' ';
                    if (parts.classification) fullPlate += parts.classification + ' ';
                    if (parts.hiragana) fullPlate += parts.hiragana + ' ';
                    if (parts.number) fullPlate += parts.number;
                    
                    if (fullPlate.trim()) {
                        plateInfo = fullPlate.trim();
                        // Update the input with full information
                        licensePlateInput.value = plateInfo;
                    }
                }
                
                // Show confidence level and confirm if low
                if (result.confidence === 'low') {
                    if (confirm(`認識結果: ${plateInfo}\nこの情報で正しいですか？\n（違う場合は「キャンセル」を押して手動で修正してください）`)) {
                        // Auto submit check-in
                        checkinForm.dispatchEvent(new Event('submit'));
                    } else {
                        // Focus on input for manual correction
                        licensePlateInput.focus();
                        licensePlateInput.select();
                    }
                } else if (result.confidence === 'high') {
                    // High confidence - show success and auto submit
                    showSuccess(`認識成功: ${plateInfo}`);
                    checkinForm.dispatchEvent(new Event('submit'));
                } else if (result.confidence === 'medium') {
                    // Medium confidence - show result and ask for confirmation
                    showSuccess(`認識結果: ${plateInfo} (中程度の信頼度)`);
                    licensePlateInput.focus();
                }
                
                // Stop camera after successful recognition
                stopCamera();
            } else {
                // OCR failed - prompt for manual input
                showError(result.message || 'ナンバープレートを認識できませんでした。撮り直すか、手動で入力してください。');
                licensePlateInput.focus();
            }
            
        } catch (error) {
            showError(error.message);
        } finally {
            ocrSpinner.classList.add('d-none');
        }
    }
    
    // --- Event Listeners for Camera ---
    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);
    captureBtn.addEventListener('click', capturePhoto);
    retakeBtn.addEventListener('click', retakePhoto);
    usePhotoBtn.addEventListener('click', usePhoto);
    
    // Mode toggle buttons
    cameraModeBtn.addEventListener('click', () => {
        cameraModeBtn.classList.add('active');
        fileModeBtn.classList.remove('active');
        cameraInterface.style.display = 'block';
        fileInterface.style.display = 'none';
    });
    
    fileModeBtn.addEventListener('click', () => {
        fileModeBtn.classList.add('active');
        cameraModeBtn.classList.remove('active');
        fileInterface.style.display = 'block';
        cameraInterface.style.display = 'none';
        stopCamera(); // Stop camera when switching to file mode
    });
});
