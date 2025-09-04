
document.addEventListener('DOMContentLoaded', function () {
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
                
                // Show confidence level and confirm if low
                if (result.confidence === 'low') {
                    if (confirm(`認識結果: ${result.license_plate}\nこの番号で正しいですか？\n（違う場合は「キャンセル」を押して手動で修正してください）`)) {
                        // Auto submit check-in
                        checkinForm.dispatchEvent(new Event('submit'));
                    } else {
                        // Focus on input for manual correction
                        licensePlateInput.focus();
                        licensePlateInput.select();
                    }
                } else if (result.confidence === 'high') {
                    // High confidence - show success and auto submit
                    showSuccess(`認識成功: ${result.license_plate}`);
                    checkinForm.dispatchEvent(new Event('submit'));
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
});
