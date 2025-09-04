import cv2
import numpy as np
import pytesseract
import re
from collections import Counter

async def fast_high_precision_ocr(file_contents):
    """
    Fast, high-precision OCR using optimized Tesseract with intelligent post-processing
    """
    np_arr = np.frombuffer(file_contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}
    
    height, width = img.shape[:2]
    
    # Auto-rotate if portrait
    if height > width * 1.5:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height
    
    # High resolution scaling (faster than EasyOCR method)
    target_width = 1600
    if width < target_width:
        scale = target_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optimized preprocessing methods (fast but effective)
    def preprocess_v1(img):
        # CLAHE + Gaussian blur
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        return sharpened
    
    def preprocess_v2(img):
        # Bilateral filter + morphological ops
        bilateral = cv2.bilateralFilter(img, 9, 75, 75)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
        return morph
    
    def preprocess_v3(img):
        # Simple but effective: denoise + sharpen
        denoised = cv2.fastNlMeansDenoising(img, h=10)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        return sharpened
    
    # Fast OCR with multiple approaches
    all_results = []
    
    # Process with different preprocessing
    preprocessing_methods = [
        ("original", gray),
        ("clahe_sharp", preprocess_v1(gray)),
        ("bilateral_morph", preprocess_v2(gray)),
        ("denoise_sharp", preprocess_v3(gray))
    ]
    
    for method_name, processed_img in preprocessing_methods:
        # Try different binarization methods
        binary_methods = []
        
        # Otsu thresholding
        _, otsu = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_methods.append(("otsu", otsu))
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        binary_methods.append(("adaptive", adaptive))
        
        for binary_name, binary_img in binary_methods:
            # Try normal and inverted
            for invert in [False, True]:
                test_img = cv2.bitwise_not(binary_img) if invert else binary_img
                
                # Multiple Tesseract configurations optimized for license plates
                configs = [
                    '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-',  # Single line numbers
                    '--oem 3 --psm 8 -c tesseract_char_whitelist=0123456789-',  # Single word numbers
                    '--oem 3 --psm 6',  # Uniform block
                ]
                
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(test_img, config=config).strip()
                        if text:
                            all_results.append({
                                'text': text,
                                'method': f"{method_name}_{binary_name}_inv{invert}",
                                'config': config
                            })
                    except:
                        continue
    
    # Advanced Japanese OCR for full plate info (only if needed)
    full_plate_info = {'area': '', 'classification': '', 'hiragana': '', 'full_text': ''}
    
    try:
        # Try Japanese OCR with timeout
        jpn_config = '--oem 3 --psm 6'
        full_text = pytesseract.image_to_string(gray, config=jpn_config, lang='jpn+eng', timeout=10)
        if full_text.strip():
            full_plate_info['full_text'] = full_text.strip().replace('\n', ' ')
            
            # Extract components
            area_patterns = [
                '東京', '大阪', '神戸', '横浜', '川崎', '千葉', '埼玉', '茨城', '栃木', '群馬',
                '山梨', '長野', '新潟', '富山', '石川', '福井', '静岡', '愛知', '三重', '滋賀',
                '京都', '兵庫', '奈良', '和歌山', '鳥取', '島根', '岡山', '広島', '山口',
                '徳島', '香川', '愛媛', '高知', '福岡', '佐賀', '長崎', '熊本', '大分', '宮崎', '鹿児島', '沖縄'
            ]
            
            for area in area_patterns:
                if area in full_text:
                    full_plate_info['area'] = area
                    break
            
            # Classification (3 digits)
            class_match = re.search(r'(\d{3})', full_text)
            if class_match:
                full_plate_info['classification'] = class_match.group(1)
            
            # Hiragana
            hiragana_match = re.search(r'([あ-ん])', full_text)
            if hiragana_match:
                full_plate_info['hiragana'] = hiragana_match.group(1)
    except:
        pass
    
    # Process number results
    def extract_license_numbers(text_results):
        numbers = []
        
        for result in text_results:
            text = result['text']
            
            # Extract number patterns
            patterns = [
                r'(\d{2}[-ー]\d{2})',  # XX-XX format
                r'(\d{4})',            # XXXX format
                r'(\d{2}\s*\d{2})',    # XX XX format
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # Clean and format
                    cleaned = re.sub(r'[^\d-]', '', match)
                    if len(cleaned) == 4 and cleaned.isdigit():
                        formatted = f"{cleaned[:2]}-{cleaned[2:]}"
                    elif len(cleaned) == 5 and '-' in cleaned:
                        formatted = cleaned
                    else:
                        continue
                    
                    # Validate
                    if formatted not in ['00-00', '11-11', '99-99']:
                        numbers.append(formatted)
        
        return numbers
    
    # Get numbers
    detected_numbers = extract_license_numbers(all_results)
    
    # Character correction for common OCR mistakes
    def correct_number(num):
        corrections = {
            '0': '8',  # Common mistake
            '6': '8',
            '5': '8', 
            'S': '8',
            'B': '8',
            'I': '1',
            'l': '1',
            'O': '0',
        }
        corrected = num
        for wrong, right in corrections.items():
            corrected = corrected.replace(wrong, right)
        return corrected
    
    # Apply corrections and vote for best result
    corrected_numbers = []
    for num in detected_numbers:
        corrected_numbers.extend([num, correct_number(num)])
    
    if corrected_numbers:
        # Use voting system
        counter = Counter(corrected_numbers)
        most_common = counter.most_common(1)[0]
        best_number = most_common[0]
        vote_count = most_common[1]
        
        # Determine confidence
        if vote_count >= 3:
            confidence = "high"
        elif vote_count >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Build display text
        display_parts = []
        if full_plate_info['area']: display_parts.append(full_plate_info['area'])
        if full_plate_info['classification']: display_parts.append(full_plate_info['classification'])
        if full_plate_info['hiragana']: display_parts.append(full_plate_info['hiragana'])
        display_parts.append(best_number)
        
        display_text = ' '.join(display_parts)
        
        return {
            "license_plate": best_number,
            "full_text": display_text if len(display_parts) > 1 else best_number,
            "confidence": confidence,
            "area": full_plate_info['area'],
            "classification": full_plate_info['classification'],
            "hiragana": full_plate_info['hiragana'],
            "number": best_number,
            "debug_info": f"Found {len(detected_numbers)} matches, best: {best_number} ({vote_count} votes)",
            "plate_parts": {
                "area": full_plate_info['area'],
                "classification": full_plate_info['classification'],
                "hiragana": full_plate_info['hiragana'],
                "number": best_number
            }
        }
    else:
        return {
            "license_plate": "",
            "full_text": "認識できませんでした",
            "confidence": "failed",
            "message": "画像をもう一度撮影するか、手動で入力してください。明るい場所で、ナンバープレート全体が画面に収まるようにしてください。"
        }