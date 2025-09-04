import cv2
import numpy as np
import easyocr
import re
from collections import Counter

async def ultra_high_precision_ocr(file_contents):
    """
    Ultra-high precision OCR using EasyOCR + advanced image processing
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
    
    # Super high resolution scaling
    target_width = 2400  # Even higher resolution
    if width < target_width:
        scale = target_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # EasyOCR Results Storage
    all_results = []
    
    # Preprocessing methods for different lighting conditions
    def get_image_variants(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variants = []
        
        # Original grayscale
        variants.append(("original", gray))
        
        # CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        variants.append(("clahe", enhanced))
        
        # Gaussian blur + sharpen
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharpened = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)
        variants.append(("sharpened", sharpened))
        
        # Bilateral filter
        bilateral = cv2.bilateralFilter(gray, 15, 80, 80)
        variants.append(("bilateral", bilateral))
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        variants.append(("morphological", morph))
        
        return variants
    
    # Initialize EasyOCR if available
    try:
        if 'easyocr_reader' not in globals():
            easyocr_reader = easyocr.Reader(['ja', 'en'], gpu=False)
        
        # Get image variants
        variants = get_image_variants(img)
        
        for variant_name, variant_img in variants:
            try:
                # EasyOCR detection with different confidence thresholds
                for confidence_threshold in [0.1, 0.3, 0.5]:
                    results = easyocr_reader.readtext(
                        variant_img,
                        detail=1,
                        paragraph=False,
                        width_ths=0.7,
                        height_ths=0.7,
                        decoder='greedy',
                        beamWidth=5,
                        batch_size=1,
                        workers=1,
                        allowlist='0123456789-あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん東京大阪神戸横浜川崎千葉埼玉茨城栃木群馬山梨長野新潟富山石川福井静岡愛知三重滋賀京都兵庫奈良和歌山鳥取島根岡山広島山口徳島香川愛媛高知福岡佐賀長崎熊本大分宮崎鹿児島沖縄'
                    )
                    
                    for (bbox, text, conf) in results:
                        if conf > confidence_threshold:
                            cleaned_text = text.strip()
                            print(f"EasyOCR found: '{cleaned_text}' confidence: {conf:.3f} variant: {variant_name}")
                            all_results.append({
                                'text': cleaned_text,
                                'confidence': conf,
                                'method': f'easyocr_{variant_name}_{confidence_threshold}',
                                'bbox': bbox
                            })
                
            except Exception as e:
                print(f"EasyOCR error on {variant_name}: {e}")
                continue
                
    except Exception as e:
        print(f"EasyOCR initialization error: {e}")
    
    # Parse results and extract license plate components
    def parse_license_plate_info(text_results):
        # Combine all detected texts
        combined_text = ' '.join([r['text'] for r in text_results])
        print(f"Combined text: '{combined_text}'")
        
        plate_info = {
            'area': '',
            'classification': '',
            'hiragana': '',
            'number': '',
            'full_text': combined_text
        }
        
        # Extract area names (地域名)
        area_patterns = [
            '東京', '大阪', '神戸', '横浜', '川崎', '千葉', '埼玉', '茨城', '栃木', '群馬',
            '山梨', '長野', '新潟', '富山', '石川', '福井', '静岡', '愛知', '三重', '滋賀',
            '京都', '兵庫', '奈良', '和歌山', '鳥取', '島根', '岡山', '広島', '山口',
            '徳島', '香川', '愛媛', '高知', '福岡', '佐賀', '長崎', '熊本', '大分', '宮崎', '鹿児島', '沖縄'
        ]
        for area in area_patterns:
            if area in combined_text:
                plate_info['area'] = area
                break
        
        # Extract 3-digit classification
        class_matches = re.findall(r'\b(\d{3})\b', combined_text)
        if class_matches:
            plate_info['classification'] = class_matches[0]
        
        # Extract hiragana
        hiragana_matches = re.findall(r'[あ-ん]', combined_text)
        if hiragana_matches:
            plate_info['hiragana'] = hiragana_matches[0]
        
        # Extract 4-digit number patterns
        number_patterns = [
            r'(\d{2}[-ー]\d{2})',  # XX-XX or XX－XX
            r'(\d{4})',           # XXXX
        ]
        
        numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, combined_text)
            for match in matches:
                # Convert to standard format
                if len(match) == 4 and match.isdigit():
                    formatted = f"{match[:2]}-{match[2:]}"
                else:
                    formatted = match.replace('ー', '-')  # Replace full-width dash
                numbers.append(formatted)
        
        # Character correction for common misreads
        def correct_number(num):
            corrections = {
                '0': '8',  # 0 might be 8
                '6': '8',  # 6 might be 8
                '5': '8',  # 5 might be 8
                'S': '8',  # S might be 8
                'B': '8',  # B might be 8
                'I': '1',  # I might be 1
                'l': '1',  # l might be 1
                'O': '0',  # O might be 0
            }
            corrected = num
            for wrong, right in corrections.items():
                corrected = corrected.replace(wrong, right)
            return corrected
        
        # Apply corrections and find best number
        corrected_numbers = []
        for num in numbers:
            corrected_numbers.extend([num, correct_number(num)])
        
        if corrected_numbers:
            # Use most common result
            counter = Counter(corrected_numbers)
            most_common = counter.most_common(1)[0]
            plate_info['number'] = most_common[0]
        
        return plate_info
    
    # Analyze all results
    if all_results:
        plate_info = parse_license_plate_info(all_results)
        
        # Calculate overall confidence
        avg_confidence = sum(r['confidence'] for r in all_results) / len(all_results)
        
        if avg_confidence > 0.7:
            confidence = "high"
        elif avg_confidence > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Build display text
        display_parts = []
        if plate_info['area']: display_parts.append(plate_info['area'])
        if plate_info['classification']: display_parts.append(plate_info['classification'])
        if plate_info['hiragana']: display_parts.append(plate_info['hiragana'])
        if plate_info['number']: display_parts.append(plate_info['number'])
        
        display_text = ' '.join(display_parts) if display_parts else plate_info['number']
        
        return {
            "license_plate": plate_info['number'] or "認識できませんでした",
            "full_text": display_text,
            "confidence": confidence,
            "area": plate_info['area'],
            "classification": plate_info['classification'],
            "hiragana": plate_info['hiragana'],
            "number": plate_info['number'],
            "debug_info": f"EasyOCR found {len(all_results)} text elements, avg confidence: {avg_confidence:.3f}",
            "plate_parts": {
                "area": plate_info['area'],
                "classification": plate_info['classification'],
                "hiragana": plate_info['hiragana'],
                "number": plate_info['number']
            }
        }
    else:
        return {
            "license_plate": "",
            "full_text": "認識できませんでした",
            "confidence": "failed",
            "message": "画像をもう一度撮影するか、手動で入力してください。明るい場所で、ナンバープレート全体が画面に収まるようにしてください。"
        }