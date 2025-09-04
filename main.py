import os
import datetime
import pytesseract
import easyocr
from fastapi import FastAPI, File, UploadFile, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
import cv2
import numpy as np
import io
import csv
import re
from collections import Counter

# --- Configuration ---
# DATABASE_URL will be provided by Railway's environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

# Railway (and Heroku) often use the "postgres://" scheme, which SQLAlchemy 1.4+
# expects to be "postgresql://". This code handles that translation.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

Base = declarative_base()
engine = create_engine(DATABASE_URL)

# --- Pydantic Models ---
class CheckInRequest(BaseModel):
    license_plate: str

class ParkingLogResponse(BaseModel):
    id: int
    license_plate: str
    check_in: datetime.datetime

    class Config:
        from_attributes = True
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Tesseract Configuration ---
# For local macOS with Homebrew
if os.path.exists('/opt/homebrew/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
# For Railway/Debian, the path is usually /usr/bin/tesseract, which is in the PATH
# so we don't need to set it explicitly if the Docker build is correct.

# Fast OCR function integrated directly
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
    
    # High resolution scaling
    target_width = 1600
    if width < target_width:
        scale = target_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optimized preprocessing methods
    def preprocess_v1(img):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        return sharpened
    
    def preprocess_v2(img):
        bilateral = cv2.bilateralFilter(img, 9, 75, 75)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
        return morph
    
    # Process with different methods
    all_results = []
    preprocessing_methods = [
        ("original", gray),
        ("clahe_sharp", preprocess_v1(gray)),
        ("bilateral_morph", preprocess_v2(gray))
    ]
    
    for method_name, processed_img in preprocessing_methods:
        # Try different binarization
        _, otsu = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        for binary_name, binary_img in [("otsu", otsu), ("adaptive", adaptive)]:
            for invert in [False, True]:
                test_img = cv2.bitwise_not(binary_img) if invert else binary_img
                
                configs = [
                    '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-',
                    '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789-',
                    '--oem 3 --psm 6',
                ]
                
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(test_img, config=config).strip()
                        if text:
                            all_results.append({'text': text, 'method': f"{method_name}_{binary_name}"})
                    except:
                        continue
    
    # Japanese OCR for full plate info
    full_plate_info = {'area': '', 'classification': '', 'hiragana': '', 'full_text': ''}
    try:
        full_text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6', lang='jpn+eng', timeout=10)
        if full_text.strip():
            full_plate_info['full_text'] = full_text.strip().replace('\n', ' ')
            
            area_patterns = ['東京', '大阪', '神戸', '横浜', '川崎', '千葉', '埼玉', '茨城', '栃木', '群馬',
                           '山梨', '長野', '新潟', '富山', '石川', '福井', '静岡', '愛知', '三重', '滋賀',
                           '京都', '兵庫', '奈良', '和歌山', '鳥取', '島根', '岡山', '広島', '山口',
                           '徳島', '香川', '愛媛', '高知', '福岡', '佐賀', '長崎', '熊本', '大分', '宮崎', '鹿児島', '沖縄']
            
            for area in area_patterns:
                if area in full_text:
                    full_plate_info['area'] = area
                    break
            
            class_match = re.search(r'(\d{3})', full_text)
            if class_match:
                full_plate_info['classification'] = class_match.group(1)
            
            hiragana_match = re.search(r'([あ-ん])', full_text)
            if hiragana_match:
                full_plate_info['hiragana'] = hiragana_match.group(1)
    except:
        pass
    
    # Extract numbers
    detected_numbers = []
    for result in all_results:
        text = result['text']
        patterns = [r'(\d{2}[-ー]\d{2})', r'(\d{4})', r'(\d{2}\s*\d{2})']
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                cleaned = re.sub(r'[^\d-]', '', match)
                if len(cleaned) == 4 and cleaned.isdigit():
                    formatted = f"{cleaned[:2]}-{cleaned[2:]}"
                elif len(cleaned) == 5 and '-' in cleaned:
                    formatted = cleaned
                else:
                    continue
                
                if formatted not in ['00-00', '11-11', '99-99']:
                    detected_numbers.append(formatted)
    
    # Character correction
    def correct_number(num):
        corrections = {'0': '8', '6': '8', '5': '8', 'S': '8', 'B': '8', 'I': '1', 'l': '1', 'O': '0'}
        corrected = num
        for wrong, right in corrections.items():
            corrected = corrected.replace(wrong, right)
        return corrected
    
    # Apply corrections and vote
    corrected_numbers = []
    for num in detected_numbers:
        corrected_numbers.extend([num, correct_number(num)])
    
    if corrected_numbers:
        counter = Counter(corrected_numbers)
        most_common = counter.most_common(1)[0]
        best_number = most_common[0]
        vote_count = most_common[1]
        
        confidence = "high" if vote_count >= 3 else "medium" if vote_count >= 2 else "low"
        
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

# --- Database Model ---
class ParkingLog(Base):
    __tablename__ = "parking_logs"
    id = Column(Integer, primary_key=True, index=True)
    license_plate = Column(String, index=True, nullable=False)
    check_in = Column(DateTime, default=datetime.datetime.utcnow)
    check_out = Column(DateTime, nullable=True)

# --- FastAPI App Initialization ---
app = FastAPI(title="Park2 App")

@app.on_event("startup")
def on_startup():
    # Create the database tables
    Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Helper Functions ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- API Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Main page, displays the parking management UI.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-image/")
async def ocr_from_image(file: UploadFile = File(...)):
    """
    Ultra-high precision OCR using EasyOCR + advanced preprocessing for Japanese license plates.
    """
    contents = await file.read()
    
    try:
        result = await fast_high_precision_ocr(contents)
        
        # Handle errors
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        print(f"OCR processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR処理中にエラーが発生しました: {str(e)}")

@app.get("/api/parking-data", response_model=list[ParkingLogResponse])
def get_parking_data(db: Session = Depends(get_db)):
    """
    Returns a list of all currently parked vehicles (check_out is null).
    """
    return db.query(ParkingLog).filter(ParkingLog.check_out == None).order_by(ParkingLog.check_in.desc()).all()

@app.post("/api/check-in", response_model=ParkingLogResponse)
def check_in_vehicle(request: CheckInRequest, db: Session = Depends(get_db)):
    """
    Checks in a new vehicle.
    """
    # Check if vehicle is already checked in
    existing_log = db.query(ParkingLog).filter(
        ParkingLog.license_plate == request.license_plate,
        ParkingLog.check_out == None
    ).first()
    if existing_log:
        raise HTTPException(status_code=400, detail="This vehicle is already checked in.")

    new_log = ParkingLog(license_plate=request.license_plate)
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    return new_log

@app.post("/api/check-out/{log_id}")
def check_out_vehicle(log_id: int, db: Session = Depends(get_db)):
    """
    Checks out a vehicle by setting the check_out time.
    """
    log_to_update = db.query(ParkingLog).filter(ParkingLog.id == log_id).first()
    if not log_to_update:
        raise HTTPException(status_code=404, detail="Log not found.")
    if log_to_update.check_out is not None:
        raise HTTPException(status_code=400, detail="This vehicle has already been checked out.")

    log_to_update.check_out = datetime.datetime.utcnow()
    db.commit()
    return {"message": "Check-out successful."}

@app.get("/download/csv")
def download_csv(db: Session = Depends(get_db)):
    """
    Generates and streams a CSV file of all parking logs.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(["ID", "License Plate", "Check-In Time (UTC)", "Check-Out Time (UTC)"])
    
    logs = db.query(ParkingLog).order_by(ParkingLog.check_in.desc()).all()
    for log in logs:
        writer.writerow([
            log.id,
            log.license_plate,
            log.check_in.strftime("%Y-%m-%d %H:%M:%S") if log.check_in else "",
            log.check_out.strftime("%Y-%m-%d %H:%M:%S") if log.check_out else "Not Checked Out"
        ])
    
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=parking_logs_{datetime.date.today()}.csv"}
    )
