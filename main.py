import os
import datetime
import pytesseract
import cv2
import numpy as np
import re
from collections import Counter
from fastapi import FastAPI, File, UploadFile, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
import io
import csv

# --- Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

Base = declarative_base()
engine = create_engine(DATABASE_URL)

# --- Tesseract Configuration ---
if os.path.exists('/opt/homebrew/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# --- Database Model ---
class ParkingLog(Base):
    __tablename__ = "parking_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    license_plate = Column(String, index=True)
    check_in = Column(DateTime, default=datetime.datetime.utcnow)
    check_out = Column(DateTime, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic Models ---
class ParkingLogResponse(BaseModel):
    id: int
    license_plate: str
    check_in: datetime.datetime
    check_out: datetime.datetime = None

    class Config:
        from_attributes = True

# --- FastAPI App ---
app = FastAPI()

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-image/")
async def ocr_from_image(file: UploadFile = File(...)):
    """
    Simple OCR function for testing
    """
    contents = await file.read()
    
    try:
        # Simple image processing
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        
        # Auto-rotate if portrait mode
        height, width = img.shape[:2]
        if height > width * 1.5:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        # High resolution scaling for better OCR
        target_width = 1600
        if width < target_width:
            scale = target_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Advanced preprocessing methods
        def preprocess_clahe_sharp(img):
            # CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img)
            
            # Unsharp masking for better edge definition
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
            sharpened = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)
            
            return sharpened
        
        def preprocess_bilateral_morph(img):
            # Bilateral filter for noise reduction while preserving edges
            bilateral = cv2.bilateralFilter(img, 11, 80, 80)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            return morph
        
        def preprocess_denoise_enhance(img):
            # Advanced denoising
            denoised = cv2.fastNlMeansDenoising(img, h=10)
            
            # Custom sharpening kernel
            kernel_sharpen = np.array([[-1,-1,-1],
                                     [-1, 9,-1], 
                                     [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
            
            return sharpened
        
        # Process with multiple preprocessing methods
        preprocessing_methods = [
            ("original", gray),
            ("clahe_sharp", preprocess_clahe_sharp(gray)),
            ("bilateral_morph", preprocess_bilateral_morph(gray)),
            ("denoise_enhance", preprocess_denoise_enhance(gray))
        ]
        
        all_results = []
        
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
                    
                    # Multiple OCR configurations to catch different text layouts
                    ocr_configs = [
                        ('numbers_only', '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789-'),
                        ('single_line', '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-'),
                        ('uniform_block', '--oem 3 --psm 6'),
                        ('sparse_text', '--oem 3 --psm 11'),
                        ('single_word', '--oem 3 --psm 8'),
                        ('raw_line', '--oem 3 --psm 13'),
                    ]
                    
                    for config_name, config in ocr_configs:
                        try:
                            text = pytesseract.image_to_string(test_img, config=config)
                            if text.strip():
                                all_results.append({
                                    'text': text.strip(),
                                    'method': f"{method_name}_{binary_name}_inv{invert}_{config_name}"
                                })
                        except Exception as e:
                            print(f"OCR config {config_name} failed: {e}")
                            continue
        
        # Enhanced OCR processing
        try:
            # Collect all detected numbers from different methods
            all_numbers = []
            
            # Character correction for common OCR mistakes
            def correct_ocr_text(text):
                corrections = {
                    'O': '0', 'o': '0',  # Letter O to number 0
                    'I': '1', 'l': '1', '|': '1',  # Letters to number 1
                    'S': '8', 's': '8',  # Letter S to number 8
                    'B': '8',  # Letter B to number 8
                    'Z': '2',  # Letter Z to number 2
                    'G': '6',  # Letter G to number 6
                    'D': '0',  # Letter D to number 0
                    'T': '7',  # Letter T to number 7
                    'A': '4',  # Letter A to number 4
                    'E': '3',  # Letter E to number 3
                }
                corrected = text
                for wrong, right in corrections.items():
                    corrected = corrected.replace(wrong, right)
                return corrected
            
            for result in all_results:
                text = result['text']
                print(f"Raw OCR text: '{text}' from {result['method']}")
                
                # Apply character corrections
                corrected_text = correct_ocr_text(text)
                if corrected_text != text:
                    print(f"After correction: '{corrected_text}'")
                
                # Extract number patterns with more flexible regex
                patterns = [
                    r'\d{2}[-ー]\d{2}',  # XX-XX or XX－XX (full-width dash)
                    r'\d{4}',            # XXXX
                    r'\d{2}\s+\d{2}',    # XX XX (with spaces)
                    r'\d{1,2}[-ー]?\d{1,2}', # More flexible pattern
                ]
                
                for pattern in patterns:
                    numbers = re.findall(pattern, corrected_text)
                    for num in numbers:
                        # Clean up the number
                        cleaned = re.sub(r'[^\d-]', '', num)
                        if len(cleaned) >= 3:  # At least 3 digits
                            all_numbers.append(cleaned)
                            print(f"Found '{cleaned}' using {result['method']} with pattern {pattern}")
                
                # Also try to find any sequence of digits
                digit_sequences = re.findall(r'\d+', corrected_text)
                for seq in digit_sequences:
                    if len(seq) >= 3:  # At least 3 digits
                        all_numbers.append(seq)
                        print(f"Found digit sequence '{seq}' from {result['method']}")
            
            print(f"All numbers found: {all_numbers}")
            
            # Choose the best result with improved validation
            if all_numbers:
                from collections import Counter
                
                # Clean and validate numbers
                valid_numbers = []
                for num in all_numbers:
                    # Remove any non-digit characters except hyphen
                    cleaned = re.sub(r'[^\d-]', '', num)
                    
                    # Convert various formats to standard XX-XX
                    if len(cleaned) == 3:
                        # XXX -> 0X-XX or XX-X
                        if cleaned[0] == '0':
                            formatted = f"0{cleaned[1]}-{cleaned[2]}0"  # Likely 0X-X0
                        else:
                            formatted = f"{cleaned[:2]}-{cleaned[2]}0"  # XX-X0
                    elif len(cleaned) == 4:
                        formatted = f"{cleaned[:2]}-{cleaned[2:]}"  # XXXX -> XX-XX
                    elif len(cleaned) == 5 and '-' in cleaned:
                        formatted = cleaned  # Already XX-XX format
                    elif len(cleaned) >= 5:
                        # Take first 4 digits
                        digits_only = re.sub(r'[^\d]', '', cleaned)[:4]
                        if len(digits_only) == 4:
                            formatted = f"{digits_only[:2]}-{digits_only[2:]}"
                        else:
                            continue
                    else:
                        continue  # Skip invalid patterns
                    
                    # Validate the result looks like a license plate number
                    if re.match(r'\d{2}-\d{2}', formatted) and formatted not in ['00-00', '11-11']:
                        valid_numbers.append(formatted)
                
                print(f"Valid numbers after formatting: {valid_numbers}")
                
                if valid_numbers:
                    counter = Counter(valid_numbers)
                    most_common = counter.most_common(1)[0]
                    best_number = most_common[0]
                    confidence_level = "high" if most_common[1] >= 2 else "medium"
                    
                return {
                    "license_plate": best_number,
                    "full_text": best_number,
                    "confidence": confidence_level,
                    "debug_info": f"Found {len(all_numbers)} results, best: {best_number} (appeared {most_common[1]} times)",
                    "area": "",
                    "classification": "",
                    "hiragana": "",
                    "number": best_number,
                    "plate_parts": {
                        "area": "",
                        "classification": "",
                        "hiragana": "",
                        "number": best_number
                    }
                }
            else:
                return {
                    "license_plate": "",
                    "full_text": "認識できませんでした",
                    "confidence": "failed",
                    "message": "画像をもう一度撮影するか、手動で入力してください。"
                }
                
        except Exception as e:
            print(f"OCR error: {e}")
            return {
                "license_plate": "",
                "full_text": "認識できませんでした",
                "confidence": "failed",
                "message": "OCR処理中にエラーが発生しました。"
            }
            
    except Exception as e:
        print(f"Image processing error: {e}")
        raise HTTPException(status_code=500, detail="画像処理中にエラーが発生しました。")

@app.get("/api/parking-data", response_model=list[ParkingLogResponse])
def get_parking_data(db: Session = Depends(get_db)):
    return db.query(ParkingLog).filter(ParkingLog.check_out == None).order_by(ParkingLog.check_in.desc()).all()

@app.post("/api/check-in")
def check_in_vehicle(request: dict, db: Session = Depends(get_db)):
    license_plate = request.get("license_plate", "").strip()
    if not license_plate:
        raise HTTPException(status_code=400, detail="License plate is required")
    
    existing = db.query(ParkingLog).filter(
        ParkingLog.license_plate == license_plate,
        ParkingLog.check_out == None
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="This vehicle is already parked")
    
    new_log = ParkingLog(license_plate=license_plate)
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    
    return {"message": "Check-in successful", "license_plate": license_plate}

@app.post("/api/check-out/{log_id}")
def check_out_vehicle(log_id: int, db: Session = Depends(get_db)):
    log = db.query(ParkingLog).filter(ParkingLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Parking log not found")
    
    if log.check_out:
        raise HTTPException(status_code=400, detail="Vehicle already checked out")
    
    log.check_out = datetime.datetime.utcnow()
    db.commit()
    
    return {"message": "Check-out successful"}

@app.get("/download/csv")
def download_csv(db: Session = Depends(get_db)):
    logs = db.query(ParkingLog).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "License Plate", "Check In", "Check Out"])
    
    for log in logs:
        writer.writerow([log.id, log.license_plate, log.check_in, log.check_out or ""])
    
    content = output.getvalue()
    output.close()
    
    return StreamingResponse(
        io.BytesIO(content.encode('utf-8')),
        media_type='text/csv',
        headers={"Content-Disposition": "attachment; filename=parking_data.csv"}
    )