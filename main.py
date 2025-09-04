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

# Google Cloud Vision import
try:
    from google.cloud import vision
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("Google Cloud Vision not available, using Tesseract only")

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
    Practical OCR - Fast and reliable for 100 vehicles parking lot
    """
    contents = await file.read()
    
    try:
        # Simple but effective image processing (3-5 seconds total)
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        
        # Basic preprocessing only
        height, width = img.shape[:2]
        
        # Auto-rotate if portrait
        if height > width * 1.2:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        # Resize if too small
        if width < 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Character correction function
        def fix_common_mistakes(text):
            fixes = {'O': '0', 'I': '1', 'S': '8', 'B': '8', 'l': '1', 'o': '0', 'G': '6', 'Z': '2'}
            for wrong, right in fixes.items():
                text = text.replace(wrong, right)
            return text
        
        # Google Cloud Vision OCR function
        def try_google_vision_ocr(image_content):
            if not VISION_AVAILABLE:
                return []
            
            try:
                client = vision.ImageAnnotatorClient()
                image = vision.Image(content=image_content)
                response = client.text_detection(image=image)
                texts = response.text_annotations
                
                results = []
                for text in texts:
                    if text.description and text.description.strip():
                        results.append(text.description.strip())
                        print(f"Google Vision OCR: '{text.description.strip()}'")
                
                if response.error.message:
                    print(f"Google Vision API error: {response.error.message}")
                
                return results
            except Exception as e:
                print(f"Google Vision OCR error: {e}")
                return []
        
        # Debug: Save processed image for analysis
        debug_info = []
        
        # Try Google Cloud Vision first (most accurate)
        google_results = try_google_vision_ocr(contents)
        results = google_results.copy()
        
        # Multiple preprocessing methods for better success rate
        processed_images = []
        
        # Original enhanced
        processed_images.append(("enhanced", enhanced))
        
        # Binary threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("binary", binary))
        
        # Inverted binary
        binary_inv = cv2.bitwise_not(binary)
        processed_images.append(("binary_inv", binary_inv))
        
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        processed_images.append(("morph", morph))
        
        # Try Tesseract OCR on all processed images (in addition to Google Vision)
        tesseract_results = []
        
        for img_name, proc_img in processed_images:
            # Multiple OCR configurations
            configs = [
                ('psm6', '--oem 3 --psm 6'),  # Uniform block
                ('psm7', '--oem 3 --psm 7'),  # Single line
                ('psm8', '--oem 3 --psm 8'),  # Single word  
                ('psm8_nums', '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789-'),
                ('psm11', '--oem 3 --psm 11'), # Sparse text
                ('psm13', '--oem 3 --psm 13'), # Raw line
            ]
            
            for config_name, config in configs:
                try:
                    raw_text = pytesseract.image_to_string(proc_img, config=config)
                    if raw_text and raw_text.strip():
                        cleaned_text = raw_text.strip()
                        corrected_text = fix_common_mistakes(cleaned_text)
                        tesseract_results.append(corrected_text)
                        debug_info.append(f"Tesseract {img_name}_{config_name}: '{cleaned_text}' -> '{corrected_text}'")
                        print(f"Tesseract OCR ({img_name}_{config_name}): '{cleaned_text}' -> '{corrected_text}'")
                except Exception as e:
                    debug_info.append(f"Tesseract {img_name}_{config_name}: ERROR - {str(e)}")
                    continue
        
        # Combine all results (Google Vision + Tesseract)
        results.extend(tesseract_results)
        
        print(f"Google Vision results: {len(google_results)}")
        print(f"Tesseract results: {len(tesseract_results)}")
        print(f"Total OCR results: {len(results)}")
        print(f"All OCR results: {results}")
        
        # Enhanced number extraction with multiple patterns
        all_numbers = []
        
        for text in results:
            # Multiple number patterns
            patterns = [
                r'\d{2}[-ー]\d{2}',        # XX-XX format
                r'\d{4}',                  # XXXX format
                r'\d{2}\s+\d{2}',         # XX XX with space
                r'\d{1,2}[-ー]?\d{1,2}',  # Flexible format
                r'\b\d+\b',               # Any digits surrounded by word boundaries
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # Clean and validate
                    cleaned = re.sub(r'[^\d-]', '', match)
                    if len(cleaned) >= 2:  # At least 2 digits
                        all_numbers.append(cleaned)
                        print(f"Found number '{cleaned}' from text '{text}' using pattern '{pattern}'")
        
        print(f"All extracted numbers: {all_numbers}")
        
        # Enhanced validation and formatting
        if all_numbers:
            # Process and format numbers
            formatted_numbers = []
            
            for num in all_numbers:
                digits_only = re.sub(r'[^\d]', '', num)
                
                if len(digits_only) == 1:
                    continue  # Skip single digits
                elif len(digits_only) == 2:
                    formatted = f"0{digits_only[0]}-{digits_only[1]}0"  # 2 digits -> 0X-Y0
                elif len(digits_only) == 3:
                    formatted = f"0{digits_only[0]}-{digits_only[1:]}"  # 3 digits -> 0X-YZ
                elif len(digits_only) == 4:
                    formatted = f"{digits_only[:2]}-{digits_only[2:]}"  # 4 digits -> XX-YY
                elif len(digits_only) > 4:
                    # Take first 4 digits
                    first_four = digits_only[:4]
                    formatted = f"{first_four[:2]}-{first_four[2:]}"
                else:
                    continue
                
                # Validate result - exclude invalid patterns
                invalid_patterns = ['0-00', '00-0', '00-00', '0-0']
                if re.match(r'\d{1,2}-\d{1,2}', formatted) and formatted not in invalid_patterns:
                    formatted_numbers.append(formatted)
            
            print(f"Formatted numbers: {formatted_numbers}")
            
            if formatted_numbers:
                # Use voting system
                counter = Counter(formatted_numbers)
                best_result = counter.most_common(1)[0]
                best_number = best_result[0]
                vote_count = best_result[1]
                
                confidence = "high" if vote_count >= 2 else "medium"
                
                print(f"Selected: {best_number} with {vote_count} votes")
                
                return {
                    "license_plate": best_number,
                    "full_text": best_number,
                    "confidence": confidence,
                    "debug_info": f"OCR attempts: {len(results)}, Numbers found: {len(all_numbers)}, Best: {best_number} ({vote_count} votes)",
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
        
        print("No valid numbers found")
        return {
            "license_plate": "",
            "full_text": "認識できませんでした",
            "confidence": "failed",
            "debug_info": f"OCR attempts: {len(results)}, Debug: {'; '.join(debug_info[:5])}",
            "message": "数字が検出されませんでした。明るい場所で、ナンバープレート全体がはっきり見えるように撮影してください。"
        }
            
    except Exception as e:
        print(f"OCR error: {e}")
        return {
            "license_plate": "",
            "full_text": "認識できませんでした",
            "confidence": "failed",
            "message": "手動で入力してください。"
        }

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