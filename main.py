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
        
        # Try only 3 most effective OCR methods (not 48!)
        results = []
        
        # Method 1: Numbers only
        try:
            text1 = pytesseract.image_to_string(enhanced, config='--psm 8 -c tessedit_char_whitelist=0123456789-')
            if text1.strip():
                results.append(fix_common_mistakes(text1.strip()))
        except: pass
        
        # Method 2: General text
        try:
            text2 = pytesseract.image_to_string(enhanced, config='--psm 7')
            if text2.strip():
                results.append(fix_common_mistakes(text2.strip()))
        except: pass
        
        # Method 3: Inverted image
        try:
            inverted = cv2.bitwise_not(enhanced)
            text3 = pytesseract.image_to_string(inverted, config='--psm 8 -c tessedit_char_whitelist=0123456789-')
            if text3.strip():
                results.append(fix_common_mistakes(text3.strip()))
        except: pass
        
        # Find numbers in results
        all_numbers = []
        for text in results:
            # Find 4-digit patterns
            numbers = re.findall(r'\d{3,4}', text)
            all_numbers.extend(numbers)
        
        # Simple validation and formatting
        if all_numbers:
            # Take the most common or first valid result
            counter = Counter(all_numbers)
            best_num = counter.most_common(1)[0][0]
            
            # Format as XX-XX
            if len(best_num) == 4:
                formatted = f"{best_num[:2]}-{best_num[2:]}"
            elif len(best_num) == 3:
                formatted = f"0{best_num[0]}-{best_num[1:]}"
            else:
                formatted = best_num
            
            return {
                "license_plate": formatted,
                "full_text": formatted,
                "confidence": "medium",
                "area": "",
                "classification": "",
                "hiragana": "",
                "number": formatted,
                "plate_parts": {
                    "area": "",
                    "classification": "",
                    "hiragana": "",
                    "number": formatted
                }
            }
        else:
            return {
                "license_plate": "",
                "full_text": "認識できませんでした",
                "confidence": "failed",
                "message": "手動で入力してください。100台規模の駐車場では手動入力も効率的です。"
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