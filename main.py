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
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple OCR
        try:
            text = pytesseract.image_to_string(gray, config='--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789-')
            
            # Extract numbers
            numbers = re.findall(r'\d{4}|\d{2}-\d{2}', text)
            
            if numbers:
                best_number = numbers[0]
                # Format as XX-XX
                if len(best_number) == 4:
                    best_number = f"{best_number[:2]}-{best_number[2:]}"
                    
                return {
                    "license_plate": best_number,
                    "full_text": best_number,
                    "confidence": "medium",
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