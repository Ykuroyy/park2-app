import os
import datetime
import pytesseract
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
    Receives an image, performs OCR for Japanese license plates.
    Returns both the 4-digit number and full plate text.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Get image dimensions for quality check
    height, width = img.shape[:2]
    
    # Auto-rotate if image is in portrait mode (common with phone cameras)
    if height > width * 1.5:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height
    
    # Resize if image is too small (improves OCR accuracy)
    if width < 800:
        scale = 800 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing for Japanese license plates
    # 1. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 2. Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 3. Sharpen the image
    kernel_sharpen = np.array([[-1,-1,-1],
                                [-1, 9,-1],
                                [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    # 4. Binary threshold
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        # Multiple OCR attempts with different configurations
        results = []
        
        # Attempt 1: Focus on 4-digit number (main identifier)
        config_numbers = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
        text_numbers = pytesseract.image_to_string(thresh, config=config_numbers)
        numbers_only = "".join(filter(str.isdigit, text_numbers))
        
        # Extract 4-digit sequence
        import re
        four_digit_match = re.search(r'\d{4}', numbers_only)
        four_digit = four_digit_match.group() if four_digit_match else ""
        
        # Attempt 2: Try with inverted image if needed
        if len(four_digit) < 4:
            thresh_inv = cv2.bitwise_not(thresh)
            text_inv = pytesseract.image_to_string(thresh_inv, config=config_numbers)
            numbers_inv = "".join(filter(str.isdigit, text_inv))
            four_digit_match_inv = re.search(r'\d{4}', numbers_inv)
            if four_digit_match_inv:
                four_digit = four_digit_match_inv.group()
        
        # Attempt 3: Try Japanese + English for full plate (optional)
        # Note: This requires tesseract-ocr-jpn package
        full_text = ""
        try:
            config_full = r'--oem 3 --psm 6'
            # Try Japanese + English
            full_text = pytesseract.image_to_string(thresh, config=config_full, lang='jpn+eng')
            full_text = full_text.strip().replace('\n', ' ')
        except:
            # If Japanese not available, use English only
            full_text = pytesseract.image_to_string(thresh, config=config_full, lang='eng')
            full_text = full_text.strip().replace('\n', ' ')
        
        # Log for debugging
        print(f"OCR Debug - Numbers found: {numbers_only}, 4-digit: {four_digit}, Full: {full_text}")
        
        # Return result with fallback message
        if four_digit:
            return {
                "license_plate": four_digit,
                "full_text": full_text if full_text else four_digit,
                "confidence": "high" if len(four_digit) == 4 else "low"
            }
        elif numbers_only:
            return {
                "license_plate": numbers_only[:4],  # Take first 4 digits
                "full_text": full_text if full_text else numbers_only,
                "confidence": "low"
            }
        else:
            return {
                "license_plate": "",
                "full_text": full_text if full_text else "認識できませんでした",
                "confidence": "failed",
                "message": "画像をもう一度撮影するか、手動で入力してください"
            }

    except pytesseract.TesseractNotFoundError:
        raise HTTPException(status_code=500, detail="Tesseract is not installed or not in your PATH.")
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return {
            "license_plate": "",
            "full_text": "エラーが発生しました",
            "confidence": "error",
            "message": f"エラー: {str(e)}"
        }

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
