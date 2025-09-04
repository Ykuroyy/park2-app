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
from ocr_fast import fast_high_precision_ocr

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

# EasyOCR will be initialized lazily when first needed to avoid startup delays
easyocr_reader = None

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
