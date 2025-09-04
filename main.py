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

Base = declarative_base()
engine = create_engine(DATABASE_URL)
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

# Create the database tables
Base.metadata.create_all(bind=engine)

# --- Pydantic Models ---
class CheckInRequest(BaseModel):
    license_plate: str

class ParkingLogResponse(BaseModel):
    id: int
    license_plate: str
    check_in: datetime.datetime

    class Config:
        from_attributes = True

# --- FastAPI App Initialization ---
app = FastAPI(title="Park2 App")

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
    Receives an image, performs OCR, and returns the recognized text.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding for better results on varied lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    try:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config, lang='eng') # Assuming English characters
        cleaned_text = "".join(filter(str.isalnum, text)).upper()
        if not cleaned_text:
             # If cleaning removes everything, try without thresholding
            text = pytesseract.image_to_string(gray, config=custom_config, lang='eng')
            cleaned_text = "".join(filter(str.isalnum, text)).upper()

    except pytesseract.TesseractNotFoundError:
        raise HTTPException(status_code=500, detail="Tesseract is not installed or not in your PATH.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during OCR: {str(e)}")

    return {"license_plate": cleaned_text}

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
