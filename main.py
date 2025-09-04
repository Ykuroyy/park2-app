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
    Enhanced OCR for Japanese license plates with multiple preprocessing techniques.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    height, width = img.shape[:2]
    
    # Auto-rotate if needed
    if height > width * 1.5:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height
    
    # Scale up for better OCR
    target_width = 1600  # Higher resolution for better accuracy
    if width < target_width:
        scale = target_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # List to store results from different preprocessing methods
    ocr_results = []
    
    # Method 1: Advanced denoising and sharpening
    def method1_preprocessing(gray_img):
        # Bilateral filter for edge-preserving smoothing
        smooth = cv2.bilateralFilter(gray_img, 15, 80, 80)
        # Unsharp masking for enhancement
        gaussian = cv2.GaussianBlur(smooth, (0, 0), 2.0)
        sharpened = cv2.addWeighted(smooth, 2.0, gaussian, -1.0, 0)
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 21, 10)
        return binary
    
    # Method 2: Morphological operations
    def method2_preprocessing(gray_img):
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_img)
        # Otsu's thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned
    
    # Method 3: Edge detection based
    def method3_preprocessing(gray_img):
        # Median blur to reduce noise
        blurred = cv2.medianBlur(gray_img, 5)
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        # Dilate edges to connect components
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        # Find contours and create mask
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray_img)
        cv2.drawContours(mask, contours, -1, 255, -1)
        # Apply mask
        result = cv2.bitwise_and(gray_img, mask)
        _, binary = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    # Method 4: Perspective correction for skewed images
    def method4_preprocessing(gray_img):
        # Simple deskewing
        coords = np.column_stack(np.where(gray_img > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:
                (h, w) = gray_img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray_img, M, (w, h), 
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            else:
                rotated = gray_img
        else:
            rotated = gray_img
        
        # Apply threshold
        _, binary = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    import re
    
    # Try multiple preprocessing methods
    preprocessing_methods = [
        ("method1", method1_preprocessing),
        ("method2", method2_preprocessing),
        ("method3", method3_preprocessing),
        ("method4", method4_preprocessing)
    ]
    
    best_result = ""
    best_confidence = 0
    
    for method_name, preprocess_func in preprocessing_methods:
        try:
            # Apply preprocessing
            processed = preprocess_func(gray)
            
            # Try normal and inverted
            for invert in [False, True]:
                if invert:
                    test_img = cv2.bitwise_not(processed)
                else:
                    test_img = processed
                
                # Multiple PSM modes for different text layouts
                psm_modes = [6, 7, 8, 11, 13]
                
                for psm in psm_modes:
                    try:
                        # OCR with number-only whitelist
                        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
                        text = pytesseract.image_to_string(test_img, config=config)
                        numbers = "".join(filter(str.isdigit, text))
                        
                        # Look for 4-digit pattern
                        four_digit_matches = re.findall(r'\d{4}', numbers)
                        
                        if four_digit_matches:
                            for match in four_digit_matches:
                                # Validate that it looks like a license plate number
                                if match != "0000" and match != "1111":  # Filter out obvious errors
                                    print(f"Found: {match} using {method_name}, PSM={psm}, inverted={invert}")
                                    ocr_results.append(match)
                                    
                                    # If we find a consistent result multiple times, it's likely correct
                                    if ocr_results.count(match) > best_confidence:
                                        best_result = match
                                        best_confidence = ocr_results.count(match)
                    except:
                        continue
        except Exception as e:
            print(f"Error in {method_name}: {str(e)}")
            continue
    
    # Also try with less strict configuration for partial matches
    try:
        config_partial = '--oem 3 --psm 6'
        text_full = pytesseract.image_to_string(gray, config=config_partial)
        # Extract any 4-digit sequences
        partial_matches = re.findall(r'\d{4}', text_full)
        ocr_results.extend(partial_matches)
    except:
        pass
    
    # Find the most common result (voting system)
    from collections import Counter
    if ocr_results:
        result_counts = Counter(ocr_results)
        most_common = result_counts.most_common(1)[0]
        best_result = most_common[0]
        confidence_score = most_common[1]
        
        print(f"OCR Results: {result_counts}")
        print(f"Best result: {best_result} (appeared {confidence_score} times)")
        
        # Determine confidence level
        if confidence_score >= 3:
            confidence = "high"
        elif confidence_score >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "license_plate": best_result,
            "full_text": best_result,
            "confidence": confidence,
            "debug_info": f"Found {len(ocr_results)} matches, best: {best_result} ({confidence_score} votes)"
        }
    else:
        # No results found
        print("No license plate numbers detected")
        return {
            "license_plate": "",
            "full_text": "認識できませんでした",
            "confidence": "failed",
            "message": "画像をもう一度撮影するか、手動で入力してください。撮影時は明るい場所で、ナンバープレート全体が画面に収まるようにしてください。"
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
