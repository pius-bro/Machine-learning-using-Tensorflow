import json
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from ultralytics import YOLO
import io
from PIL import Image

import models
from database import engine, get_db

# Initialize database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="HCI Object Detection API")

# Enable CORS for seamless Frontend-Backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight YOLOv8 model (auto-downloads on first run)
model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Run Inference
    results = model(image)
    result = results[0]
    
    # Parse predictions into an HCI-friendly JSON format
    predictions = []
    boxes = result.boxes
    for box in boxes:
        coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
        confidence = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        
        predictions.append({
            "bbox": [int(c) for c in coords],
            "label": label,
            "confidence": round(confidence, 2)
        })
    
    # Log to Database for user history/auditability
    log_entry = models.DetectionLog(
        filename=file.filename,
        predictions=json.dumps(predictions)
    )
    db.add(log_entry)
    db.commit()
    
    return {"filename": file.filename, "predictions": predictions}

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    logs = db.query(models.DetectionLog).order_index(models.DetectionLog.id.desc()).limit(10).all()
    return [
        {
            "id": log.id,
            "filename": log.filename,
            "predictions": json.loads(log.predictions),
            "timestamp": log.timestamp
        } for log in logs
    ]