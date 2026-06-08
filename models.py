from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from database import Base

class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    predictions = Column(String)  # Stored as a JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)