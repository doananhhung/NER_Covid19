# backend_api/api_models.py
"""
Pydantic models cho request/response của API endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class HealthCheckResponse(BaseModel):
    """Response cho health check endpoint"""
    status: str
    model_loaded: bool
    vncorenlp_available: bool
    gemini_configured: bool
    timestamp: str


class NERPredictRequest(BaseModel):
    """Request body cho NER prediction"""
    text: str = Field(..., description="Văn bản cần phân tích", min_length=1)


class EntityResponse(BaseModel):
    """Cấu trúc của một entity"""
    text: str
    tag: str
    start: int
    end: int


class NERPredictResponse(BaseModel):
    """Response cho NER prediction"""
    success: bool
    entities: List[EntityResponse]
    processing_time: float
    error: Optional[str] = None


class PatientRecordResponse(BaseModel):
    """Response cho thông tin bệnh nhân"""
    patient_id: Optional[str] = None
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    job: Optional[str] = None
    locations: List[str] = []
    organizations: List[str] = []
    symptoms_and_diseases: List[str] = []
    transportations: List[str] = []
    dates: Dict[str, List[str]] = {}
    confidence: float = 1.0
    warnings: List[str] = []


class ManualExtractRequest(BaseModel):
    """Request cho manual extraction"""
    text: str = Field(..., description="Văn bản về 1 bệnh nhân", min_length=1)


class ManualExtractResponse(BaseModel):
    """Response cho manual extraction"""
    success: bool
    entities: List[EntityResponse]
    patient_record: PatientRecordResponse
    processing_time: float
    error: Optional[str] = None


class AutoExtractRequest(BaseModel):
    """Request cho auto extraction với Gemini"""
    text: str = Field(..., description="Văn bản có thể chứa nhiều bệnh nhân", min_length=1)
    gemini_api_key: Optional[str] = None


class PatientSegmentResponse(BaseModel):
    """Response cho 1 segment bệnh nhân trong auto mode"""
    patient_index: int
    original_text: str
    entities: List[EntityResponse]
    patient_record: PatientRecordResponse


class AutoExtractResponse(BaseModel):
    """Response cho auto extraction"""
    success: bool
    num_patients: int
    patients: List[PatientSegmentResponse]
    processing_time: float
    error: Optional[str] = None
