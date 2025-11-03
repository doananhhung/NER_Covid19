# backend_api/main.py
"""
FastAPI Backend Server cho Chrome Extension
Cung cấp REST API endpoints để xử lý NER và trích xuất thông tin bệnh nhân
"""

import sys
import os
from datetime import datetime
from typing import Optional
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import logging utilities
from backend_api.logger import (
    setup_logger,
    log_separator,
    log_text_preview,
    log_entities,
    log_patient_record
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment variables from: {env_path}")
    else:
        print(f"No .env file found at: {env_path}")
        print("Will use system environment variables")
except ImportError:
    print("python-dotenv not installed. Using system environment variables only.")
    print("To use .env file: pip install python-dotenv")

# Import models từ src
from src.inference import NERPredictor
from src import config as ner_config
from src.patient_extraction.manual_extractor import extract_single_patient
from src.patient_extraction.gemini_splitter import split_text_with_gemini

# Import API models
from backend_api.api_models import (
    HealthCheckResponse,
    NERPredictRequest,
    NERPredictResponse,
    EntityResponse,
    ManualExtractRequest,
    ManualExtractResponse,
    PatientRecordResponse,
    AutoExtractRequest,
    AutoExtractResponse,
    PatientSegmentResponse
)


# Khởi tạo FastAPI app
app = FastAPI(
    title="Vietnamese COVID-19 NER API",
    description="API Backend cho Chrome Extension - NER và trích xuất thông tin bệnh nhân",
    version="1.0.0"
)

# Cấu hình CORS - cho phép Extension gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables để cache model
ner_predictor: Optional[NERPredictor] = None
gemini_api_key_env: Optional[str] = None

# Setup logger
api_logger = setup_logger("ner_api", "logs/ner_api.log")


def load_model():
    """Load NER model vào memory"""
    global ner_predictor
    
    if ner_predictor is not None:
        return ner_predictor
    
    try:
        print("Đang tải mô hình NER...")
        predictor = NERPredictor(
            model_path=ner_config.MODEL_OUTPUT_DIR,
            use_word_segmentation=True
        )
        
        if predictor.model is None:
            raise Exception("Model không thể load được")
        
        print("Mô hình đã được tải thành công!")
        ner_predictor = predictor
        return predictor
    
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        raise


def load_gemini_key():
    """Load Gemini API key từ environment variable"""
    global gemini_api_key_env
    gemini_api_key_env = os.getenv("GEMINI_API_KEY")
    return gemini_api_key_env


# Load model khi khởi động server
@app.on_event("startup")
async def startup_event():
    """Khởi tạo khi server start"""
    print("=" * 80)
    print("KHỞI ĐỘNG BACKEND SERVER")
    print("=" * 80)
    
    try:
        load_model()
        load_gemini_key()
        print("\nServer sẵn sàng!")
    except Exception as e:
        print(f"\nCảnh báo: Server khởi động nhưng có lỗi: {e}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - thông tin cơ bản"""
    return {
        "message": "Vietnamese COVID-19 NER API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/ner/predict",
            "extract_manual": "/api/ner/extract-manual",
            "extract_auto": "/api/ner/extract-auto"
        }
    }


@app.get("/api/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Kiểm tra trạng thái server và các thành phần
    """
    model_loaded = ner_predictor is not None and ner_predictor.model is not None
    vncorenlp_ok = False
    
    if ner_predictor and ner_predictor.text_processor:
        vncorenlp_ok = ner_predictor.text_processor.is_available()
    
    gemini_ok = gemini_api_key_env is not None
    
    return HealthCheckResponse(
        status="online",
        model_loaded=model_loaded,
        vncorenlp_available=vncorenlp_ok,
        gemini_configured=gemini_ok,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/ner/predict", response_model=NERPredictResponse, tags=["NER"])
async def predict_ner(request: NERPredictRequest):
    """
    Endpoint cơ bản: Chỉ thực hiện NER, trả về entities
    
    Args:
        request: Chứa text cần phân tích
        
    Returns:
        NERPredictResponse với danh sách entities
    """
    if ner_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model chưa được load. Vui lòng khởi động lại server."
        )
    
    try:
        start_time = time.time()
        
        # Validate input
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Văn bản rỗng")
        
        if len(text) > 500000:
            raise HTTPException(
                status_code=400,
                detail="Văn bản quá dài (> 500,000 ký tự). Vui lòng rút ngắn."
            )
        
        # Chạy NER
        entities_raw = ner_predictor.predict(text, show_debug=False)
        
        # Convert sang EntityResponse format
        entities = [
            EntityResponse(
                text=e['text'],
                tag=e['tag'],
                start=e['start'],
                end=e['end']
            )
            for e in entities_raw
        ]
        
        processing_time = time.time() - start_time
        
        return NERPredictResponse(
            success=True,
            entities=entities,
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        return NERPredictResponse(
            success=False,
            entities=[],
            processing_time=0,
            error=str(e)
        )


@app.post("/api/ner/extract-manual", response_model=ManualExtractResponse, tags=["Extraction"])
async def extract_manual(request: ManualExtractRequest):
    """
    Endpoint Manual Mode: NER + Trích xuất thông tin 1 bệnh nhân
    
    Args:
        request: Chứa text về 1 bệnh nhân
        
    Returns:
        ManualExtractResponse với entities và patient_record
    """
    if ner_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model chưa được load. Vui lòng khởi động lại server."
        )
    
    try:
        start_time = time.time()
        
        # Log request
        log_separator(api_logger, "MANUAL EXTRACT REQUEST")
        
        # Validate input
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Văn bản rỗng")
        
        api_logger.info(f"Input text length: {len(text)} characters")
        log_text_preview(api_logger, text, max_length=200)
        
        if len(text) > 500000:
            raise HTTPException(
                status_code=400,
                detail="Văn bản quá dài (> 500,000 ký tự)"
            )
        
        # Bước 1: Chạy NER
        api_logger.info("Running NER prediction...")
        entities_raw = ner_predictor.predict(text, show_debug=False)
        log_entities(api_logger, entities_raw, max_entities=20)
        
        # Bước 2: Trích xuất patient record
        api_logger.info("Extracting patient information...")
        patient_record = extract_single_patient(entities_raw, text)
        log_patient_record(api_logger, patient_record)
        
        # Convert entities sang response format
        entities = [
            EntityResponse(
                text=e['text'],
                tag=e['tag'],
                start=e['start'],
                end=e['end']
            )
            for e in entities_raw
        ]
        
        # Convert patient record sang response format - LẤY TRỰC TIẾP TỪ OBJECT
        patient_response = PatientRecordResponse(
            patient_id=patient_record.patient_id,
            name=patient_record.name,
            age=patient_record.age,
            gender=patient_record.gender,
            job=patient_record.job,
            locations=patient_record.locations,  # List[str]
            organizations=patient_record.organizations,  # List[str]
            symptoms_and_diseases=patient_record.symptoms_and_diseases,  # List[str]
            transportations=patient_record.transportations,  # List[str]
            dates=patient_record.dates,  # Dict[str, List[str]]
            confidence=patient_record.confidence,
            warnings=patient_record.warnings  # List[str]
        )
        
        processing_time = time.time() - start_time
        
        api_logger.info(f"Manual extraction completed in {processing_time:.2f} seconds")
        log_separator(api_logger)
        
        return ManualExtractResponse(
            success=True,
            entities=entities,
            patient_record=patient_response,
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error in manual extraction: {str(e)}", exc_info=True)
        return ManualExtractResponse(
            success=False,
            entities=[],
            patient_record=PatientRecordResponse(),
            processing_time=0,
            error=str(e)
        )


@app.post("/api/ner/extract-auto", response_model=AutoExtractResponse, tags=["Extraction"])
async def extract_auto(request: AutoExtractRequest):
    """
    Endpoint Auto Mode: Tách văn bản với Gemini + NER + Trích xuất nhiều bệnh nhân
    
    Args:
        request: Chứa text có thể có nhiều bệnh nhân và optional API key
        
    Returns:
        AutoExtractResponse với danh sách patients
    """
    if ner_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model chưa được load"
        )
    
    # Xác định API key sử dụng
    api_key = request.gemini_api_key or gemini_api_key_env
    
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Gemini API key không được cung cấp. Vui lòng set GEMINI_API_KEY environment variable hoặc truyền vào request."
        )
    
    try:
        start_time = time.time()
        
        # Log request
        log_separator(api_logger, "AUTO EXTRACT REQUEST")
        
        # Validate input
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Văn bản rỗng")
        
        api_logger.info(f"Input text length: {len(text)} characters")
        log_text_preview(api_logger, text, max_length=200)
        
        if len(text) > 1000000:
            raise HTTPException(
                status_code=400,
                detail="Văn bản quá dài (> 1,000,000 ký tự)"
            )
        
        # Bước 1: Tách văn bản với Gemini
        api_logger.info("Step 1: Calling Gemini API to split text...")
        text_segments = split_text_with_gemini(text, api_key)
        
        if not text_segments:
            api_logger.warning("Gemini returned empty segments, using fallback (original text)")
            text_segments = [text]  # Fallback
        
        api_logger.info(f"Gemini split result: {len(text_segments)} segment(s)")
        
        # Bước 2: Xử lý từng segment
        api_logger.info("Step 2: Processing each segment with NER + Extraction...")
        patients_data = []
        
        for idx, segment_text in enumerate(text_segments, start=1):
            try:
                log_separator(api_logger, f"SEGMENT {idx}/{len(text_segments)}")
                api_logger.info(f"Segment {idx} length: {len(segment_text)} characters")
                api_logger.info(f"Full segment text:")
                api_logger.info(f"{segment_text}")
                api_logger.info(f"--- End of segment text ---")
                
                # NER cho segment
                api_logger.info(f"Running NER on segment {idx}...")
                entities_raw = ner_predictor.predict(segment_text, show_debug=False)
                log_entities(api_logger, entities_raw, max_entities=15)
                
                # Trích xuất patient info
                api_logger.info(f"Extracting patient info from segment {idx}...")
                patient_record = extract_single_patient(entities_raw, segment_text)
                log_patient_record(api_logger, patient_record)
                
                # Convert entities
                entities = [
                    EntityResponse(
                        text=e['text'],
                        tag=e['tag'],
                        start=e['start'],
                        end=e['end']
                    )
                    for e in entities_raw
                ]
                
                # Convert patient record - LẤY TRỰC TIẾP TỪ OBJECT
                patient_response = PatientRecordResponse(
                    patient_id=patient_record.patient_id,
                    name=patient_record.name,
                    age=patient_record.age,
                    gender=patient_record.gender,
                    job=patient_record.job,
                    locations=patient_record.locations,  # List[str]
                    organizations=patient_record.organizations,  # List[str]
                    symptoms_and_diseases=patient_record.symptoms_and_diseases,  # List[str]
                    transportations=patient_record.transportations,  # List[str]
                    dates=patient_record.dates,  # Dict[str, List[str]]
                    confidence=patient_record.confidence,
                    warnings=patient_record.warnings  # List[str]
                )
                
                # Tạo segment response
                segment_response = PatientSegmentResponse(
                    patient_index=idx,
                    original_text=segment_text,
                    entities=entities,
                    patient_record=patient_response
                )
                
                patients_data.append(segment_response)
                api_logger.info(f"Segment {idx} processed successfully")
                
            except Exception as segment_error:
                api_logger.error(f"Error processing segment {idx}: {segment_error}", exc_info=True)
                continue
        
        processing_time = time.time() - start_time
        
        api_logger.info(f"Auto extraction completed: {len(patients_data)} patient(s) found in {processing_time:.2f} seconds")
        log_separator(api_logger)
        
        return AutoExtractResponse(
            success=True,
            num_patients=len(patients_data),
            patients=patients_data,
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error in auto extraction: {str(e)}", exc_info=True)
        return AutoExtractResponse(
            success=False,
            num_patients=0,
            patients=[],
            processing_time=0,
            error=str(e)
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "error_type": "SERVER_ERROR"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print(" VIETNAMESE COVID-19 NER - BACKEND API SERVER")
    print("=" * 80)
    print("\nĐang khởi động server...")
    print("Server sẽ chạy tại: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\n" + "=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
