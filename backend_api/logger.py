# backend_api/logger.py
"""
Module cấu hình logging cho Backend API
Ghi log chi tiết về quá trình xử lý NER và trích xuất thông tin bệnh nhân
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str = "ner_api", log_file: str = "logs/ner_api.log") -> logging.Logger:
    """
    Cấu hình logger với file handler và console handler
    
    Args:
        name: Tên logger
        log_file: Đường dẫn file log
        
    Returns:
        logging.Logger: Logger đã được cấu hình
    """
    # Tạo thư mục logs nếu chưa có
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Tránh duplicate handlers nếu gọi setup_logger nhiều lần
    if logger.handlers:
        return logger
    
    # Format log
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - ghi vào file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - in ra console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_separator(logger: logging.Logger, title: str = ""):
    """
    Ghi dòng phân cách vào log để dễ đọc
    
    Args:
        logger: Logger instance
        title: Tiêu đề section (optional)
    """
    if title:
        logger.info(f"{'=' * 20} {title} {'=' * 20}")
    else:
        logger.info("=" * 60)


def log_text_preview(logger: logging.Logger, text: str, max_length: int = 200):
    """
    Log preview của văn bản (tránh log quá dài)
    
    Args:
        logger: Logger instance
        text: Văn bản cần log
        max_length: Độ dài tối đa preview
    """
    if len(text) <= max_length:
        logger.info(f"Text preview: \"{text}\"")
    else:
        logger.info(f"Text preview (first {max_length} chars): \"{text[:max_length]}...\"")


def log_entities(logger: logging.Logger, entities: list, max_entities: int = None):
    """
    Log danh sách entities một cách có cấu trúc
    
    Args:
        logger: Logger instance
        entities: List of entity dicts
        max_entities: Số entity tối đa để log (None = log tất cả)
    """
    total = len(entities)
    logger.info(f"Total entities found: {total}")
    
    if total == 0:
        return
    
    display_count = min(total, max_entities) if max_entities else total
    
    for i, entity in enumerate(entities[:display_count], 1):
        text = entity.get('text', '')
        tag = entity.get('tag', '')
        start = entity.get('start', '')
        end = entity.get('end', '')
        logger.info(f"  Entity {i}/{total}: \"{text}\" ({tag}) [{start}-{end}]")
    
    if max_entities and total > max_entities:
        logger.info(f"  ... and {total - max_entities} more entities")


def log_patient_record(logger: logging.Logger, patient_record):
    """
    Log thông tin PatientRecord chi tiết
    
    Args:
        logger: Logger instance
        patient_record: PatientRecord object hoặc PatientRecordResponse
    """
    logger.info("Patient Record Details:")
    logger.info(f"  Patient ID: {patient_record.patient_id or 'N/A'}")
    logger.info(f"  Name: {patient_record.name or 'N/A'}")
    logger.info(f"  Age: {patient_record.age or 'N/A'}")
    logger.info(f"  Gender: {patient_record.gender or 'N/A'}")
    logger.info(f"  Job: {patient_record.job or 'N/A'}")
    
    # Locations
    locations = patient_record.locations if isinstance(patient_record.locations, list) else []
    logger.info(f"  Locations ({len(locations)}): {locations if locations else 'N/A'}")
    
    # Organizations
    orgs = patient_record.organizations if isinstance(patient_record.organizations, list) else []
    logger.info(f"  Organizations ({len(orgs)}): {orgs if orgs else 'N/A'}")
    
    # Symptoms
    symptoms = patient_record.symptoms_and_diseases if isinstance(patient_record.symptoms_and_diseases, list) else []
    logger.info(f"  Symptoms/Diseases ({len(symptoms)}): {symptoms if symptoms else 'N/A'}")
    
    # Transportations
    transports = patient_record.transportations if isinstance(patient_record.transportations, list) else []
    logger.info(f"  Transportations ({len(transports)}): {transports if transports else 'N/A'}")
    
    # Dates
    dates = patient_record.dates if isinstance(patient_record.dates, dict) else {}
    if dates:
        logger.info(f"  Dates:")
        for date_type, date_list in dates.items():
            if date_list:
                logger.info(f"    {date_type}: {date_list}")
    else:
        logger.info(f"  Dates: N/A")
    
    # Confidence & Warnings
    logger.info(f"  Confidence: {patient_record.confidence}")
    warnings = patient_record.warnings if isinstance(patient_record.warnings, list) else []
    if warnings:
        logger.info(f"  Warnings ({len(warnings)}):")
        for warning in warnings:
            logger.info(f"    - {warning}")
    else:
        logger.info(f"  Warnings: None")
