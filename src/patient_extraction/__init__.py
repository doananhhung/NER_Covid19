# src/patient_extraction/__init__.py
#
# Patient Information Extraction Package
#
# Các modules:
# - entity_structures: Core data structures (Entity, PatientRecord)
# - csv_exporter: MODULE - Export sang CSV
# - manual_extractor: MODULE - Trích xuất thủ công (1 bệnh nhân)

from .entity_structures import Entity, PatientRecord
from .manual_extractor import ManualPatientExtractor, extract_single_patient

__all__ = [
    'Entity',
    'PatientRecord',
    'ManualPatientExtractor',
    'extract_single_patient',
]
