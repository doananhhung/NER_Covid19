// chrome_extension/shared/constants.js
/**
 * Cac hang so dung chung cho Extension
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

const API_ENDPOINTS = {
    HEALTH: '/api/health',
    PREDICT: '/api/ner/predict',
    EXTRACT_MANUAL: '/api/ner/extract-manual',
    EXTRACT_AUTO: '/api/ner/extract-auto'
};

// Entity types va mau sac tuong ung
const ENTITY_COLORS = {
    'PATIENT_ID': '#f9c5c7',
    'NAME': '#ffcccb',
    'AGE': '#fdfd96',
    'GENDER': '#87ceeb',
    'JOB': '#f5deb3',
    'LOCATION': '#ace4aa',
    'ORGANIZATION': '#c7b5e4',
    'SYMPTOM_AND_DISEASE': '#b2d8d8',
    'TRANSPORTATION': '#d3d3d3',
    'DATE': '#ffb347',
    'DEFAULT': '#f0f2f6'
};

// Entity type labels (tieng Viet)
const ENTITY_LABELS = {
    'PATIENT_ID': 'Ma benh nhan',
    'NAME': 'Ho ten',
    'AGE': 'Tuoi',
    'GENDER': 'Gioi tinh',
    'JOB': 'Nghe nghiep',
    'LOCATION': 'Dia diem',
    'ORGANIZATION': 'To chuc',
    'SYMPTOM_AND_DISEASE': 'Trieu chung/Benh',
    'TRANSPORTATION': 'Phuong tien',
    'DATE': 'Ngay thang'
};

// Message types cho Chrome messaging
const MESSAGE_TYPES = {
    GET_PAGE_TEXT: 'GET_PAGE_TEXT',
    HIGHLIGHT_ENTITIES: 'HIGHLIGHT_ENTITIES',
    CLEAR_HIGHLIGHTS: 'CLEAR_HIGHLIGHTS'
};

// Processing modes
const PROCESSING_MODES = {
    MANUAL: 'manual',
    AUTO: 'auto'
};

// Input sources
const INPUT_SOURCES = {
    WEBPAGE: 'webpage',
    MANUAL: 'manual'
};

// Gioi han
const LIMITS = {
    MAX_TEXT_LENGTH: 500000,      // 500K ký tự cho Manual mode
    MAX_TEXT_LENGTH_AUTO: 1000000 // 1M ký tự cho Auto mode
};

// Export to window for content scripts
if (typeof window !== 'undefined') {
    window.NER_CONSTANTS = {
        API_BASE_URL,
        API_ENDPOINTS,
        ENTITY_COLORS,
        ENTITY_LABELS,
        MESSAGE_TYPES,
        PROCESSING_MODES,
        INPUT_SOURCES,
        LIMITS
    };
}
