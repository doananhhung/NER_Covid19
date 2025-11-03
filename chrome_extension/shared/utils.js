// chrome_extension/shared/utils.js
/**
 * Cac ham tien ich dung chung
 */

/**
 * Format timestamp thanh string
 */
function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('vi-VN');
}

/**
 * Truncate text neu qua dai
 */
function truncateText(text, maxLength = 100) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

/**
 * Escape HTML de tranh XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Group entities theo tag type
 */
function groupEntitiesByTag(entities) {
    const grouped = {};
    entities.forEach(entity => {
        if (!grouped[entity.tag]) {
            grouped[entity.tag] = [];
        }
        grouped[entity.tag].push(entity);
    });
    return grouped;
}

/**
 * Tao CSV tu patient records
 */
function generateCSV(patients) {
    if (!patients || patients.length === 0) {
        return '';
    }

    // CSV Headers
    const headers = [
        'Ma BN',
        'Ho ten',
        'Tuoi',
        'Gioi tinh',
        'Nghe nghiep',
        'Dia diem',
        'To chuc',
        'Trieu chung/Benh',
        'Phuong tien',
        'Ngay thang',
        'Do tin cay'
    ];

    let csv = headers.join(',') + '\n';

    // Data rows
    patients.forEach(patient => {
        const record = patient.patient_record;
        const row = [
            escapeCSVField(record.patient_id || ''),
            escapeCSVField(record.name || ''),
            escapeCSVField(record.age || ''),
            escapeCSVField(record.gender || ''),
            escapeCSVField(record.job || ''),
            escapeCSVField(record.locations.join('; ')),
            escapeCSVField(record.organizations.join('; ')),
            escapeCSVField(record.symptoms_and_diseases.join('; ')),
            escapeCSVField(record.transportations.join('; ')),
            escapeCSVField(formatDates(record.dates)),
            record.confidence.toFixed(2)
        ];
        csv += row.join(',') + '\n';
    });

    return csv;
}

/**
 * Escape CSV field
 */
function escapeCSVField(field) {
    if (field === null || field === undefined) {
        return '""';
    }

    const stringField = String(field);

    if (stringField.includes(',') || stringField.includes('"') || stringField.includes('\n')) {
        return '"' + stringField.replace(/"/g, '""') + '"';
    }

    return '"' + stringField + '"';
}

/**
 * Format dates object thanh string
 */
function formatDates(dates) {
    if (!dates) return '';

    const parts = [];
    for (const [key, values] of Object.entries(dates)) {
        if (values && values.length > 0) {
            parts.push(`${key}: ${values.join(', ')}`);
        }
    }
    return parts.join('; ');
}

/**
 * Download file
 */
function downloadFile(content, filename, contentType = 'text/plain') {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        console.error('Failed to copy:', err);
        return false;
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    // Type: 'success', 'error', 'warning', 'info'
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

/**
 * Get entity color by tag
 */
function getEntityColor(tag) {
    const ENTITY_COLORS = window.NER_CONSTANTS?.ENTITY_COLORS || {};
    return ENTITY_COLORS[tag] || ENTITY_COLORS['DEFAULT'] || '#f0f2f6';
}

// Export to window for content scripts
if (typeof window !== 'undefined') {
    window.NER_UTILS = {
        formatTimestamp,
        truncateText,
        escapeHtml,
        groupEntitiesByTag,
        generateCSV,
        escapeCSVField,
        formatDates,
        downloadFile,
        copyToClipboard,
        showNotification,
        getEntityColor
    };
}
