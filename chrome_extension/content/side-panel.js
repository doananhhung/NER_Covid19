// side-panel.js
// Quản lý side panel chính (UI, logic, API calls)

(function () {
    'use strict';

    // Import constants từ shared (sẽ được load trước)
    const API_BASE_URL = window.NER_CONSTANTS?.API_BASE_URL || 'http://localhost:8000';
    const ENTITY_LABELS = window.NER_CONSTANTS?.ENTITY_LABELS || {};
    const getEntityColor = window.NER_UTILS?.getEntityColor || (() => '#999');

    // State
    let panelState = {
        isOpen: false,
        isProcessing: false,
        lastResults: null,
        serverOnline: false,
        deletedPatientIndices: []  // Track deleted patients for undo
    };

    // DOM Elements (sẽ được khởi tạo sau khi inject)
    let elements = {};

    // ============================================================================
    // INITIALIZATION
    // ============================================================================

    async function init() {
        // Kiểm tra xem panel đã tồn tại chưa
        if (document.getElementById('ner-side-panel')) {
            console.log('[NER Extension] Panel already exists');
            return;
        }

        // Inject CSS
        injectCSS();

        // Inject HTML và đợi hoàn thành
        await injectHTML();

        // Đợi một chút để đảm bảo DOM đã render
        await new Promise(resolve => setTimeout(resolve, 100));

        // Cache DOM elements
        cacheElements();

        // Kiểm tra xem elements có được load không
        if (!elements.panel) {
            console.error('[NER Extension] Failed to load panel elements');
            return;
        }

        // Setup event listeners
        setupEventListeners();

        // Apply initial state (webpage mode → disable manual processing)
        applyInitialState();

        // Kiểm tra server health
        checkServerHealth();

        console.log('[NER Extension] Side panel initialized successfully');
    }

    function injectCSS() {
        const linkId = 'ner-side-panel-css';
        if (!document.getElementById(linkId)) {
            const link = document.createElement('link');
            link.id = linkId;
            link.rel = 'stylesheet';
            link.href = chrome.runtime.getURL('content/side-panel.css');
            document.head.appendChild(link);
        }
    }

    async function injectHTML() {
        try {
            const response = await fetch(chrome.runtime.getURL('content/side-panel.html'));
            const html = await response.text();

            // Parse HTML
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const panelElement = doc.getElementById('ner-side-panel');

            if (panelElement) {
                // Thêm class hidden ban đầu
                panelElement.classList.add('hidden');
                document.body.appendChild(panelElement);
                console.log('[NER Extension] Panel HTML injected');
            } else {
                console.error('[NER Extension] Panel element not found in HTML');
            }
        } catch (error) {
            console.error('[NER Extension] Failed to inject HTML:', error);
        }
    }

    function cacheElements() {
        elements = {
            panel: document.getElementById('ner-side-panel'),
            closeBtn: document.getElementById('close-panel-btn'),
            inputSourceRadios: document.querySelectorAll('input[name="input-source"]'),
            processingModeRadios: document.querySelectorAll('input[name="processing-mode"]'),
            manualInputArea: document.getElementById('manual-input-area'),
            manualTextInput: document.getElementById('manual-text-input'),
            autoModeLabel: document.getElementById('auto-mode-label'),
            analyzeBtn: document.getElementById('analyze-btn'),
            serverStatus: document.getElementById('server-status'),
            resultsArea: document.getElementById('results-area'),
            tabBtns: document.querySelectorAll('.tab-btn'),
            entitiesTab: document.getElementById('entities-tab'),
            patientsTab: document.getElementById('patients-tab'),
            entitiesList: document.getElementById('entities-list'),
            patientsList: document.getElementById('patients-list'),
            highlightEntitiesBtn: document.getElementById('highlight-entities-btn'),
            copyEntitiesBtn: document.getElementById('copy-entities-csv-btn'),
            copyPatientsBtn: document.getElementById('copy-patients-csv-btn'),
            downloadPatientsBtn: document.getElementById('download-patients-csv-btn'),
            undoDeleteBtn: document.getElementById('undo-delete-btn'),
            processingInfo: document.getElementById('processing-info'),
            loadingOverlay: document.getElementById('loading-overlay')
        };
    }

    // ============================================================================
    // EVENT LISTENERS
    // ============================================================================

    function setupEventListeners() {
        if (!elements.closeBtn) {
            console.error('[NER Extension] Elements not found, cannot setup listeners');
            return;
        }

        // Close button
        elements.closeBtn.addEventListener('click', closePanel);

        // Input source change
        elements.inputSourceRadios.forEach(radio => {
            radio.addEventListener('change', onInputSourceChange);
        });

        // Processing mode change
        elements.processingModeRadios.forEach(radio => {
            radio.addEventListener('change', onProcessingModeChange);
        });

        // Analyze button
        elements.analyzeBtn.addEventListener('click', onAnalyze);

        // Tab buttons
        elements.tabBtns.forEach(btn => {
            btn.addEventListener('click', () => switchTab(btn.dataset.tab));
        });

        // Action buttons
        elements.highlightEntitiesBtn.addEventListener('click', () => highlightEntitiesOnPage());
        elements.copyEntitiesBtn.addEventListener('click', () => copyEntitiesCSV());
        elements.copyPatientsBtn.addEventListener('click', () => copyPatientsCSV());
        elements.downloadPatientsBtn.addEventListener('click', () => downloadPatientsCSV());
        elements.undoDeleteBtn.addEventListener('click', () => undoDeletePatient());

        // Listen for toggle messages from floating button
        window.addEventListener('message', (event) => {
            if (event.data.type === 'NER_TOGGLE_PANEL' &&
                event.data.source === 'ner-floating-button') {
                togglePanel();
            }
        });

        // Prevent panel from closing when clicking inside
        elements.panel.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        console.log('[NER Extension] Event listeners setup complete');
    }

    // ============================================================================
    // PANEL MANAGEMENT
    // ============================================================================

    function togglePanel() {
        console.log('[NER Extension] Toggle panel called, isOpen:', panelState.isOpen);
        if (!elements.panel) {
            console.error('[NER Extension] Panel element not found');
            return;
        }

        if (panelState.isOpen) {
            closePanel();
        } else {
            openPanel();
        }
    }

    function openPanel() {
        console.log('[NER Extension] Opening panel');
        if (!elements.panel) return;

        elements.panel.classList.remove('hidden');
        panelState.isOpen = true;
        notifyPanelStateChange();
    }

    function closePanel() {
        console.log('[NER Extension] Closing panel');
        if (!elements.panel) return;

        elements.panel.classList.add('hidden');
        panelState.isOpen = false;
        notifyPanelStateChange();
        // Không xóa nội dung - giữ nguyên lastResults
    }

    function notifyPanelStateChange() {
        window.postMessage({
            type: 'NER_PANEL_STATE_CHANGED',
            isOpen: panelState.isOpen
        }, '*');
    }

    // ============================================================================
    // UI HANDLERS
    // ============================================================================

    function applyInitialState() {
        // Mặc định chọn "Trang web hiện tại" → Force Auto mode và disable Manual mode
        const webpageRadio = document.querySelector('input[name="input-source"][value="webpage"]');
        if (webpageRadio && webpageRadio.checked) {
            // Force Auto mode
            const autoRadio = document.querySelector('input[name="processing-mode"][value="auto"]');
            if (autoRadio) {
                autoRadio.checked = true;
            }
            // Disable Manual mode
            disableProcessingMode('manual');
            console.log('[NER Extension] Initial state: Webpage mode → Manual processing disabled');
        }
    }

    function onInputSourceChange(e) {
        const source = e.target.value;

        if (source === 'manual') {
            elements.manualInputArea.style.display = 'block';
            // Enable cả 2 mode
            enableProcessingMode('manual');
            enableProcessingMode('auto');
            console.log('[NER Extension] Switched to Manual input → Both modes enabled');
        } else {
            elements.manualInputArea.style.display = 'none';
            // Webpage mode: force Auto mode
            const autoRadio = document.querySelector('input[name="processing-mode"][value="auto"]');
            if (autoRadio) {
                autoRadio.checked = true;
            }
            // Disable Manual mode
            disableProcessingMode('manual');
            console.log('[NER Extension] Switched to Webpage → Manual processing disabled');
        }
    }

    function onProcessingModeChange(e) {
        // Có thể thêm logic nếu cần
    }

    function enableProcessingMode(mode) {
        const radio = document.querySelector(`input[name="processing-mode"][value="${mode}"]`);
        const label = radio?.closest('.radio-label');
        if (radio && label) {
            radio.disabled = false;
            label.classList.remove('disabled');
        }
    }

    function disableProcessingMode(mode) {
        const radio = document.querySelector(`input[name="processing-mode"][value="${mode}"]`);
        const label = radio?.closest('.radio-label');
        if (radio && label) {
            radio.disabled = true;
            label.classList.add('disabled');
        }
    }

    function switchTab(tabName) {
        // Update buttons
        elements.tabBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        // Update panes
        const panes = {
            'entities': elements.entitiesTab,
            'patients': elements.patientsTab
        };

        Object.entries(panes).forEach(([name, pane]) => {
            pane.classList.toggle('active', name === tabName);
        });
    }

    // ============================================================================
    // SERVER HEALTH CHECK
    // ============================================================================

    async function checkServerHealth() {
        console.log('[NER Extension] Checking server health at:', API_BASE_URL);
        try {
            const response = await fetch(`${API_BASE_URL}/api/health`);
            console.log('[NER Extension] Health check response:', response.status);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            console.log('[NER Extension] Health data:', data);

            if (data.status === 'online' && data.model_loaded) {
                updateServerStatus(true, 'Server sẵn sàng');
                panelState.serverOnline = true;
                if (elements.analyzeBtn) {
                    elements.analyzeBtn.disabled = false;
                }
            } else {
                updateServerStatus(false, 'Server chưa sẵn sàng');
                panelState.serverOnline = false;
                if (elements.analyzeBtn) {
                    elements.analyzeBtn.disabled = true;
                }
            }
        } catch (error) {
            console.error('[NER Extension] Health check failed:', error);
            updateServerStatus(false, 'Không thể kết nối server');
            panelState.serverOnline = false;
            if (elements.analyzeBtn) {
                elements.analyzeBtn.disabled = true;
            }
        }
    }

    function updateServerStatus(online, message) {
        if (!elements.serverStatus) {
            console.error('[NER Extension] serverStatus element not found');
            return;
        }
        elements.serverStatus.className = online ? 'status-indicator online' : 'status-indicator offline';
        const statusText = elements.serverStatus.querySelector('.status-text');
        if (statusText) {
            statusText.textContent = message;
        }
    }

    // ============================================================================
    // ANALYZE LOGIC
    // ============================================================================

    async function onAnalyze() {
        if (panelState.isProcessing) return;

        // Get input
        const inputSource = document.querySelector('input[name="input-source"]:checked').value;
        const processingMode = document.querySelector('input[name="processing-mode"]:checked').value;

        let text = '';
        if (inputSource === 'manual') {
            text = elements.manualTextInput.value.trim();
            if (!text) {
                alert('Vui lòng nhập văn bản');
                return;
            }
        } else {
            // Get text from webpage
            text = extractTextFromPage();
            if (!text) {
                alert('Không thể lấy văn bản từ trang web. Vui lòng thử chế độ nhập thủ công.');
                return;
            }
        }

        // Validate text length (sử dụng limit từ constants)
        const MAX_LENGTH = window.NER_CONSTANTS?.LIMITS?.MAX_TEXT_LENGTH || 500000;
        if (text.length > MAX_LENGTH) {
            alert(`Văn bản quá dài (${text.length.toLocaleString()} ký tự). Giới hạn: ${MAX_LENGTH.toLocaleString()} ký tự.`);
            return;
        }

        // Show loading
        showLoading(true);
        panelState.isProcessing = true;

        // Reset deleted patients list for new analysis
        panelState.deletedPatientIndices = [];

        try {
            let results;
            if (processingMode === 'manual') {
                results = await processManualMode(text);
            } else {
                results = await processAutoMode(text);
            }

            // Save results
            panelState.lastResults = results;

            // Display results
            displayResults(results, processingMode);

            // Show results area
            elements.resultsArea.style.display = 'block';

        } catch (error) {
            console.error('Analysis error:', error);
            alert('Lỗi khi phân tích: ' + error.message);
        } finally {
            showLoading(false);
            panelState.isProcessing = false;
        }
    }

    function extractTextFromPage() {
        // Lấy text từ body, loại bỏ scripts và styles
        const clone = document.body.cloneNode(true);

        // Remove script, style, và panel của chúng ta
        clone.querySelectorAll('script, style, #ner-side-panel, #ner-floating-btn').forEach(el => el.remove());

        return clone.innerText.trim();
    }

    async function processManualMode(text) {
        const response = await fetch(`${API_BASE_URL}/api/ner/extract-manual`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Unknown error');
        }

        return data;
    }

    async function processAutoMode(text) {
        const response = await fetch(`${API_BASE_URL}/api/ner/extract-auto`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Unknown error');
        }

        return data;
    }

    // ============================================================================
    // DISPLAY RESULTS
    // ============================================================================

    function displayResults(results, mode) {
        if (mode === 'manual') {
            displayManualResults(results);
        } else {
            displayAutoResults(results);
        }

        // Update processing info
        elements.processingInfo.textContent = `Xử lý xong trong ${results.processing_time.toFixed(2)}s`;

        // Show highlight button only if input source is webpage
        const inputSource = document.querySelector('input[name="input-source"]:checked').value;
        if (inputSource === 'webpage') {
            elements.highlightEntitiesBtn.style.display = 'inline-block';
        } else {
            elements.highlightEntitiesBtn.style.display = 'none';
        }
    }

    function displayManualResults(data) {
        // Display entities
        displayEntities(data.entities);

        // Display single patient
        displayPatients([{
            patient_index: 1,
            patient_record: data.patient_record
        }]);
    }

    function displayAutoResults(data) {
        // Aggregate all entities from all patients
        const allEntities = [];
        data.patients.forEach(p => {
            allEntities.push(...p.entities);
        });
        displayEntities(allEntities);

        // Display all patients
        displayPatients(data.patients);
    }

    function displayEntities(entities) {
        if (!entities || entities.length === 0) {
            elements.entitiesList.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📭</div><div class="empty-state-text">Không tìm thấy entities</div></div>';
            return;
        }

        elements.entitiesList.innerHTML = entities.map(entity => {
            const color = getEntityColor(entity.tag);
            return `
                <div class="entity-item" style="border-left-color: ${color}">
                    <div class="entity-text">${escapeHtml(entity.text)}</div>
                    <div class="entity-tag">${ENTITY_LABELS[entity.tag] || entity.tag}</div>
                </div>
            `;
        }).join('');
    }

    function displayPatients(patients) {
        if (!patients || patients.length === 0) {
            elements.patientsList.innerHTML = '<div class="empty-state"><div class="empty-state-icon">👤</div><div class="empty-state-text">Không tìm thấy thông tin bệnh nhân</div></div>';
            updateUndoButton();
            return;
        }

        // Determine if this is manual mode (1 patient only)
        const isManualMode = patients.length === 1 && patients[0].patient_index === 1;

        elements.patientsList.innerHTML = patients.map(p => {
            const record = p.patient_record;
            const isDeleted = panelState.deletedPatientIndices.includes(p.patient_index);
            const deleteBtnHtml = !isManualMode && !isDeleted
                ? `<button class="delete-patient-btn" data-index="${p.patient_index}" title="Xóa bệnh nhân">×</button>`
                : '';
            const cardClass = isDeleted ? 'patient-card deleted' : 'patient-card';

            return `
                <div class="${cardClass}" data-index="${p.patient_index}">
                    <div class="patient-header">
                        Bệnh nhân ${p.patient_index}
                        ${deleteBtnHtml}
                    </div>
                    ${renderPatientField('Mã BN', record.patient_id)}
                    ${renderPatientField('Họ tên', record.name)}
                    ${renderPatientField('Tuổi', record.age)}
                    ${renderPatientField('Giới tính', record.gender)}
                    ${renderPatientField('Nghề nghiệp', record.job)}
                    ${renderPatientListField('Địa điểm', record.locations)}
                    ${renderPatientListField('Tổ chức', record.organizations)}
                    ${renderPatientListField('Triệu chứng/Bệnh', record.symptoms_and_diseases)}
                    ${renderPatientListField('Phương tiện', record.transportations)}
                    ${renderPatientDatesField(record.dates)}
                    ${renderPatientListField('Cảnh báo', record.warnings)}
                </div>
            `;
        }).join('');

        // Add event listeners for delete buttons
        document.querySelectorAll('.delete-patient-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const index = parseInt(btn.dataset.index);
                deletePatient(index);
            });
        });

        updateUndoButton();
    }

    function renderPatientField(label, value) {
        if (!value) return '';
        return `<div class="patient-field"><strong>${label}:</strong> ${escapeHtml(value)}</div>`;
    }

    function renderPatientListField(label, list) {
        if (!list || list.length === 0) return '';
        return `
            <div class="patient-field">
                <strong>${label}:</strong>
                <div class="patient-field-list">
                    ${list.map(item => `<div class="patient-field-list-item">• ${escapeHtml(item)}</div>`).join('')}
                </div>
            </div>
        `;
    }

    function renderPatientDatesField(dates) {
        if (!dates || Object.keys(dates).length === 0) return '';

        const dateTypes = {
            'admission_date': 'Nhập viện',
            'discharge_date': 'Xuất viện',
            'test_date': 'Xét nghiệm',
            'positive_date': 'Dương tính',
            'negative_date': 'Âm tính',
            'entry_date': 'Nhập cảnh',
            'recovery_date': 'Hồi phục',
            'death_date': 'Tử vong',
            'unknown_date': 'Khác'
        };

        const items = [];
        Object.entries(dates).forEach(([type, dateList]) => {
            if (dateList && dateList.length > 0) {
                const label = dateTypes[type] || type;
                items.push(`<div class="patient-field-list-item">• ${label}: ${dateList.join(', ')}</div>`);
            }
        });

        if (items.length === 0) return '';

        return `
            <div class="patient-field">
                <strong>Ngày tháng:</strong>
                <div class="patient-field-list">
                    ${items.join('')}
                </div>
            </div>
        `;
    }

    // ============================================================================
    // COPY CSV FUNCTIONS
    // ============================================================================

    function copyEntitiesCSV() {
        if (!panelState.lastResults) {
            alert('Chưa có dữ liệu để copy');
            return;
        }

        let entities = [];
        if (panelState.lastResults.entities) {
            entities = panelState.lastResults.entities;
        } else if (panelState.lastResults.patients) {
            // Auto mode: aggregate entities
            panelState.lastResults.patients.forEach(p => {
                entities.push(...p.entities);
            });
        }

        const csv = convertEntitiesToCSV(entities);
        copyToClipboard(csv);
        alert('Đã copy entities sang CSV!');
    }

    function copyPatientsCSV() {
        if (!panelState.lastResults) {
            alert('Chưa có dữ liệu để copy');
            return;
        }

        let patients = [];
        if (panelState.lastResults.patient_record) {
            // Manual mode
            patients = [{
                patient_index: 1,
                patient_record: panelState.lastResults.patient_record
            }];
        } else if (panelState.lastResults.patients) {
            // Auto mode
            patients = panelState.lastResults.patients;
        }

        const csv = convertPatientsToCSV(patients);
        copyToClipboard(csv);
        alert('Đã copy thông tin bệnh nhân sang CSV!');
    }

    function downloadPatientsCSV() {
        if (!panelState.lastResults) {
            alert('Chưa có dữ liệu để tải xuống');
            return;
        }

        let patients = [];
        if (panelState.lastResults.patient_record) {
            // Manual mode
            patients = [{
                patient_index: 1,
                patient_record: panelState.lastResults.patient_record
            }];
        } else if (panelState.lastResults.patients) {
            // Auto mode
            patients = panelState.lastResults.patients;
        }

        const csv = convertPatientsToCSV(patients);
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
        const filename = `covid19_patients_${timestamp}.csv`;
        downloadFile(csv, filename);
    }

    function highlightEntitiesOnPage() {
        if (!panelState.lastResults) {
            alert('Chưa có dữ liệu để highlight');
            return;
        }

        // Collect all entities
        let allEntities = [];
        if (panelState.lastResults.entities) {
            allEntities = panelState.lastResults.entities;
        } else if (panelState.lastResults.patients) {
            // Auto mode: aggregate entities
            panelState.lastResults.patients.forEach(p => {
                allEntities.push(...p.entities);
            });
        }

        if (allEntities.length === 0) {
            alert('Không có entities để highlight');
            return;
        }

        // Send message to content script to highlight
        window.postMessage({
            type: 'NER_HIGHLIGHT_ENTITIES',
            entities: allEntities,
            source: 'ner-side-panel'
        }, '*');

        console.log(`[NER Extension] Sent ${allEntities.length} entities to highlight`);
        alert(`Đã highlight ${allEntities.length} entities trên trang!`);
    }

    function convertEntitiesToCSV(entities) {
        const headers = ['Text', 'Tag', 'Start', 'End'];
        const rows = entities.map(e => [
            escapeCSV(e.text),
            escapeCSV(ENTITY_LABELS[e.tag] || e.tag),
            e.start,
            e.end
        ]);

        return [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');
    }

    function convertPatientsToCSV(patients) {
        const BOM = '\uFEFF';  // UTF-8 BOM for Excel compatibility

        const headers = [
            'STT',
            'Mã BN',
            'Họ tên',
            'Tuổi',
            'Giới tính',
            'Nghề nghiệp',
            'Địa điểm',
            'Tổ chức',
            'Triệu chứng/Bệnh',
            'Phương tiện',
            'Ngày nhập viện',
            'Ngày xuất viện',
            'Ngày xét nghiệm',
            'Ngày dương tính',
            'Ngày âm tính',
            'Ngày nhập cảnh',
            'Ngày hồi phục',
            'Ngày tử vong',
            'Ngày khác'
        ];

        // Filter out deleted patients (only for Auto mode)
        const activePatients = patients.filter(p =>
            !panelState.deletedPatientIndices.includes(p.patient_index)
        );

        const rows = activePatients.map(p => {
            const r = p.patient_record;
            const dates = r.dates || {};

            return [
                p.patient_index || '',
                r.patient_id || '',
                r.name || '',
                r.age || '',
                r.gender || '',
                r.job || '',
                (r.locations || []).join('; '),
                (r.organizations || []).join('; '),
                (r.symptoms_and_diseases || []).join('; '),
                (r.transportations || []).join('; '),
                (dates.admission_date || []).join('; '),
                (dates.discharge_date || []).join('; '),
                (dates.test_date || []).join('; '),
                (dates.positive_date || []).join('; '),
                (dates.negative_date || []).join('; '),
                (dates.entry_date || []).join('; '),
                (dates.recovery_date || []).join('; '),
                (dates.death_date || []).join('; '),
                (dates.unknown_date || []).join('; ')
            ].map(escapeCSV);
        });

        return BOM + [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');
    }

    function escapeCSV(value) {
        if (value === null || value === undefined) return '';
        const str = String(value);
        if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return `"${str.replace(/"/g, '""')}"`;
        }
        return str;
    }

    function copyToClipboard(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    }

    // ============================================================================
    // DELETE & UNDO PATIENT
    // ============================================================================

    function deletePatient(patientIndex) {
        if (!confirm(`Bạn có chắc muốn xóa Bệnh nhân ${patientIndex}?`)) {
            return;
        }

        // Add to deleted list
        if (!panelState.deletedPatientIndices.includes(patientIndex)) {
            panelState.deletedPatientIndices.push(patientIndex);
        }

        // Re-render patients list
        let patients = [];
        if (panelState.lastResults.patients) {
            patients = panelState.lastResults.patients;
        }
        displayPatients(patients);

        console.log('[NER Extension] Deleted patient:', patientIndex);
    }

    function undoDeletePatient() {
        if (panelState.deletedPatientIndices.length === 0) {
            alert('Không có bệnh nhân nào để hoàn tác');
            return;
        }

        // Remove last deleted patient from list
        const restoredIndex = panelState.deletedPatientIndices.pop();

        // Re-render patients list
        let patients = [];
        if (panelState.lastResults.patients) {
            patients = panelState.lastResults.patients;
        }
        displayPatients(patients);

        console.log('[NER Extension] Restored patient:', restoredIndex);
    }

    function updateUndoButton() {
        if (!elements.undoDeleteBtn) return;

        const hasDeleted = panelState.deletedPatientIndices.length > 0;
        elements.undoDeleteBtn.style.display = hasDeleted ? 'inline-block' : 'none';

        if (hasDeleted) {
            const count = panelState.deletedPatientIndices.length;
            elements.undoDeleteBtn.textContent = `↶ Hoàn tác (${count})`;
        }
    }

    // ============================================================================
    // UTILITIES
    // ============================================================================

    function showLoading(show) {
        elements.loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function downloadFile(content, filename) {
        const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        console.log(`[NER Extension] Downloaded ${filename}`);
    }

    // ============================================================================
    // INIT
    // ============================================================================

    // Chạy khi DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
