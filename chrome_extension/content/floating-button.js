// floating-button.js
// Tạo và quản lý nút floating ở góc phải màn hình

(function () {
    'use strict';

    // Tạo floating button
    function createFloatingButton() {
        const button = document.createElement('button');
        button.id = 'ner-floating-btn';
        button.innerHTML = '🦠';
        button.setAttribute('data-tooltip', 'Mở COVID-19 NER Panel');
        button.setAttribute('aria-label', 'Mở COVID-19 NER Panel');

        // Event listener
        button.addEventListener('click', togglePanel);

        return button;
    }

    // Toggle panel hiển thị
    function togglePanel() {
        // Gửi message tới side-panel.js để toggle
        window.postMessage({
            type: 'NER_TOGGLE_PANEL',
            source: 'ner-floating-button'
        }, '*');
    }

    // Lắng nghe trạng thái panel để cập nhật button position
    window.addEventListener('message', (event) => {
        if (event.data.type === 'NER_PANEL_STATE_CHANGED') {
            const button = document.getElementById('ner-floating-btn');
            if (button) {
                if (event.data.isOpen) {
                    button.classList.add('panel-open');
                    button.setAttribute('data-tooltip', 'Đóng COVID-19 NER Panel');
                } else {
                    button.classList.remove('panel-open');
                    button.setAttribute('data-tooltip', 'Mở COVID-19 NER Panel');
                }
            }
        }
    });

    // Inject CSS
    function injectCSS() {
        const linkId = 'ner-floating-button-css';
        if (!document.getElementById(linkId)) {
            const link = document.createElement('link');
            link.id = linkId;
            link.rel = 'stylesheet';
            link.href = chrome.runtime.getURL('content/floating-button.css');
            document.head.appendChild(link);
        }
    }

    // Khởi tạo
    function init() {
        // Kiểm tra xem button đã tồn tại chưa
        if (document.getElementById('ner-floating-btn')) {
            return;
        }

        // Inject CSS
        injectCSS();

        // Tạo và thêm button vào DOM
        const button = createFloatingButton();
        document.body.appendChild(button);

        console.log('[NER Extension] Floating button initialized');
    }

    // Chạy khi DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
