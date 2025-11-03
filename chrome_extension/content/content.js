// chrome_extension/content/content.js
/**
 * Content Script - Chay trong context cua trang web
 * Chuc nang:
 * 1. Trich xuat text tu trang web
 * 2. Highlight entities truc tiep tren DOM
 */

console.log('COVID-19 NER Content Script loaded');

// CSS da duoc inject, check xem da inject chua
let highlightStylesInjected = false;

/**
 * Trich xuat text tu trang web
 */
function extractPageText() {
    // Chien luoc: Tim vung noi dung chinh
    let mainContent = null;

    // Thu tim cac elements theo thu tu uu tien
    const selectors = [
        'article',
        'main',
        '[role="main"]',
        '.content',
        '#content',
        'body'
    ];

    for (const selector of selectors) {
        mainContent = document.querySelector(selector);
        if (mainContent) {
            console.log(`Found content in: ${selector}`);
            break;
        }
    }

    if (!mainContent) {
        mainContent = document.body;
    }

    // Loai bo cac elements khong can thiet
    const clone = mainContent.cloneNode(true);

    const removeSelectors = [
        'script',
        'style',
        'svg',
        'nav',
        'aside',
        'footer',
        'header',
        '[class*="ad"]',
        '[class*="menu"]',
        '[class*="sidebar"]',
        '[class*="comment"]',
        'iframe'
    ];

    removeSelectors.forEach(selector => {
        const elements = clone.querySelectorAll(selector);
        elements.forEach(el => el.remove());
    });

    // Lay text
    let text = clone.innerText || clone.textContent;

    // Lam sach
    text = text
        .replace(/\s+/g, ' ')  // Nhieu space -> 1 space
        .replace(/\n\s*\n/g, '\n\n')  // Giu line breaks
        .trim();

    // Gioi han do dai (tăng lên 500K)
    const MAX_LENGTH = 500000;
    if (text.length > MAX_LENGTH) {
        text = text.substring(0, MAX_LENGTH);
        console.warn(`Text truncated to ${MAX_LENGTH} characters`);
    }

    return {
        text: text,
        length: text.length,
        source: mainContent.tagName.toLowerCase()
    };
}

/**
 * Highlight entities tren trang web
 */
function highlightEntities(entities) {
    console.log('Highlighting entities:', entities.length);

    // Inject CSS neu chua co
    if (!highlightStylesInjected) {
        injectHighlightStyles();
        highlightStylesInjected = true;
    }

    // Xoa highlights cu neu co
    clearHighlights();

    // Group entities theo text de xu ly
    const entityMap = new Map();
    entities.forEach(entity => {
        if (!entityMap.has(entity.text)) {
            entityMap.set(entity.text, []);
        }
        entityMap.get(entity.text).push(entity);
    });

    // Highlight tung entity
    let highlightedCount = 0;

    entityMap.forEach((entityList, entityText) => {
        const entity = entityList[0]; // Lay entity dau tien

        // Tao TreeWalker de duyet text nodes
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        let node;
        const nodesToHighlight = [];

        while (node = walker.nextNode()) {
            if (node.nodeValue && node.nodeValue.includes(entityText)) {
                // Kiem tra parent khong phai la highlight node
                if (!node.parentElement.classList.contains('ner-entity')) {
                    nodesToHighlight.push(node);
                }
            }
        }

        // Highlight cac nodes
        nodesToHighlight.forEach(textNode => {
            highlightTextInNode(textNode, entityText, entity.tag);
            highlightedCount++;
        });
    });

    console.log(`Highlighted ${highlightedCount} entities`);

    // Thong bao cho user
    showHighlightNotification(highlightedCount);
}

/**
 * Highlight text trong mot text node
 */
function highlightTextInNode(textNode, searchText, tag) {
    const text = textNode.nodeValue;
    const index = text.indexOf(searchText);

    if (index === -1) return;

    // Split text node
    const beforeText = text.substring(0, index);
    const matchText = text.substring(index, index + searchText.length);
    const afterText = text.substring(index + searchText.length);

    // Tao highlight element
    const mark = document.createElement('mark');
    mark.className = `ner-entity ner-${tag}`;
    mark.setAttribute('data-entity-type', tag);
    mark.textContent = matchText;
    mark.title = `${matchText} - ${tag}`;

    // Replace text node
    const parent = textNode.parentNode;

    if (beforeText) {
        parent.insertBefore(document.createTextNode(beforeText), textNode);
    }
    parent.insertBefore(mark, textNode);
    if (afterText) {
        parent.insertBefore(document.createTextNode(afterText), textNode);
    }
    parent.removeChild(textNode);
}

/**
 * Xoa tat ca highlights
 */
function clearHighlights() {
    const highlights = document.querySelectorAll('.ner-entity');
    highlights.forEach(mark => {
        const text = mark.textContent;
        const textNode = document.createTextNode(text);
        mark.parentNode.replaceChild(textNode, mark);
    });

    // Xoa notification neu co
    const notification = document.getElementById('ner-highlight-notification');
    if (notification) {
        notification.remove();
    }
}

/**
 * Inject CSS cho highlighting
 */
function injectHighlightStyles() {
    // CSS da duoc inject qua manifest, nhung co the them custom styles
    console.log('Highlight styles ready');
}

/**
 * Hien thi notification khi highlight xong
 */
function showHighlightNotification(count) {
    // Xoa notification cu
    const oldNotif = document.getElementById('ner-highlight-notification');
    if (oldNotif) {
        oldNotif.remove();
    }

    // Tao notification moi
    const notification = document.createElement('div');
    notification.id = 'ner-highlight-notification';
    notification.className = 'ner-notification';
    notification.innerHTML = `
        <div class="ner-notification-content">
            <strong>COVID-19 NER</strong>
            <p>Da highlight ${count} entities tren trang</p>
            <button id="ner-clear-highlights">Xoa highlights</button>
        </div>
    `;

    document.body.appendChild(notification);

    // Auto hide sau 5s
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, 5000);

    // Clear button
    const clearBtn = document.getElementById('ner-clear-highlights');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            clearHighlights();
            notification.remove();
        });
    }
}

/**
 * Message listener
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Content script received message:', request.type);

    if (request.type === 'GET_PAGE_TEXT') {
        try {
            const result = extractPageText();
            sendResponse({
                success: true,
                ...result
            });
        } catch (error) {
            console.error('Error extracting text:', error);
            sendResponse({
                success: false,
                error: error.message
            });
        }
        return true;
    }

    if (request.type === 'HIGHLIGHT_ENTITIES') {
        try {
            highlightEntities(request.entities);
            sendResponse({
                success: true,
                count: request.entities.length
            });
        } catch (error) {
            console.error('Error highlighting:', error);
            sendResponse({
                success: false,
                error: error.message
            });
        }
        return true;
    }

    if (request.type === 'CLEAR_HIGHLIGHTS') {
        try {
            clearHighlights();
            sendResponse({
                success: true
            });
        } catch (error) {
            console.error('Error clearing highlights:', error);
            sendResponse({
                success: false,
                error: error.message
            });
        }
        return true;
    }

    return true;
});

// Listen for window messages from side-panel
window.addEventListener('message', (event) => {
    // Only accept messages from same origin
    if (event.source !== window) return;

    if (event.data.type === 'NER_HIGHLIGHT_ENTITIES' && event.data.source === 'ner-side-panel') {
        console.log('[NER Content] Received highlight request:', event.data.entities.length, 'entities');
        try {
            highlightEntities(event.data.entities);
            console.log('[NER Content] Highlighted successfully');
        } catch (error) {
            console.error('[NER Content] Error highlighting:', error);
        }
    }
});

console.log('[NER Content] Window message listener registered');
