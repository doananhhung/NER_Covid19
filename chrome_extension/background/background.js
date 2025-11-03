// chrome_extension/background/background.js
/**
 * Background Service Worker
 * Quan ly lifecycle cua extension va routing messages
 */

console.log('Background service worker started');

// Listener khi extension duoc cai dat
chrome.runtime.onInstalled.addListener((details) => {
    console.log('Extension installed:', details);

    // Khoi tao storage voi default values
    chrome.storage.local.set({
        gemini_api_key: '',
        last_mode: 'manual',
        last_source: 'manual'
    });
});

// Listener cho messages tu popup hoac content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Background received message:', request);

    // Forward messages neu can
    // Hien tai popup goi API truc tiep nen khong can routing phuc tap

    return true; // Keep channel open for async response
});

// Optional: Context menu integration (co the them sau)
// chrome.contextMenus.create({
//     id: "analyze-selection",
//     title: "Phan tich van ban da chon",
//     contexts: ["selection"]
// });
