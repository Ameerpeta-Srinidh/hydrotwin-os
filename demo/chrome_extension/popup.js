// popup.js
document.addEventListener('DOMContentLoaded', () => {
    chrome.storage.local.get(['totalWater'], (result) => {
        if (result.totalWater) {
            document.getElementById('total-ml').innerText = result.totalWater.toFixed(1);
        }
    });
});
