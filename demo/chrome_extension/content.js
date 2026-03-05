// content.js
// Injects the floating water droplet UI and monitors textbox inputs

console.log("HydroTwin Tracker initialized.");

let waterTotalML = 0;
const WATER_PER_QUERY_ML = 22.5; // Average water cost of an LLM query in milliliters

// Create floating UI
const ui = document.createElement("div");
ui.id = "hydrotwin-floating-tracker";
ui.innerHTML = `
  <div class="hydro-logo">💧</div>
  <div class="hydro-text">
    <span id="hydro-ml">0</span> ml
    <div class="hydro-sub">Water Footprint</div>
  </div>
`;
document.body.appendChild(ui);

// Listen for enter key or clicking the send button to trigger a query simulation
document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        // Very basic detection — if enter is pressed inside a textarea 
        if (document.activeElement.tagName === "TEXTAREA" || document.activeElement.isContentEditable) {
            triggerWaterSpike();
        }
    }
});

document.addEventListener("click", (e) => {
    // Detect clicking a "send" button icon commonly used in AI UIs
    if (e.target.closest('button[data-testid="send-button"]') ||
        e.target.closest('button[aria-label="Send message"]')) {
        triggerWaterSpike();
    }
});

function triggerWaterSpike() {
    waterTotalML += WATER_PER_QUERY_ML;
    const counter = document.getElementById("hydro-ml");

    // Animate the counter
    ui.classList.add("pulse");
    setTimeout(() => ui.classList.remove("pulse"), 500);

    // Update number
    counter.innerText = waterTotalML.toFixed(1);

    // Save to storage
    chrome.storage.local.set({ totalWater: waterTotalML });
}
