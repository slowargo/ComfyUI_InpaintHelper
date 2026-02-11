import { ComfyApp } from "../../scripts/app.js";
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// === Turbo Mode State ===
const turboState = {
    active: false,         // 防止重入
    enabled: false,        // Turbo Mode 是否启用（通过 toggle 控制）
    sourceNodeId: null,    // 发起 turbo 的节点 ID
};

// === Color Memory ===
const colorMemory = {
    key: "slowargo_turbo_color",
    save: function(hexColor) {
        localStorage.setItem(this.key, hexColor);
    },
    load: function() {
        return localStorage.getItem(this.key);
    }
};

// === Turbo Mode Helper Functions ===

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function getTurboTargetNode() {
    // 使用 open_maskeditor 打开 mask editor 时。兜底值，不一定是当前选择的节点
    let selectedNode = ComfyApp.clipspace_return_node;

    // 使用 commandStore.execute('Comfy.MaskEditor.OpenMaskEditor') 打开 mask editor 时。最常用打开方式，优先使用
    const selectedNodes = app.canvas.selected_nodes;
    if (selectedNodes && Object.keys(selectedNodes).length === 1) {
        selectedNode = Object.values(selectedNodes)[0];
        console.log("[slowargo.js] Turbo: using selected node from canvas");
    }

    if (selectedNode?.comfyClass !== "LoadRecentImagePlusV1") {
        return null;
    }
    return selectedNode;
}

function isMaskNonEmpty() {
    const canvases = document.querySelectorAll("#maskEditorCanvasContainer canvas");
    if (canvases.length < 3) return false;

    const maskCanvas = canvases[2]; // z-30, the mask layer
    const ctx = maskCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return false;

    const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const data = imageData.data;

    // Sample every 16th pixel for performance
    const step = 4 * 16;
    for (let i = 3; i < data.length; i += step) {
        if (data[i] > 0) return true;
    }
    return false;
}

async function performMaskSave() {
    // Save current color before saving mask
    const colorInput = document.querySelector("div.maskEditor_sidePanel input[type=color]");
    if (colorInput && colorInput.value) {
        colorMemory.save(colorInput.value);
    }

    // Wait for GPU readback
    const canvases = document.querySelectorAll("#maskEditorCanvasContainer canvas");
    for (let i = 0; i < 50; i++) {
        if (![...canvases].some((c) => c.style.opacity === "0")) break;
        await sleep(100);
    }

    // Wait for save button to be enabled and click it
    for (let i = 0; i < 20; i++) {
        const btn = document.querySelector("#global-mask-editor button:has(i.pi-check)");
        if (!btn) throw new Error("Save button not found");
        if (!btn.disabled) {
            btn.click();
            return;
        }
        await sleep(100);
    }
    throw new Error("Save button remained disabled");
}

function waitForExecutionComplete(timeoutMs = 120000) {
    return new Promise((resolve, reject) => {
        const timeoutId = setTimeout(() => {
            cleanup();
            reject(new Error("Execution timed out"));
        }, timeoutMs);

        function onExecutionSuccess(event) {
            cleanup();
            resolve("success");
        }

        function onExecutionError(event) {
            cleanup();
            reject(new Error("Execution error: " + (event?.detail?.exception_message || "unknown")));
        }

        function onExecutionInterrupted(event) {
            cleanup();
            reject(new Error("Execution interrupted"));
        }

        function cleanup() {
            clearTimeout(timeoutId);
            api.removeEventListener("execution_success", onExecutionSuccess);
            api.removeEventListener("execution_error", onExecutionError);
            api.removeEventListener("execution_interrupted", onExecutionInterrupted);
        }

        api.addEventListener("execution_success", onExecutionSuccess);
        api.addEventListener("execution_error", onExecutionError);
        api.addEventListener("execution_interrupted", onExecutionInterrupted);
    });
}

async function executeTurboCycle(targetNode) {
    turboState.active = true;

    try {
        console.log("[slowargo.js] Turbo Mode: Saving mask...");
        await performMaskSave();

        // Wait for editor to close completely
        console.log("[slowargo.js] Turbo Mode: Waiting for editor to close...");
        for (let i = 0; i < 50; i++) {
            if (!ComfyApp.maskeditor_is_opended()) break;
            await sleep(100);
        }

        console.log("[slowargo.js] Turbo Mode: Executing workflow...");
        app.queuePrompt(0);

        await waitForExecutionComplete();

        await sleep(500);

        console.log("[slowargo.js] Turbo Mode: Refreshing image...");
        const node = app.graph.getNodeById(targetNode.id);
        if (!node || !node.refreshImageList) {
            console.warn("[slowargo.js] Turbo Mode: target node lost or no refreshFn");
            return;
        }
        await node.refreshImageList();

        await sleep(300);

        console.log("[slowargo.js] Turbo Mode: Opening editor...");
        ComfyApp.clipspace_return_node = node;
        turboState.sourceNodeId = node.id;
        ComfyApp.open_maskeditor?.();

        // MutationObserver will automatically call restoreColorAndAddToggle when editor opens
        console.log("[slowargo.js] Turbo Mode: cycle complete");

    } catch (error) {
        console.error("[slowargo.js] Turbo Mode: cycle failed", error);
    } finally {
        turboState.active = false;
    }
}

function restoreColorAndAddToggle() {
    const colorInput = document.querySelector("div.maskEditor_sidePanel input[type=color]");
    const savedColor = colorMemory.load();

    if (colorInput && savedColor) {
        colorInput.value = savedColor;
        colorInput.dispatchEvent(new Event('input', { bubbles: true }));
        colorInput.dispatchEvent(new Event('change', { bubbles: true }));
        console.log("[slowargo.js] Restored color:", savedColor);
    }

    // Add turbo toggle button if not already added
    addTurboToggleButton();
}

function addTurboToggleButton() {
    // Check if toggle already exists
    if (document.querySelector(".turbo-mode-toggle")) {
        return;
    }

    const targetNode = getTurboTargetNode();
    if (!targetNode) {
        console.log("[slowargo.js] Turbo Mode: target node not found, skipping toggle");
        return;
    }

    turboState.sourceNodeId = targetNode.id;

    // Find a button in the topbar
    const refBtn = document.querySelector("#global-mask-editor button:has(i.pi-check)");
    if (!refBtn) {
        console.warn("[slowargo.js] Turbo Mode: undo button not found");
        return;
    }

    // Create toggle button
    const toggleBtn = document.createElement("button");
    toggleBtn.className = "turbo-mode-toggle";
    // toggleBtn.title = "Turbo Mode: Press Enter to save and run prompt, and auto refresh after completion\n(CapsLock to toggle)";
    toggleBtn.title = "Turbo Mode: Press Enter to save and run prompt, and auto refresh after completion";
    const icon = document.createElement("i");
    icon.className = "pi pi-fast-forward";
    toggleBtn.appendChild(icon);
    const text = document.createElement("span");
    text.textContent = "Turbo";
    toggleBtn.appendChild(text);
    toggleBtn.style.cssText = `
        background: #1e90ff;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        margin-right: 8px;
        cursor: pointer;
        font-size: 12px;
        font-weight: bold;
        opacity: 0.6;
        transition: opacity 0.2s;
        display: flex;
        align-items: center;
        gap: 4px;
    `;

    // Update style based on enabled state
    function updateToggleStyle() {
        if (turboState.enabled) {
            toggleBtn.style.opacity = "1";
            toggleBtn.style.boxShadow = "0 0 8px rgba(30, 144, 255, 0.8)";
        } else {
            toggleBtn.style.opacity = "0.6";
            toggleBtn.style.boxShadow = "none";
        }
    }

    toggleBtn.addEventListener("click", () => {
        turboState.enabled = !turboState.enabled;
        updateToggleStyle();
        console.log("[slowargo.js] Turbo Mode:", turboState.enabled ? "enabled" : "disabled");
    });

    // Insert before undo button
    //refBtn.parentNode.insertBefore(toggleBtn, refBtn);
    refBtn.parentNode.appendChild(toggleBtn);

    updateToggleStyle();
}

// === Turbo Mode Initialization ===

export function initTurboMode() {
    // Sync Turbo Mode with CapsLock state (CapsLock ON = Turbo ON, OFF = Turbo OFF)
    window.addEventListener('keydown', async function(e) {
        if (!ComfyApp.maskeditor_is_opended()) return;

        // const capsLockOn = e.getModifierState('CapsLock');
        const targetNode = getTurboTargetNode();

        // Sync CapsLock state with Turbo Mode enabled state
        // if (capsLockOn !== turboState.enabled && targetNode) {
        //     turboState.enabled = capsLockOn;
        //     const toggleBtn = document.querySelector(".turbo-mode-toggle");
        //     if (toggleBtn) {
        //         const style = toggleBtn.style;
        //         if (turboState.enabled) {
        //             style.opacity = "1";
        //             style.boxShadow = "0 0 8px rgba(30, 144, 255, 0.8)";
        //         } else {
        //             style.opacity = "0.6";
        //             style.boxShadow = "none";
        //         }
        //     }
        //     console.log("[slowargo.js] Turbo Mode:", turboState.enabled ? "enabled" : "disabled");
        // }

        // Enter executes turbo cycle if enabled and mask is not empty
        if (e.key !== 'Enter') return;
        if (!turboState.enabled) return;
        if (turboState.active) return;
        if (!targetNode) return;

        if (!isMaskNonEmpty()) {
            console.log("[slowargo.js] Turbo Mode: mask is empty, skipping");
            return;
        }

        e.preventDefault();
        e.stopImmediatePropagation();

        console.log("[slowargo.js] Turbo Mode: starting cycle for node", targetNode.id);
        await executeTurboCycle(targetNode);
    }, true);

    // Monitor mask editor container to detect opening (handles all open methods)
    const observer = new MutationObserver(() => {
        //const maskEditorPanel = document.querySelector("div.maskEditor_sidePanel");
        const refBtn = document.querySelector("#global-mask-editor button:has(i.pi-check)");
        if (refBtn && !document.querySelector(".turbo-mode-toggle")) {
            // Mask editor just opened, add toggle and restore color
            // setTimeout(() => {
            //     restoreColorAndAddToggle();
            // }, 100);
            restoreColorAndAddToggle();
        }
    });

    // Start observing document for changes
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: false,
        characterData: false
    });
}

// Export performMaskSave for use in maskeditor.save command
export { performMaskSave };
