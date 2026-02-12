import { ComfyApp } from "../../scripts/app.js";
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// === Fast Forward Mode State ===
const fastForwardState = {
    active: false,         // 防止重入
    enabled: true,        // Fast Forward Mode 是否启用（通过 toggle 控制）
    sourceNodeId: null,    // 发起 fast forward 的节点 ID
};

// === Color Memory ===
const colorMemory = {
    key: "slowargo_fast_forward_color",
    save: function(hexColor) {
        localStorage.setItem(this.key, hexColor);
    },
    load: function() {
        return localStorage.getItem(this.key);
    }
};

// === Load Clipspace Content to Current Editor ===
async function loadClipspaceToEditor() {
    try {
        const response = await api.fetchApi('/slowargo_api/refresh_previews_recent', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({watch_folders: 'clipspace [1][input]'})
        });

        const data = await response.json();
        if (!data.image_name?.[0]) {
            console.warn("[slowargo.js] No clipspace files found");
            return;
        }

        const timestamp = data.image_name[0].match(/clipspace-painted-masked-(\d+)\.png/)?.[1];
        if (!timestamp) {
            console.warn("[slowargo.js] Invalid clipspace filename");
            return;
        }

        const canvases = document.querySelectorAll("#maskEditorCanvasContainer canvas");
        if (canvases.length < 3) return;

        console.log("[slowargo.js] Loading clipspace, timestamp:", timestamp);

        const params = `${app.getPreviewFormatParam?.() || ""}${app.getRandParam?.() || ""}`;
        const loadImg = (url) => new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });

        // Load base and mask in parallel
        const [baseImg, maskImg] = await Promise.all([
            loadImg(api.apiURL(`/view?filename=clipspace-mask-${timestamp}.png&subfolder=clipspace&type=input&channel=rgb${params}`)),
            loadImg(api.apiURL(`/view?filename=clipspace-mask-${timestamp}.png&subfolder=clipspace&type=input&channel=a${params}`))
        ]);

        // Draw base
        const baseCtx = canvases[0].getContext('2d', {willReadFrequently: true});
        baseCtx.clearRect(0, 0, canvases[0].width, canvases[0].height);
        baseCtx.drawImage(baseImg, 0, 0, canvases[0].width, canvases[0].height);
        baseImg.src = '';

        // Draw and invert mask
        const maskCtx = canvases[2].getContext('2d', {willReadFrequently: true});
        maskCtx.clearRect(0, 0, canvases[2].width, canvases[2].height);
        maskCtx.drawImage(maskImg, 0, 0, canvases[2].width, canvases[2].height);
        maskImg.src = '';

        const maskData = maskCtx.getImageData(0, 0, canvases[2].width, canvases[2].height);
        for (let i = 3; i < maskData.data.length; i += 4) {
            maskData.data[i] = 255 - maskData.data[i];
        }
        maskCtx.putImageData(maskData, 0, 0);

        // Load paint layer (optional)
        try {
            const paintImg = await loadImg(api.apiURL(`/view?filename=clipspace-paint-${timestamp}.png&subfolder=clipspace&type=input${params}`));
            const paintCtx = canvases[1].getContext('2d', {willReadFrequently: true});
            paintCtx.clearRect(0, 0, canvases[1].width, canvases[1].height);
            paintCtx.drawImage(paintImg, 0, 0, canvases[1].width, canvases[1].height);
            paintImg.src = '';
        } catch (e) {
            // Paint layer is optional
        }

        console.log("[slowargo.js] Clipspace loaded successfully");
    } catch (error) {
        console.error("[slowargo.js] Failed to load clipspace:", error);
    }
}
// === Fast Forward Mode Helper Functions ===

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function getFastForwardTargetNode() {
    // 使用 open_maskeditor 打开 mask editor 时。兜底值，不一定是当前选择的节点
    let selectedNode = ComfyApp.clipspace_return_node;

    // 使用 commandStore.execute('Comfy.MaskEditor.OpenMaskEditor') 打开 mask editor 时。最常用打开方式，优先使用
    const selectedNodes = app.canvas.selected_nodes;
    if (selectedNodes && Object.keys(selectedNodes).length === 1) {
        selectedNode = Object.values(selectedNodes)[0];
    } else {
        console.log("[slowargo.js] Fast Forward mode: using clipspace_return_node from canvas");
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

async function executeFastForwardCycle(targetNode) {
    fastForwardState.active = true;

    try {
        console.log("[slowargo.js] Fast Forward Mode: Saving mask...");
        await performMaskSave();

        // Wait for editor to close completely
        console.log("[slowargo.js] Fast Forward Mode: Waiting for editor to close...");
        for (let i = 0; i < 50; i++) {
            if (!ComfyApp.maskeditor_is_opended()) break;
            await sleep(100);
        }

        console.log("[slowargo.js] Fast Forward Mode: Executing workflow...");
        app.queuePrompt(0);

        await waitForExecutionComplete();

        await sleep(500);

        console.log("[slowargo.js] Fast Forward Mode: Refreshing image...");
        const node = app.graph.getNodeById(targetNode.id);
        if (!node || !node.refreshImageList) {
            console.warn("[slowargo.js] Fast Forward Mode: target node lost or no refreshImageList");
            return;
        }

        fastForwardState.sourceNodeId = node.id;
        await node.refreshImageList(null, true);

        // MutationObserver will automatically call restoreColorAndAddToggle when editor opens
        console.log("[slowargo.js] Fast Forward Mode: cycle complete");

    } catch (error) {
        console.error("[slowargo.js] Fast Forward Mode: cycle failed", error);
    } finally {
        fastForwardState.active = false;
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

    // Add fast forward toggle button if not already added
    addFastForwardToggleButton();
}

function addFastForwardToggleButton() {
    // Check if toggle already exists
    if (document.querySelector(".fast-forward-mode-toggle")) {
        return;
    }

    const targetNode = getFastForwardTargetNode();
    if (!targetNode) {
        console.log("[slowargo.js] Fast Forward Mode: target node not found, skipping toggle");
        return;
    }

    fastForwardState.sourceNodeId = targetNode.id;

    // Find a button in the topbar
    const refBtn = document.querySelector("#global-mask-editor button:has(i.pi-check)");
    if (!refBtn) {
        console.warn("[slowargo.js] Fast Forward Mode: undo button not found");
        return;
    }

    // Create Reload Mask button
    const reloadBtn = document.createElement("button");
    reloadBtn.className = "reload-mask-button";
    reloadBtn.title = "Reload Mask (Ctrl+L): Load most recent clipspace content";
    const reloadIcon = document.createElement("i");
    reloadIcon.className = "pi pi-refresh";
    reloadBtn.appendChild(reloadIcon);
    const reloadText = document.createElement("span");
    reloadText.textContent = "Reload Mask";
    reloadBtn.appendChild(reloadText);
    reloadBtn.style.cssText = `
        background: rgba(30, 144, 255, 0.8);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        cursor: pointer;
        font-size: 12px;
        font-weight: bold;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 4px;
        hover: background rgba(30, 144, 255, 1);
    `;

    reloadBtn.addEventListener("click", async () => {
        await loadClipspaceToEditor();
    });

    // Create toggle button
    const toggleBtn = document.createElement("button");
    toggleBtn.className = "fast-forward-mode-toggle";
    // toggleBtn.title = "Fast Forward Mode: Press Enter to save and run prompt, and auto refresh after completion\n(CapsLock to toggle)";
    toggleBtn.title = "Fast Forward Mode: Press Enter to save and run prompt, and auto refresh after completion";
    const icon = document.createElement("i");
    icon.className = "pi pi-fast-forward";
    toggleBtn.appendChild(icon);
    const text = document.createElement("span");
    text.textContent = "FF";
    toggleBtn.appendChild(text);
    toggleBtn.style.cssText = `
        background: rgba(30, 144, 255, 0.5);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        cursor: pointer;
        font-size: 12px;
        font-weight: bold;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 4px;
    `;

    // Update style based on enabled state
    function updateToggleStyle() {
        if (fastForwardState.enabled) {
            toggleBtn.style.background = "rgba(30, 144, 255, 0.9)";
            toggleBtn.style.boxShadow = "0 0 8px rgba(30, 144, 255, 0.8)";
        } else {
            toggleBtn.style.background = "rgba(30, 144, 255, 0.4)";
            toggleBtn.style.boxShadow = "none";
        }
    }

    toggleBtn.addEventListener("click", () => {
        fastForwardState.enabled = !fastForwardState.enabled;
        updateToggleStyle();
        console.log("[slowargo.js] Fast Forward Mode:", fastForwardState.enabled ? "enabled" : "disabled");
    });

    // Create a container for buttons at top-left of canvas (avoiding sidebar)
    let buttonContainer = document.querySelector(".ff-mode-button-container");
    if (!buttonContainer) {
        buttonContainer = document.createElement("div");
        buttonContainer.className = "ff-mode-button-container";

        const canvasContainer = document.querySelector("#maskEditorCanvasContainer");
        if (canvasContainer && canvasContainer.parentNode) {
            canvasContainer.parentNode.style.position = "relative";

            buttonContainer.style.cssText = `
                position: absolute;
                top: 10px;
                left: 74px;
                display: flex;
                gap: 8px;
                z-index: 100;
                background: rgba(0, 0, 0, 0.3);
                padding: 8px;
                border-radius: 6px;
                pointer-events: auto;
            `;

            canvasContainer.parentNode.insertBefore(buttonContainer, canvasContainer);
        }
    }

    buttonContainer.appendChild(toggleBtn);
    buttonContainer.appendChild(reloadBtn);

    updateToggleStyle();
}

// === Fast Forward Mode Initialization ===
export function initFastForwardMode() {
    // Sync Fast Forward Mode with CapsLock state (CapsLock ON = Fast Forward ON, OFF = Fast Forward OFF)
    window.addEventListener('keydown', async function(e) {
        if (!ComfyApp.maskeditor_is_opended()) return;

        // const capsLockOn = e.getModifierState('CapsLock');
        const targetNode = getFastForwardTargetNode();

        // Sync CapsLock state with Fast Forward Mode enabled state
        // if (capsLockOn !== fastForwardState.enabled && targetNode) {
        //     fastForwardState.enabled = capsLockOn;
        //     const toggleBtn = document.querySelector(".fast-forward-mode-toggle");
        //     if (toggleBtn) {
        //         const style = toggleBtn.style;
        //         if (fastForwardState.enabled) {
        //             style.opacity = "1";
        //             style.boxShadow = "0 0 8px rgba(30, 144, 255, 0.8)";
        //         } else {
        //             style.opacity = "0.6";
        //             style.boxShadow = "none";
        //         }
        //     }
        //     console.log("[slowargo.js] Fast Forward Mode:", fastForwardState.enabled ? "enabled" : "disabled");
        // }

        if (!targetNode) return;

        // Ctrl+L loads clipspace content into current editor
        if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
            e.preventDefault();
            e.stopImmediatePropagation();
            console.log("[slowargo.js] Loading clipspace content...");
            await loadClipspaceToEditor();
            return;
        }

        // Enter executes fast forward cycle if enabled and mask is not empty
        if (e.key !== 'Enter') return;
        if (!fastForwardState.enabled) return;
        if (fastForwardState.active) return;

        if (!isMaskNonEmpty()) {
            console.log("[slowargo.js] Fast Forward Mode: mask is empty, skipping");
            return;
        }

        e.preventDefault();
        e.stopImmediatePropagation();

        console.log("[slowargo.js] Fast Forward Mode: starting cycle for node", targetNode.id);
        await executeFastForwardCycle(targetNode);
    }, true);

    // Monitor mask editor container to detect opening (handles all open methods)
    const observer = new MutationObserver(() => {
        //const maskEditorPanel = document.querySelector("div.maskEditor_sidePanel");
        const refBtn = document.querySelector("#global-mask-editor button:has(i.pi-check)");
        if (refBtn && !document.querySelector(".fast-forward-mode-toggle")) {
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
