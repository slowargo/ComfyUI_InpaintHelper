import { ComfyApp } from "../../scripts/app.js";
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// === Turbo Mode State ===
const turboState = {
    active: false,         // 防止重入
    sourceNodeId: null,    // 发起 turbo 的节点 ID
};

// === Turbo Mode Helper Functions ===

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function getTurboTargetNode() {
    // 使用 open_maskeditor 打开 mask editor 时
    let selectedNode = ComfyApp.clipspace_return_node;
    if (!selectedNode) {
        // 使用 commandStore.execute('Comfy.MaskEditor.OpenMaskEditor') 打开 mask editor 时
        const selectedNodes = app.canvas.selected_nodes;
        if (selectedNodes && Object.keys(selectedNodes).length === 1) {
            selectedNode = Object.values(selectedNodes)[0];
            console.log("[slowargo.js] Turbo: using selected node from canvas");
        }
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
    turboState.sourceNodeId = targetNode.id;

    try {
        console.log("[slowargo.js] Turbo: Saving mask...");
        await performMaskSave();

        await sleep(300);

        console.log("[slowargo.js] Turbo: Executing workflow...");
        app.queuePrompt(0);

        await waitForExecutionComplete();

        await sleep(500);

        console.log("[slowargo.js] Turbo: Refreshing image...");
        const node = app.graph.getNodeById(turboState.sourceNodeId);
        if (!node || !node.refreshFn) {
            console.warn("[slowargo.js] Turbo Mode: target node lost or no refreshFn");
            return;
        }
        await node.refreshFn();

        await sleep(300);

        console.log("[slowargo.js] Turbo: Opening editor...");
        ComfyApp.clipspace_return_node = node;
        ComfyApp.open_maskeditor?.();

        console.log("[slowargo.js] Turbo Mode: cycle complete");

    } catch (error) {
        console.error("[slowargo.js] Turbo Mode: cycle failed", error);
    } finally {
        turboState.active = false;
    }
}

// === Turbo Mode Initialization ===

export function initTurboMode() {
    // Track CapsLock state and handle Turbo Mode Enter key
    window.addEventListener('keydown', async function(e) {
        if (e.key !== 'Enter') return;
        if (!ComfyApp.maskeditor_is_opended()) return;
        if (!e.getModifierState('CapsLock')) return;
        if (turboState.active) return;

        const targetNode = getTurboTargetNode();
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
}

// Export performMaskSave for use in maskeditor.save command
export { performMaskSave };
