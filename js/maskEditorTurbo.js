import { ComfyApp } from "../../scripts/app.js";
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Load CSS dynamically
const link = document.createElement("link");
link.rel = "stylesheet";
link.href = new URL("./maskEditorTurbo.css", import.meta.url).href;
document.head.appendChild(link);

// === Editor State ===
const editorState = {
    // Fast Forward Mode
    cycleInProgress: false, // cycle 是否在进行中（防止重入）
    fastForwardModeOn: true, // Fast Forward Mode 是否启用（通过 toggle 控制）
    sourceNodeId: null,     // 发起 Fast Forward 的节点 ID

    // Editor Blur
    isBlurred: false,       // 编辑器是否模糊化（默认清晰，可按 Esc 切到模糊态）
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

// === Access Pinia Store for GPU Sync ===
function getMaskEditorStore() {
    try {
        const vueApp = document.querySelector('#vue-app')?.__vue_app__;
        if (!vueApp) return null;
        const pinia = vueApp.config.globalProperties?.$pinia;
        if (!pinia?._s) return null;
        return pinia._s.get('maskEditor') || null;
    } catch (e) {
        return null;
    }
}

// === Load Clipspace Content to Current Editor ===
async function loadClipspaceToEditor(reloadMaskOnly = false) {
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

        console.log("[slowargo.js] Loading clipspace, timestamp:", timestamp, "maskOnly:", reloadMaskOnly);

        // Click Clear button to reset mask and GPU state
        // const clearBtn = document.querySelector("#global-mask-editor > div.flex.items-center > div > button:nth-child(4)");
        // if (clearBtn) {
        //     clearBtn.click();
        //     await new Promise(resolve => setTimeout(resolve, 200));
        // } else {
        //     console.warn("[slowargo.js] Clear button not found");
        // }

        const params = `${app.getPreviewFormatParam?.() || ""}${app.getRandParam?.() || ""}`;
        const loadImg = (url) => new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });

        // Load mask layer
        const maskImg = await loadImg(api.apiURL(`/view?filename=clipspace-mask-${timestamp}.png&subfolder=clipspace&type=input&channel=a${params}`));
        const maskCtx = canvases[2].getContext('2d', {willReadFrequently: true});
        maskCtx.clearRect(0, 0, canvases[2].width, canvases[2].height);
        maskCtx.drawImage(maskImg, 0, 0, canvases[2].width, canvases[2].height);
        maskImg.src = '';

        // Invert alpha channel
        const maskData = maskCtx.getImageData(0, 0, canvases[2].width, canvases[2].height);
        for (let i = 3; i < maskData.data.length; i += 4) {
            maskData.data[i] = 255 - maskData.data[i];
        }
        maskCtx.putImageData(maskData, 0, 0);

        // Load base layer if not mask-only
        if (!reloadMaskOnly) {
            const baseImg = await loadImg(api.apiURL(`/view?filename=clipspace-mask-${timestamp}.png&subfolder=clipspace&type=input&channel=rgb${params}`));
            const baseCtx = canvases[0].getContext('2d', {willReadFrequently: true});
            baseCtx.clearRect(0, 0, canvases[0].width, canvases[0].height);
            baseCtx.drawImage(baseImg, 0, 0, canvases[0].width, canvases[0].height);
            baseImg.src = '';

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
        }

        // Sync GPU textures via Pinia store's canvasHistory.
        // saveState() increments currentStateIndex, which triggers the watch
        // in useBrushDrawing.ts that calls updateGPUFromCanvas().
        const store = getMaskEditorStore();
        if (store?.canvasHistory?.saveState) {
            store.canvasHistory.saveState();
            console.log("[slowargo.js] GPU sync triggered via canvasHistory.saveState()");
        } else {
            console.warn("[slowargo.js] Pinia store not accessible, GPU textures may be stale");
        }

        console.log("[slowargo.js] Clipspace loaded successfully");
        return data.image_name[0]; // Return clipspace filename for skip-save optimization
    } catch (error) {
        console.error("[slowargo.js] Failed to load clipspace:", error);
        return null;
    }
}
// === Fast Forward Mode Helper Functions ===

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// === Editor Blur Toggle ===
function toggleEditorBlur() {
    editorState.isBlurred = !editorState.isBlurred;
    const editor = document.querySelector(".mask-editor-dialog");
    const mask = document.querySelector(".p-dialog-mask");

    if (editor) {
        if (editorState.isBlurred) {
            editor.classList.add("editor-blurred");
            // Hide mask overlay
            if (mask) mask.classList.add("editor-blurred-mask");
        } else {
            editor.classList.remove("editor-blurred");
            // Show mask overlay
            if (mask) mask.classList.remove("editor-blurred-mask");

            // Restore selected node when returning from blur
            if (editorState.sourceNodeId) {
                const node = app.graph.getNodeById(editorState.sourceNodeId);
                if (node) {
                    app.canvas.selectNodes([node]);
                    console.log("[slowargo.js] Restored selected node:", node.id);
                }
            }
        }
    }
    console.log("[slowargo.js] Editor blur toggled:", editorState.isBlurred ? "blurred" : "focused");
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

async function executeFastForwardCycle(targetNode, skipSaveWithClipspace = null) {
    editorState.cycleInProgress = true;

    try {
        if (skipSaveWithClipspace) {
            // Reload All + Shift: clipspace files already on disk, skip redundant save.
            // Just save color, update widget to existing clipspace file, and close editor.
            console.log("[slowargo.js] Fast Forward Mode: Skipping save, using existing clipspace:", skipSaveWithClipspace);
            const colorInput = document.querySelector("div.maskEditor_sidePanel input[type=color]");
            if (colorInput?.value) colorMemory.save(colorInput.value);

            const imageWidget = targetNode.widgets?.find(w => w.name === 'image');
            if (imageWidget) {
                imageWidget.value = skipSaveWithClipspace;
            }

            // Close editor without saving (click Cancel button)
            const cancelBtn = document.querySelector("#global-mask-editor button:has(i.pi-times)");
            if (cancelBtn) cancelBtn.click();
        } else {
            console.log("[slowargo.js] Fast Forward Mode: Saving mask...");
            await performMaskSave();
        }

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

        editorState.sourceNodeId = node.id;
        await node.refreshImageList(null, true);

        // MutationObserver will automatically call restoreColorAndAddToggle when editor opens
        console.log("[slowargo.js] Fast Forward Mode: cycle complete");

    } catch (error) {
        console.error("[slowargo.js] Fast Forward Mode: cycle failed", error);
    } finally {
        editorState.cycleInProgress = false;
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

    const targetNode = getFastForwardTargetNode();
    if (!targetNode) {
        console.log("[slowargo.js] Fast Forward Mode: target node not found, skipping toggle");
        return;
    }

    editorState.sourceNodeId = targetNode.id;

    // Add Fast Forward toggle button if not already added
    addFastForwardToggleButton();

    // === 最大化 mask editor dialog ===
    const maximizeBtn = document.querySelector("div.mask-editor-dialog button.p-dialog-maximize-button");
    if (maximizeBtn) {
        // maxHeight isn't available yet at this moment
        // const dialogMask = document.querySelector("div.p-dialog-mask")
        // const maxHeight = window.getComputedStyle(dialogMask).maxHeight;
        maximizeBtn.click();
        // console.log("[slowargo.js] Mask editor dialog maximized");
    }

    // Reset blur state on editor open
    editorState.isBlurred = false;

    // Apply blur state to editor
    // const editor = document.querySelector(".mask-editor-dialog");
    // const mask = document.querySelector(".p-dialog-mask");
    // if (editor) {
    //     if (editorState.isBlurred) {
    //         editor.classList.add("editor-blurred");
    //         if (mask) mask.classList.add("editor-blurred-mask");
    //     } else {
    //         editor.classList.remove("editor-blurred");
    //         if (mask) mask.classList.remove("editor-blurred-mask");
    //     }
    // }

}

function addFastForwardToggleButton() {
    // Check if toggle already exists
    if (document.querySelector(".fast-forward-mode-toggle")) {
        return;
    }

    // Find a button in the topbar
    const refBtn = document.querySelector("#global-mask-editor button:has(i.pi-check)");
    if (!refBtn) {
        console.warn("[slowargo.js] Fast Forward Mode: undo button not found");
        return;
    }

    // Create Reload Mask Layer Only button
    const reloadMaskOnlyBtn = document.createElement("button");
    reloadMaskOnlyBtn.className = "reload-mask-button";
    reloadMaskOnlyBtn.title = "Restore Mask Layer Only: Load mask layer from most recent clipspace";
    const reloadMaskOnlyIcon = document.createElement("i");
    reloadMaskOnlyIcon.className = "pi pi-refresh";
    reloadMaskOnlyBtn.appendChild(reloadMaskOnlyIcon);
    const reloadMaskOnlyText = document.createElement("span");
    reloadMaskOnlyText.textContent = "Mask";
    reloadMaskOnlyBtn.appendChild(reloadMaskOnlyText);

    reloadMaskOnlyBtn.addEventListener("click", async (e) => {
        await loadClipspaceToEditor(true);
        if (e.shiftKey) {
            const targetNode = app.graph.getNodeById(editorState.sourceNodeId);//getFastForwardTargetNode();
            if (targetNode && editorState.fastForwardModeOn) {
                console.log("[slowargo.js] Shift+Reload triggered, executing Fast Forward cycle");
                await executeFastForwardCycle(targetNode);
            }
        }
    });

    // Create Reload All Layers button
    const reloadAllBtn = document.createElement("button");
    reloadAllBtn.className = "reload-mask-button";
    reloadAllBtn.title = "Restore All Layers (Ctrl+L): Load base, mask, and paint layers from most recent clipspace";
    const reloadAllIcon = document.createElement("i");
    reloadAllIcon.className = "pi pi-refresh";
    reloadAllBtn.appendChild(reloadAllIcon);
    const reloadAllText = document.createElement("span");
    reloadAllText.textContent = "All";
    reloadAllBtn.appendChild(reloadAllText);

    reloadAllBtn.addEventListener("click", async (e) => {
        const clipspaceFilename = await loadClipspaceToEditor(false);
        if (e.shiftKey) {
            const targetNode = app.graph.getNodeById(editorState.sourceNodeId);//getFastForwardTargetNode();
            if (targetNode && editorState.fastForwardModeOn) {
                console.log("[slowargo.js] Shift+Reload All triggered, executing Fast Forward cycle (skip save)");
                await executeFastForwardCycle(targetNode, clipspaceFilename);
            }
        }
    });

    // Create toggle button
    const toggleBtn = document.createElement("button");
    toggleBtn.className = "fast-forward-mode-toggle";
    toggleBtn.title = "Fast Forward Mode (Click to toggle): Press Enter to save and run prompt, and auto refresh after completion";
    const icon = document.createElement("i");
    icon.className = "pi pi-fast-forward";
    toggleBtn.appendChild(icon);
    const text = document.createElement("span");
    text.textContent = "FF";
    toggleBtn.appendChild(text);

    // Update style based on enabled state
    function updateToggleStyle() {
        if (editorState.fastForwardModeOn) {
            toggleBtn.classList.add("enabled");
            toggleBtn.classList.remove("disabled");
        } else {
            toggleBtn.classList.add("disabled");
            toggleBtn.classList.remove("enabled");
        }
    }

    toggleBtn.addEventListener("click", () => {
        editorState.fastForwardModeOn = !editorState.fastForwardModeOn;
        updateToggleStyle();
        console.log("[slowargo.js] Fast Forward Mode:", editorState.fastForwardModeOn ? "enabled" : "disabled");
    });

    // Create a container for buttons at top-left of canvas (avoiding sidebar)
    let buttonContainer = document.querySelector(".ff-mode-button-container");
    if (!buttonContainer) {
        buttonContainer = document.createElement("div");
        buttonContainer.className = "ff-mode-button-container";

        const canvasContainer = document.querySelector("#maskEditorCanvasContainer");
        if (canvasContainer && canvasContainer.parentNode) {
            canvasContainer.parentNode.classList.add("ff-mode-canvas-container-parent");
            canvasContainer.parentNode.insertBefore(buttonContainer, canvasContainer);
        }
    }

    // Create Blur toggle button (clear → blur)
    const blurBtn = document.createElement("button");
    blurBtn.className = "reload-mask-button";
    blurBtn.title = "Blur Mode: Minimize editor to left side and access main interface";
    const blurIcon = document.createElement("i");
    blurIcon.className = "pi pi-eye-slash";
    blurBtn.appendChild(blurIcon);

    blurBtn.addEventListener("click", () => {
        if (!editorState.isBlurred) {
            toggleEditorBlur();
        }
    });

    buttonContainer.appendChild(toggleBtn);
    buttonContainer.appendChild(reloadMaskOnlyBtn);
    buttonContainer.appendChild(reloadAllBtn);
    buttonContainer.appendChild(blurBtn);

    updateToggleStyle();
}

// === Fast Forward Mode Initialization ===
export function initFastForwardMode() {
    // Sync Fast Forward Mode with CapsLock state (CapsLock ON = Fast Forward ON, OFF = Fast Forward OFF)
    window.addEventListener('keydown', async function(e) {
        if (!ComfyApp.maskeditor_is_opended()) return;

        // const capsLockOn = e.getModifierState('CapsLock');
        let targetNode = getFastForwardTargetNode();

        // Sync CapsLock state with Fast Forward Mode enabled state
        // if (capsLockOn !== editorState.fastForwardModeOn && targetNode) {
        //     editorState.fastForwardModeOn = capsLockOn;
        //     const toggleBtn = document.querySelector(".fast-forward-mode-toggle");
        //     if (toggleBtn) {
        //         const style = toggleBtn.style;
        //         if (editorState.fastForwardModeOn) {
        //             style.opacity = "1";
        //             style.boxShadow = "0 0 8px rgba(30, 144, 255, 0.8)";
        //         } else {
        //             style.opacity = "0.6";
        //             style.boxShadow = "none";
        //         }
        //     }
        //     console.log("[slowargo.js] Fast Forward Mode:", editorState.fastForwardModeOn ? "enabled" : "disabled");
        // }

        if (editorState.isBlurred) {
            // 进入 blur 模式后可能选了其他节点，让 targetNode 有值走后面的 Escape 键处理函数，退出 blur 模式
            targetNode = app.graph.getNodeById(editorState.sourceNodeId);
        }

        if (!targetNode) return;

        // Esc key toggles blur mode (intercept before blur mode pass-through)
        if (e.key === 'Escape') {
            // If mask is empty, let dialog close naturally
            if (!isMaskNonEmpty()) {
                return;
            }
            // Mask is non-empty: toggle blur state and prevent dialog from closing
            e.preventDefault();
            e.stopImmediatePropagation();
            toggleEditorBlur();
            return;
        }

        // In blur mode, let all keyboard events pass through to main UI
        if (editorState.isBlurred) return;

        // Ctrl+L loads clipspace content into current editor
        if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
            e.preventDefault();
            e.stopImmediatePropagation();
            console.log("[slowargo.js] Loading clipspace content...");
            await loadClipspaceToEditor();
            return;
        }

        // Enter executes Fast Forward cycle if enabled and mask is not empty
        if (e.key !== 'Enter') return;
        if (!editorState.fastForwardModeOn) return;
        if (editorState.cycleInProgress) return;

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
        // const maskEditorPanel = document.querySelector("div.maskEditor_sidePanel");
        // const refBtn = document.querySelector("#global-mask-editor button:has(i.pi-check)");
        const refBtn = document.querySelector("div.mask-editor-dialog button.p-dialog-maximize-button");
        if (refBtn && !document.querySelector(".fast-forward-mode-toggle")) {
            // Mask editor just opened, add toggle and restore color
            // setTimeout(() => {
            //     restoreColorAndAddToggle();
            // }, 100);

            // restoreColorAndAddToggle会最大化对话框，临时断开 observer，执行完操作后再重新连接
            // observer.disconnect();
            restoreColorAndAddToggle();
            // observer.observe(document.body, {
            //     childList: true,
            //     subtree: true,
            //     attributes: false,
            //     characterData: false
            // });

            // Add click listener to toggle blur state when clicking on blurred editor
            const editor = document.querySelector(".mask-editor-dialog");
            if (editor && !editor.dataset.blurListenerAdded) {
                editor.addEventListener('click', (e) => {
                    if (editorState.isBlurred) {
                        // Check if clicked on button or control - don't toggle
                        const button = e.target.closest('button, input[type="color"], input[type="range"]');
                        if (!button) {
                            e.stopImmediatePropagation();
                            toggleEditorBlur();
                        }
                    }
                });
                editor.dataset.blurListenerAdded = 'true';
            }
        }
    });

    // Start observing document for changes
    observer.observe(document.body, {
        childList: true,
        subtree: false,
        attributes: false,
        characterData: false
    });
}

// Export performMaskSave for use in maskeditor.save command
export { performMaskSave, toggleEditorBlur };
