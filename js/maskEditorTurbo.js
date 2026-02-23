import { ComfyApp } from "../../scripts/app.js";
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { loadCSS, sleep, getKeybindingStore, getMaskEditorStore, eventMatchesCommand, isMaskNonEmpty } from "./utils.js";
import {
    initBrushToolOverlay,
    updateCloneStyle,
    updateSmudgeStyle,
    createCloneButton,
    createSmudgeButton,
    handleBrushToolKeydown,
    handleBrushToolKeyup,
    initCloneToolEvents,
    initSmudgeToolEvents,
    cleanupAllBrushTools,
} from "./maskEditorBrushTools.js";

loadCSS(import.meta.url, "./maskEditorTurbo.css");

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
    // 打开 editor 后我把 clipspace_return_node 置空了，这里应该也用不上了
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

function cleanupFastForwardUI(dialog) {
    if (!dialog) return;

    // Remove button container from the dialog
    const buttonContainer = dialog.querySelector(".ff-mode-button-container");
    if (buttonContainer) {
        buttonContainer.innerHTML = '';
        buttonContainer.parentNode?.removeChild(buttonContainer);
    }

    // Clear dialog reference and dataset to aid garbage collection
    if (dialog.dataset.blurListenerAdded) {
        delete dialog.dataset.blurListenerAdded;
    }

    console.log("[slowargo.js] Cleaned up Fast Forward UI elements");
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

    // Create Clone Brush and Smudge Brush buttons
    const cloneBtn = createCloneButton();
    const smudgeBtn = createSmudgeButton();

    buttonContainer.appendChild(toggleBtn);
    buttonContainer.appendChild(reloadMaskOnlyBtn);
    buttonContainer.appendChild(reloadAllBtn);
    buttonContainer.appendChild(cloneBtn);
    buttonContainer.appendChild(smudgeBtn);
    buttonContainer.appendChild(blurBtn);

    updateToggleStyle();
    updateCloneStyle();
    updateSmudgeStyle();
}

// Keyboard event handlers for mask editor
function onMaskEditorKeyup(e) {
    if (!ComfyApp.maskeditor_is_opended()) return;

    // In blur mode, let all keyboard events pass through to main UI
    if (editorState.isBlurred) return;

    handleBrushToolKeyup(e);
}

async function onMaskEditorKeydown(e) {
    if (!ComfyApp.maskeditor_is_opended()) return;

    let targetNode = getFastForwardTargetNode();

    const isUndoRedo = (e.ctrlKey || e.metaKey) &&
        (e.key === 'z' || e.key === 'Z' || e.key === 'y' || e.key === 'Y');

    // 针对 blur 模式的特殊处理
    // 因为 editor 未关闭，src/composables/maskeditor/useKeyboard.ts 的 event handler 仍然有效，在主界面操作时默认会走
    //   editor 的键盘事件 handler，即空格、CTRL+Z、CTRL+Y 会被拦截
    // 仅仅阻止事件传播到 useKeyboard.ts 也不够完美。此时 maskeditor_is_opended, 因此主界面的 src/scripts/changeTracker.ts 也是
    //   不工作的，无法使用快捷键触发 undo/redo。暂时无解，但允许 default handler 至少让 textarea 的 undo/redo 可以工作
    if (editorState.isBlurred) {
        // 进入 blur 模式后可能选了其他非LoadRecentImagePlusV1节点，getFastForwardTargetNode() 会返回 null
        // 让 targetNode 有值避免下面 return，继续走后面的 Escape 键处理函数，退出 blur 模式
        targetNode = app.graph.getNodeById(editorState.sourceNodeId);

        // 阻止空格键继续传播到 useKeyboard.ts, 但 default handler 仍然允许执行，从而允许在主界面文本框输入空格
        if (e.key === ' ') {
            e.stopImmediatePropagation();
            return;
        }

        // Block undo/redo shortcuts in blur mode to prevent useKeyboard.ts from handling them
        // Allow default handler to enable undo/redo work in textarea in the main interface
        if (isUndoRedo) {
            // e.preventDefault();
            e.stopImmediatePropagation();
            return;
        }
    }

    if (!targetNode) return;

    // Toggle blur mode when user presses the keybinding for Comfy.MaskEditor.OpenMaskEditor command
    if (eventMatchesCommand(e, 'Comfy.MaskEditor.OpenMaskEditor')) {
        e.preventDefault();
        e.stopImmediatePropagation();
        toggleEditorBlur();
        return;
    }

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

    // Brush radius adjustment and Esc handling for brush tools
    if (handleBrushToolKeydown(e)) {
        return;
    }

    // Ctrl+L loads clipspace content into current editor
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        e.stopImmediatePropagation();
        console.log("[slowargo.js] Loading clipspace content...");
        await loadClipspaceToEditor();
        return;
    }

    // Non blur more. disable undo/redo shortcuts outside the mask editor
    if (isUndoRedo) {
        e.preventDefault(); // Prevent browser's default undo behavior (textarea undo)
        // Don't to this. It will break undoing in the mask editor
        //e.stopImmediatePropagation(); // Prevent other possible script handling.
        // console.log('[slowargo.js] preventDefault for Ctrl+Z ');
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
}

// === Fast Forward Mode Initialization ===
export function initFastForwardMode() {
    let initialized = false; // 标志位：是否已初始化当前 editor
    let dialogClickHandler = null;
    let currentDialog = null;

    // Note: Keyboard events are bound/unbound dynamically in onEditorReady and cleanup

    // Two-phase observer for detecting mask editor opening
    // Phase 1: Monitor body direct children for p-dialog-mask appearance/removal (cheap)
    // Phase 2: Once dialog found, monitor its subtree for side panel readiness (scoped)
    function onEditorReady(dialog) {
        initialized = true;
        currentDialog = dialog;
        restoreColorAndAddToggle();
        initBrushToolOverlay();
        initCloneToolEvents();
        initSmudgeToolEvents();

        // Bind keyboard events for editor
        window.addEventListener('keydown', onMaskEditorKeydown, true);
        window.addEventListener('keyup', onMaskEditorKeyup, true);

        // Add click listener to toggle blur state when clicking on blurred editor
        if (!dialog.dataset.blurListenerAdded) {
            dialogClickHandler = (e) => {
                if (editorState.isBlurred) {
                    const button = e.target.closest('button, input[type="color"], input[type="range"]');
                    if (!button) {
                        e.stopImmediatePropagation();
                        toggleEditorBlur();
                    }
                }
            };
            dialog.addEventListener('click', dialogClickHandler);
            dialog.dataset.blurListenerAdded = 'true';
        }
    }

    function waitForSidePanel(dialog) {
        // Check if already ready (synchronous fast path)
        const refPanel = dialog.querySelector("div.maskEditor_sidePanel input[type=color]");
        if (refPanel && !document.querySelector(".fast-forward-mode-toggle")) {
            onEditorReady(dialog);
            return;
        }

        // Not ready yet — watch dialog subtree until side panel renders
        const innerObserver = new MutationObserver(() => {
            const refPanel = dialog.querySelector("div.maskEditor_sidePanel input[type=color]");
            if (refPanel && !document.querySelector(".fast-forward-mode-toggle")) {
                innerObserver.disconnect();
                onEditorReady(dialog);
            }
        });
        innerObserver.observe(dialog, {
            childList: true,
            subtree: true
        });
    }

    // Phase 1: Only watch body direct children — triggers when p-dialog-mask is added/removed
    const observer = new MutationObserver(() => {
        const dialog = document.querySelector("body > .p-dialog-mask .mask-editor-dialog");

        if (!dialog && initialized) {
            // Editor closed — cleanup resources
            initialized = false;
            console.log("[slowargo.js] Mask editor closed, cleaning up resources");

            // Cleanup all brush tools (clone, smudge, overlay, canvas refs)
            cleanupAllBrushTools();

            // Remove keyboard event listeners
            window.removeEventListener('keydown', onMaskEditorKeydown, true);
            window.removeEventListener('keyup', onMaskEditorKeyup, true);

            // Remove click listener from dialog
            if (currentDialog && dialogClickHandler) {
                currentDialog.removeEventListener('click', dialogClickHandler);
                dialogClickHandler = null;
            }

            // Cleanup UI elements
            cleanupFastForwardUI(currentDialog);
            currentDialog.querySelectorAll('canvas').forEach(c => { c.width = 0; c.height = 0; c.parentNode?.removeChild(c);});
            currentDialog.querySelectorAll('img').forEach(c => { c.src = '';c.parentNode?.removeChild(c); })
            currentDialog.innerHTML = "";
            currentDialog = null;
            return;
        }

        if (dialog && !initialized) {
            // Editor opened — start phase 2 to wait for side panel
            waitForSidePanel(dialog);
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

// Export for use in slowargo.js
export { performMaskSave, toggleEditorBlur };
