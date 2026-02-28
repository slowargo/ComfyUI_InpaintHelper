import { getMaskEditorStore, displayToCanvas, getBrushOpacity, setBrushOpacity } from "../utils.js";
import {
    cleanupTransform,
    isTransformActive,
    initTransformToolEvents,
} from "./transform/maskEditorBrushToolsTransform.js";
import { ensurePaintLayerVisible, saveCanvasHistory, getMaskEditorCanvasContainer } from "./common/helpers.js";
import { setSharedOverlay, setSharedCanvases } from "./common/sharedCanvasRefs.js";
import { getLerpPoint, getStrokeInterpolation } from "./math.js";
import { sampleComposite, stampClone, stampSmudge } from "./layer.js";

// === Editor State ===
const editorState = {
    // Clone Brush
    cloneBrush: {
        active: false,
        hasSample: false,
        sampleX: 0, sampleY: 0,
        strokeStartX: 0, strokeStartY: 0,
        lastDrawX: 0, lastDrawY: 0,
        isDrawing: false,
        eventsBound: false,
    },

    // Smudge Brush
    smudgeBrush: {
        active: false,
        isDrawing: false,
        lastDrawX: 0, lastDrawY: 0,
        carriedBuffer: null,      // Float32Array carrying sampled pixels
        eventsBound: false,
    }
};

// Space key state for allowing mask editor pan functionality
let isSpacePressed = false;

// === Module-level Shared Tool Resources ===
let brushToolOverlay = null;
let baseCanvas = null;
let paintCanvas = null;

// === Opacity Override State (for Clone/Smudge/Transform tools) ===
const opacityOverrideState = {
    originalOpacity: null,
    isOverridden: false
};

/**
 * Activate opacity override: save current opacity and set to 1 (fully opaque).
 * Safe to call multiple times - only saves on first call.
 */
function activateOpacityOverride() {
    if (!opacityOverrideState.isOverridden) {
        opacityOverrideState.originalOpacity = getBrushOpacity();
        opacityOverrideState.isOverridden = true;
        setBrushOpacity(1);
    }
}

/**
 * Restore original opacity if currently overridden.
 * Safe to call multiple times - only restores when actually overridden.
 */
function restoreOpacityOverride() {
    if (opacityOverrideState.isOverridden) {
        setBrushOpacity(opacityOverrideState.originalOpacity);
        opacityOverrideState.originalOpacity = null;
        opacityOverrideState.isOverridden = false;
    }
}

// === Brush Tool Functions ===

let globalBrushRadius = 20;
let lastBrushReadTime = 0;

/**
 * Get the current brush radius.
 * Prefers the Pinia store value (read every time), then falls back to the DOM size slider (cached for 500ms).
 * @returns {number} The current brush radius in pixels
 */
function getBrushRadius() {
    const now = Date.now();

    // 1) Always read from store (source of truth, no caching)
    const store = getMaskEditorStore();
    const storeSize = store?.brushSettings?.size;
    if (Number.isFinite(storeSize)) {
        globalBrushRadius = storeSize;
        return globalBrushRadius;
    }

    // 2) Fallback to DOM: use cache to avoid excessive DOM reads
    if (now - lastBrushReadTime < 500) {
        return globalBrushRadius;
    }

    const rangeInputs = document.querySelectorAll('input.maskEditor_sidePanelBrushRange');
    if (rangeInputs.length > 0) {
        const sizeInput = Array.from(rangeInputs).find((input) => input.max === '500') || rangeInputs[0];
        const parsed = parseFloat(sizeInput.value);
        if (Number.isFinite(parsed)) {
            globalBrushRadius = parsed;
        }
    }

    lastBrushReadTime = now;
    return globalBrushRadius;
}

/**
 * Initialize the shared brush tool overlay canvas inside #maskEditorCanvasContainer.
 * The overlay serves three roles:
 * 1. Event capture layer: intercepts all pointer events when Clone or Smudge is active,
 *    preventing them from reaching the mask editor's native drawing tools.
 * 2. Visual feedback: renders crosshairs, dashed connection lines, and brush circles.
 * 3. Coordinate reference: used as the basis for display-to-canvas coordinate mapping.
 *
 * 初始化 BrushToolOverlay。Overlay 是一个共享的覆盖层画布，为 Clone Brush 和 Smudge Brush 提供两个核心功能：
 * 1. 事件捕获层: 当 Clone 或 Smudge 工具激活时，覆盖层捕获所有鼠标/指针事件，阻止事件传递到 mask editor 的原生绘制工具。
 * 2. 视觉反馈绘制: 十字准星、虚线连接、笔刷圆圈等
 * 3. 坐标映射基准: 覆盖层作为坐标转换的参考对象
 */
function initBrushToolOverlay() {
    const container = getMaskEditorCanvasContainer();
    if (!container) return;

    if (getComputedStyle(container).position === 'static') {
        container.style.position = 'relative';
    }

    const canvases = container.querySelectorAll('canvas');
    if (canvases.length < 2) return;

    baseCanvas  = canvases[0];
    paintCanvas = canvases[1];
    setSharedCanvases(baseCanvas, paintCanvas);

    const existingOverlay = container.querySelector('#brush-tool-overlay');
    if (existingOverlay) {
        brushToolOverlay = existingOverlay;
        setSharedOverlay(existingOverlay);
        return;
    }

    const overlay = document.createElement('canvas');
    overlay.id = 'brush-tool-overlay';
    overlay.width  = canvases[0].width;
    overlay.height = canvases[0].height;
    // Match other canvas elements' CSS positioning to fill container properly
    overlay.className = 'absolute top-0 left-0 w-full h-full';
    overlay.style.zIndex = '50'; // Above other canvases (z-40 is highest native)
    container.appendChild(overlay);
    brushToolOverlay = overlay;
    setSharedOverlay(overlay);
    // console.log("[slowargo.js] Brush tool overlay initialized");
}

/**
 * Check whether a pointer event falls within the brush tool overlay bounds.
 * Used as a guard to ignore events outside the canvas area.
 *
 * Pointer 事件的护栏。
 *
 * @param {PointerEvent} e - The pointer event to check
 * @returns {boolean} true if the pointer is within the overlay bounds
 */
function isPointerInBrushArea(e) {
    if (!brushToolOverlay) return false;
    const rect = brushToolOverlay.getBoundingClientRect();
    return e.clientX >= rect.left && e.clientX <= rect.right &&
        e.clientY >= rect.top  && e.clientY <= rect.bottom;
}


// Module-level button references for style updates
let cloneBtnRef = null;
let smudgeBtnRef = null;

/**
 * Sync the clone brush button's CSS classes to reflect the current active state.
 */
function updateCloneStyle() {
    if (!cloneBtnRef) return;
    cloneBtnRef.classList.toggle('enabled',  editorState.cloneBrush.active);
    cloneBtnRef.classList.toggle('disabled', !editorState.cloneBrush.active);
}

/**
 * Sync the smudge brush button's CSS classes to reflect the current active state.
 */
function updateSmudgeStyle() {
    if (!smudgeBtnRef) return;
    smudgeBtnRef.classList.toggle('enabled',  editorState.smudgeBrush.active);
    smudgeBtnRef.classList.toggle('disabled', !editorState.smudgeBrush.active);
}

/**
 * Deactivate all custom brush tools and clear the overlay canvas.
 * Resets active/hasSample state for both Clone and Smudge tools.
 */
function deactivateAllCustomTools() {
    if (editorState.cloneBrush.active) {
        editorState.cloneBrush.active = false;
        editorState.cloneBrush.hasSample = false;
        updateCloneStyle();
    }
    if (editorState.smudgeBrush.active) {
        editorState.smudgeBrush.active = false;
        updateSmudgeStyle();
    }
    // Deactivate Transform tool
    if (isTransformActive()) {
        cleanupTransform();
        updateTransformStyle();
    }
    if (brushToolOverlay) {
        const ctx = brushToolOverlay.getContext('2d');
        ctx?.clearRect(0, 0, brushToolOverlay.width, brushToolOverlay.height);
    }

    // Restore original opacity if overridden
    restoreOpacityOverride();
}

/**
 * Check whether any custom brush tool (Clone or Smudge) is currently active.
 * @returns {boolean} true if at least one custom tool is active
 */
function isAnyCustomToolActive() {
    return editorState.cloneBrush.active || editorState.smudgeBrush.active || isTransformActive();
}

// === Clone Brush Functions ===

/**
 * Bind document-level capture pointer events for the Clone Brush tool.
 * Idempotent: does nothing if events are already bound.
 */
function initCloneToolEvents() {
    if (editorState.cloneBrush.eventsBound) return;
    document.addEventListener('pointerdown', onCloneMouseDown, true);
    document.addEventListener('pointermove', onCloneMouseMove, true);
    document.addEventListener('pointerup',   onCloneMouseUp,   true);
    editorState.cloneBrush.eventsBound = true;
    console.log("[slowargo.js] Clone tool events bound to document (capture)");
}

/**
 * Handle pointerdown for the Clone Brush.
 * Alt+click sets the clone source point; plain click starts a clone stroke.
 * @param {PointerEvent} e
 */
function onCloneMouseDown(e) {
    if (!editorState.cloneBrush.active) return;
    if (!isPointerInBrushArea(e)) return;

    // Allow mask editor pan when space is held
    if (isSpacePressed) return;

    e.stopImmediatePropagation();
    e.preventDefault();

    const { x: cx, y: cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

    if (e.altKey || !editorState.cloneBrush.hasSample) {
        editorState.cloneBrush.hasSample  = true;
        editorState.cloneBrush.sampleX    = cx;
        editorState.cloneBrush.sampleY    = cy;
        renderCloneOverlay(cx, cy);
        return;
    }

    if (!editorState.cloneBrush.hasSample) return;

    editorState.cloneBrush.isDrawing    = true;
    editorState.cloneBrush.strokeStartX = cx;
    editorState.cloneBrush.strokeStartY = cy;
    editorState.cloneBrush.lastDrawX    = cx;
    editorState.cloneBrush.lastDrawY    = cy;
    applyCloneStroke(cx, cy);
}

/**
 * Handle pointermove for the Clone Brush.
 * Applies clone strokes while drawing, and always updates the overlay cursor.
 * @param {PointerEvent} e
 */
function onCloneMouseMove(e) {
    if (!editorState.cloneBrush.active) return;
    if (!isPointerInBrushArea(e)) return;

    // Allow mask editor pan when space is held
    if (isSpacePressed) {
        renderCloneOverlay(0, 0); // Clear overlay
        return;
    }

    const { x: cx, y: cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

    if (editorState.cloneBrush.isDrawing) {
        e.stopImmediatePropagation();
        applyCloneStroke(cx, cy);
    }
    renderCloneOverlay(cx, cy);
}

/**
 * Handle pointerup for the Clone Brush.
 * Ends the active stroke and saves canvas history.
 * @param {PointerEvent} e
 */
function onCloneMouseUp(e) {
    if (!editorState.cloneBrush.active || !editorState.cloneBrush.isDrawing) return;
    e.stopImmediatePropagation();
    editorState.cloneBrush.isDrawing = false;

    // When CapsLock is on or Shift is held, sample point moves with the mouse (preserving relative offset)
    // if (e.getModifierState('CapsLock') || e.shiftKey) {
    //     const { x: cx, y: cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);
    //     const trackedX = editorState.cloneBrush.sampleX + (cx - editorState.cloneBrush.strokeStartX);
    //     const trackedY = editorState.cloneBrush.sampleY + (cy - editorState.cloneBrush.strokeStartY);
    //
    //     editorState.cloneBrush.sampleX = trackedX;
    //     editorState.cloneBrush.sampleY = trackedY;
    // }

    saveCanvasHistory();
}

/**
 * Apply a clone stroke from the last draw position to (cx, cy) using linear interpolation.
 * Interpolation fills gaps that would appear when the mouse moves faster than the step size,
 * where step = max(1, radius / 3) ensures enough stamp overlap.
 *
 * Clone Brush 的笔触插值函数，解决快速移动鼠标时绘制不连续的问题
 *
 * @param {number} cx - Target canvas X coordinate
 * @param {number} cy - Target canvas Y coordinate
 */
function applyCloneStroke(cx, cy) {
    if (!baseCanvas || !paintCanvas) return;

    const radius = getBrushRadius();
    const opacity = getBrushOpacity();
    // 步长与笔刷半径成正比, 确保至少为 1 像素, radius/3 保证相邻 stamp 有足够重叠
    const step = Math.max(1, radius / 3);
    // 两点间直线距离与插值步数
    const { steps } = getStrokeInterpolation(
        editorState.cloneBrush.lastDrawX,
        editorState.cloneBrush.lastDrawY,
        cx,
        cy,
        step
    );

    // 线性插值填充间隙
    if (steps === 0) {
        // 没有移动距离时至少绘制一个点（仅 mouse down 的情况）
        stampClone(baseCanvas, paintCanvas, editorState.cloneBrush, cx, cy, radius, opacity);
    } else {
        for (let i = 1; i <= steps; i++) {
            const t  = i / steps; // 0.0 ~ 1.0 的插值因子
            // 在 lastDraw 和 current 之间均匀分布 stamp
            const point = getLerpPoint(
                editorState.cloneBrush.lastDrawX,
                editorState.cloneBrush.lastDrawY,
                cx,
                cy,
                t
            );
            // 在插值点执行克隆
            stampClone(
                baseCanvas,
                paintCanvas,
                editorState.cloneBrush,
                point.x,
                point.y,
                radius,
                opacity
            );
        }
    }

    editorState.cloneBrush.lastDrawX = cx;
    editorState.cloneBrush.lastDrawY = cy;
}

/**
 * Render the Clone Brush overlay at the current mouse position.
 * Draws: source crosshair, tracked source crosshair + dashed line while drawing,
 * and a brush circle at the cursor.
 * @param {number} mouseX - Current cursor X in canvas pixels
 * @param {number} mouseY - Current cursor Y in canvas pixels
 */
function renderCloneOverlay(mouseX, mouseY) {
    if (!brushToolOverlay || !editorState.cloneBrush.hasSample) return;

    const ctx = brushToolOverlay.getContext('2d');
    ctx.clearRect(0, 0, brushToolOverlay.width, brushToolOverlay.height);

    const radius = getBrushRadius();
    drawCrosshair(ctx, editorState.cloneBrush.sampleX, editorState.cloneBrush.sampleY, '#ff6666', radius);

    if (editorState.cloneBrush.isDrawing) {
        const trackedX = editorState.cloneBrush.sampleX + (mouseX - editorState.cloneBrush.strokeStartX);
        const trackedY = editorState.cloneBrush.sampleY + (mouseY - editorState.cloneBrush.strokeStartY);
        drawCrosshair(ctx, trackedX, trackedY, 'rgba(255, 102, 102, 0.7)', radius);

        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(trackedX, trackedY);
        ctx.lineTo(mouseX, mouseY);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    ctx.beginPath();
    ctx.arc(mouseX, mouseY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth = 1;
    ctx.stroke();
}

/**
 * Draw a crosshair (plus sign) on the given canvas context.
 * @param {CanvasRenderingContext2D} ctx - The 2D rendering context
 * @param {number} x - Center X in canvas pixels
 * @param {number} y - Center Y in canvas pixels
 * @param {string} color - CSS color string
 * @param {number} [size=8] - Half-length of each crosshair arm in pixels
 */
function drawCrosshair(ctx, x, y, color, size = 8) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - size, y); ctx.lineTo(x + size, y);
    ctx.moveTo(x, y - size); ctx.lineTo(x, y + size);
    ctx.stroke();
}

/**
 * Remove Clone Brush event listeners, save any in-progress stroke, and reset tool state.
 */
function cleanupCloneTool() {
    if (editorState.cloneBrush.isDrawing) {
        saveCanvasHistory();
        editorState.cloneBrush.isDrawing = false;
    }

    if (editorState.cloneBrush.eventsBound) {
        document.removeEventListener('pointerdown', onCloneMouseDown, true);
        document.removeEventListener('pointermove', onCloneMouseMove, true);
        document.removeEventListener('pointerup',   onCloneMouseUp,   true);
        editorState.cloneBrush.eventsBound = false;
    }

    editorState.cloneBrush.active     = false;
    editorState.cloneBrush.hasSample  = false;
    editorState.cloneBrush.isDrawing  = false;
}

// === Smudge Brush Functions ===

/**
 * Bind document-level capture pointer events for the Smudge Brush tool.
 * Idempotent: does nothing if events are already bound.
 */
function initSmudgeToolEvents() {
    if (editorState.smudgeBrush.eventsBound) return;
    document.addEventListener('pointerdown', onSmudgeMouseDown, true);
    document.addEventListener('pointermove', onSmudgeMouseMove, true);
    document.addEventListener('pointerup',   onSmudgeMouseUp,   true);
    editorState.smudgeBrush.eventsBound = true;
    console.log("[slowargo.js] Smudge tool events bound to document (capture)");
}

/**
 * Handle pointerdown for the Smudge Brush.
 * Samples the composite image at the brush location to initialize the carried buffer,
 * then begins the smudge stroke.
 * @param {PointerEvent} e
 */
function onSmudgeMouseDown(e) {
    if (!editorState.smudgeBrush.active) return;
    if (!isPointerInBrushArea(e)) return;

    // Allow mask editor pan when space is held
    if (isSpacePressed) return;

    e.stopImmediatePropagation();
    e.preventDefault();

    const { x: cx, y: cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

    // Sample composite at brush location
    const radius = getBrushRadius();
    const carried = sampleComposite(baseCanvas, paintCanvas, cx, cy, radius);
    if (!carried) return;

    editorState.smudgeBrush.carriedBuffer = carried;
    editorState.smudgeBrush.isDrawing = true;
    editorState.smudgeBrush.lastDrawX = cx;
    editorState.smudgeBrush.lastDrawY = cy;

    applySmudgeStroke(cx, cy);
}

/**
 * Handle pointermove for the Smudge Brush.
 * Applies smudge strokes while drawing, and always updates the overlay cursor.
 * @param {PointerEvent} e
 */
function onSmudgeMouseMove(e) {
    if (!editorState.smudgeBrush.active) return;
    if (!isPointerInBrushArea(e)) return;

    // Allow mask editor pan when space is held
    if (isSpacePressed) {
        renderSmudgeOverlay(0, 0); // Clear overlay
        return;
    }

    const { x: cx, y: cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

    if (editorState.smudgeBrush.isDrawing) {
        e.stopImmediatePropagation();
        applySmudgeStroke(cx, cy);
    }
    renderSmudgeOverlay(cx, cy);
}

/**
 * Handle pointerup for the Smudge Brush.
 * Ends the stroke, clears the carried buffer, and saves canvas history.
 * @param {PointerEvent} e
 */
function onSmudgeMouseUp(e) {
    if (!editorState.smudgeBrush.active || !editorState.smudgeBrush.isDrawing) return;
    e.stopImmediatePropagation();
    editorState.smudgeBrush.isDrawing = false;

    // Clear carried buffer
    editorState.smudgeBrush.carriedBuffer = null;

    saveCanvasHistory();
}

/**
 * Apply a smudge stroke from the last draw position to (cx, cy) using linear interpolation.
 * Step size = max(1, radius / 3) ensures enough stamp overlap for a continuous smear.
 * @param {number} cx - Target canvas X coordinate
 * @param {number} cy - Target canvas Y coordinate
 */
function applySmudgeStroke(cx, cy) {
    if (!baseCanvas || !paintCanvas) return;

    const radius = getBrushRadius();
    const step = Math.max(1, radius / 3);
    const { steps } = getStrokeInterpolation(
        editorState.smudgeBrush.lastDrawX,
        editorState.smudgeBrush.lastDrawY,
        cx,
        cy,
        step
    );

    for (let i = 1; i <= steps; i++) {
        const t  = i / steps;
        const point = getLerpPoint(
            editorState.smudgeBrush.lastDrawX,
            editorState.smudgeBrush.lastDrawY,
            cx,
            cy,
            t
        );
        stampSmudge(
            baseCanvas,
            paintCanvas,
            editorState.smudgeBrush.carriedBuffer,
            point.x,
            point.y,
            radius,
            SMUDGE_STRENGTH,
            1
        );
    }

    editorState.smudgeBrush.lastDrawX = cx;
    editorState.smudgeBrush.lastDrawY = cy;
}

const SMUDGE_STRENGTH = 0.8;

/**
 * Render the Smudge Brush overlay at the current mouse position.
 * Draws a brush circle in orange.
 * @param {number} mouseX - Current cursor X in canvas pixels
 * @param {number} mouseY - Current cursor Y in canvas pixels
 */
function renderSmudgeOverlay(mouseX, mouseY) {
    if (!brushToolOverlay) return;

    const ctx = brushToolOverlay.getContext('2d');
    ctx.clearRect(0, 0, brushToolOverlay.width, brushToolOverlay.height);

    const radius = getBrushRadius();

    // Brush circle in orange
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 165, 0, 0.9)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

/**
 * Remove Smudge Brush event listeners, save any in-progress stroke, and reset tool state.
 */
function cleanupSmudgeTool() {
    if (editorState.smudgeBrush.isDrawing) {
        saveCanvasHistory();
        editorState.smudgeBrush.isDrawing = false;
    }

    if (editorState.smudgeBrush.eventsBound) {
        document.removeEventListener('pointerdown', onSmudgeMouseDown, true);
        document.removeEventListener('pointermove', onSmudgeMouseMove, true);
        document.removeEventListener('pointerup',   onSmudgeMouseUp,   true);
        editorState.smudgeBrush.eventsBound = false;
    }

    editorState.smudgeBrush.carriedBuffer = null;
    editorState.smudgeBrush.active = false;
    editorState.smudgeBrush.isDrawing = false;
}

// === Brush Button Creation ===

/**
 * Create the Clone Brush toggle button.
 * Clicking the button toggles the Clone Brush on/off, deactivating any other active tool.
 * @returns {HTMLButtonElement} The clone button element
 */
function createCloneButton() {
    const cloneBtn = document.createElement("button");
    cloneBtn.className = "fast-forward-mode-toggle";
    cloneBtn.id = "clone-brush-button";
    cloneBtn.title = "Clone Brush (C): Alt+Click to set source, drag to paint";

    const cloneIcon = document.createElement("i");
    cloneIcon.className = "pi pi-clone";
    cloneBtn.appendChild(cloneIcon);

    const cloneText = document.createElement("span");
    cloneText.textContent = "Clone";
    cloneBtn.appendChild(cloneText);

    // Store reference for style updates
    cloneBtnRef = cloneBtn;

    cloneBtn.addEventListener("click", () => {
        const willActivate = !editorState.cloneBrush.active;
        deactivateAllCustomTools();
        if (willActivate) {
            ensurePaintLayerVisible();
            activateOpacityOverride();
            editorState.cloneBrush.active = true;
            updateCloneStyle();
        }
        if (brushToolOverlay) {
            brushToolOverlay.classList.toggle('active', isAnyCustomToolActive());
        }
        // console.log("[slowargo.js] Clone Brush:", editorState.cloneBrush.active ? "enabled" : "disabled");
    });

    return cloneBtn;
}

/**
 * Create the Smudge Brush toggle button.
 * Clicking the button toggles the Smudge Brush on/off, deactivating any other active tool.
 * @returns {HTMLButtonElement} The smudge button element
 */
function createSmudgeButton() {
    const smudgeBtn = document.createElement("button");
    smudgeBtn.className = "fast-forward-mode-toggle";
    smudgeBtn.id = "smudge-brush-button";
    smudgeBtn.title = "Smudge Brush (S): Drag to smudge pixels";

    const smudgeIcon = document.createElement("i");
    smudgeIcon.className = "pi pi-arrow-right-arrow-left";
    smudgeBtn.appendChild(smudgeIcon);

    const smudgeText = document.createElement("span");
    smudgeText.textContent = "Smudge";
    smudgeBtn.appendChild(smudgeText);

    // Store reference for style updates
    smudgeBtnRef = smudgeBtn;

    smudgeBtn.addEventListener("click", () => {
        const willActivate = !editorState.smudgeBrush.active;
        deactivateAllCustomTools();
        if (willActivate) {
            ensurePaintLayerVisible();
            activateOpacityOverride();
            editorState.smudgeBrush.active = true;
            updateSmudgeStyle();
        }
        if (brushToolOverlay) {
            brushToolOverlay.classList.toggle('active', isAnyCustomToolActive());
        }
        // console.log("[slowargo.js] Smudge Brush:", editorState.smudgeBrush.active ? "enabled" : "disabled");
    });

    return smudgeBtn;
}

// Module-level button reference for Transform tool
let transformBtnRef = null;

/**
 * Sync the transform tool button's CSS classes to reflect the current active state.
 */
function updateTransformStyle() {
    if (!transformBtnRef) return;
    transformBtnRef.classList.toggle('enabled', isTransformActive());
    transformBtnRef.classList.toggle('disabled', !isTransformActive());
}

/**
 * Create the Transform tool toggle button.
 * Clicking the button toggles the Transform tool on/off, deactivating any other active tool.
 * @returns {HTMLButtonElement} The transform button element
 */
function createTransformButton() {
    const transformBtn = document.createElement("button");
    transformBtn.className = "fast-forward-mode-toggle";
    transformBtn.id = "transform-tool-button";
    transformBtn.title = "Transform Tool (Q): Select and transform paint layer content";

    const transformIcon = document.createElement("i");
    transformIcon.className = "pi pi-arrows-alt";
    transformBtn.appendChild(transformIcon);

    const transformText = document.createElement("span");
    transformText.textContent = "Transform";
    transformBtn.appendChild(transformText);

    // Store reference for style updates
    transformBtnRef = transformBtn;

    transformBtn.addEventListener("click", () => {
        const willActivate = !isTransformActive();
        deactivateAllCustomTools();
        if (willActivate) {
            ensurePaintLayerVisible();
            activateOpacityOverride();
            // 直接激活Transform工具，不使用toggleTransform
            initTransformToolEvents();
            updateTransformStyle();
        }
        if (brushToolOverlay) {
            brushToolOverlay.classList.toggle('active', isAnyCustomToolActive());
        }
        // console.log("[slowargo.js] Transform Tool:", isTransformActive() ? "enabled" : "disabled");
    });

    return transformBtn;
}

// === Keyboard Event Handler ===

/**
 * Handle keyboard events for brush tools.
 * Currently handles Esc to deactivate tools, Space to allow mask editor pan,
 * 'C' to toggle Clone Brush, 'S' to toggle Smudge Brush.
 * @param {KeyboardEvent} e - The keyboard event
 * @returns {boolean} true if the event was handled, false otherwise
 */
function handleBrushToolKeydown(e) {
    // Ignore tool hotkeys when system modifiers are pressed (e.g. Ctrl+C / Alt+S / Cmd+Q).
    const hasSystemModifier = e.ctrlKey || e.altKey || e.metaKey;
    if (hasSystemModifier) {
        return false;
    }

    // Track space key for allowing mask editor pan (always track, even if no tool active)
    if (e.key === ' ') {
        isSpacePressed = true;
        return false; // Don't intercept, let mask editor handle it
    }

    // 'C' key to toggle Clone Brush (works even when no tool is active)
    if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        e.stopImmediatePropagation();
        const willActivate = !editorState.cloneBrush.active;
        deactivateAllCustomTools();
        if (willActivate) {
            ensurePaintLayerVisible();
            activateOpacityOverride();
            editorState.cloneBrush.active = true;
            updateCloneStyle();
        }
        if (brushToolOverlay) {
            brushToolOverlay.classList.toggle('active', isAnyCustomToolActive());
        }
        return true;
    }

    // 'S' key to toggle Smudge Brush (works even when no tool is active)
    if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        e.stopImmediatePropagation();
        const willActivate = !editorState.smudgeBrush.active;
        deactivateAllCustomTools();
        if (willActivate) {
            ensurePaintLayerVisible();
            activateOpacityOverride();
            editorState.smudgeBrush.active = true;
            updateSmudgeStyle();
        }
        if (brushToolOverlay) {
            brushToolOverlay.classList.toggle('active', isAnyCustomToolActive());
        }
        return true;
    }

    // 'Q' key to toggle Transform tool (works even when no tool is active)
    if (e.key === 'q' || e.key === 'Q') {
        e.preventDefault();
        e.stopImmediatePropagation();
        const willActivate = !isTransformActive();
        deactivateAllCustomTools();
        if (willActivate) {
            ensurePaintLayerVisible();
            activateOpacityOverride();
            initTransformToolEvents();
            updateTransformStyle();
        }
        if (brushToolOverlay) {
            brushToolOverlay.classList.toggle('active', isAnyCustomToolActive());
        }
        return true;
    }

    // Early return if no custom tool is active (below shortcuts require active tool)
    if (!isAnyCustomToolActive()) return false;

    // Esc to deactivate all custom tools
    if (e.key === 'Escape') {
        e.preventDefault();
        e.stopImmediatePropagation();
        deactivateAllCustomTools();
        if (brushToolOverlay) {
            brushToolOverlay.classList.remove('active');
        }
        return true;
    }

    return false;
}

/**
 * Handle keyboard up events for brush tools.
 * Tracks space key release for mask editor pan functionality.
 * @param {KeyboardEvent} e - The keyboard event
 * @returns {boolean} true if the event was handled, false otherwise
 */
function handleBrushToolKeyup(e) {
    if (isSpacePressed && e.key === ' ') {
        // always exit pan mode
        isSpacePressed = false;
        return false;
    }

    if (!isAnyCustomToolActive()) return false;

    return false;
}

// === Cleanup All Brush Tools ===

/**
 * Cleanup all brush tool resources including:
 * - Clone tool events and state
 * - Smudge tool events and state
 * - Transform tool events and state
 * - brushToolOverlay element
 * - Canvas references
 */
function cleanupAllBrushTools() {
    // Cleanup individual tools
    cleanupCloneTool();
    cleanupSmudgeTool();
    cleanupTransform();

    // Remove overlay element
    const overlay = brushToolOverlay || document.querySelector('#maskEditorCanvasContainer #brush-tool-overlay');
    if (overlay) overlay.remove();
    brushToolOverlay = null;

    // Clear transform module shared canvas references to avoid detached canvas retention
    setSharedOverlay(null);

    // Clear canvas references
    baseCanvas = null;
    paintCanvas = null;
    setSharedCanvases(null, null);

    // Restore original opacity if overridden
    restoreOpacityOverride();
}

// === Exports ===

export {
    // State (read-only access)
    editorState,

    // Shared resources
    brushToolOverlay,
    baseCanvas,
    paintCanvas,

    // Coordinate mapping
    displayToCanvas,

    // Brush radius
    getBrushRadius,

    // Overlay
    initBrushToolOverlay,

    // Tool management
    deactivateAllCustomTools,
    isAnyCustomToolActive,

    // Style updates
    updateCloneStyle,
    updateSmudgeStyle,
    updateTransformStyle,

    // Button refs (for external assignment)
    cloneBtnRef,
    smudgeBtnRef,
    transformBtnRef,

    // Button creation
    createCloneButton,
    createSmudgeButton,
    createTransformButton,

    // Keyboard handlers
    handleBrushToolKeydown,
    handleBrushToolKeyup,

    // Clone Brush
    initCloneToolEvents,
    renderCloneOverlay,
    cleanupCloneTool,

    // Smudge Brush
    initSmudgeToolEvents,
    renderSmudgeOverlay,
    cleanupSmudgeTool,

    // Transform Tool
    cleanupTransform,
    isTransformActive,

    // Cleanup all
    cleanupAllBrushTools,
};
