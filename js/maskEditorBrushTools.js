import { getMaskEditorStore } from "./utils.js";

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

// === Brush Tool Functions ===

let globalBrushRadius = 20;
let lastBrushReadTime = 0;

/**
 * Get the current brush radius, reading from the DOM input at most once per 500ms.
 * Falls back to the cached value when the input is absent or the cache is fresh.
 * @returns {number} The current brush radius in pixels
 */
function getBrushRadius() {
    const now = Date.now();
    if (now - lastBrushReadTime >= 500) {
        const rangeInput = document.querySelector('input.maskEditor_sidePanelBrushRange');
        if (rangeInput?.value) {
            globalBrushRadius = parseFloat(rangeInput.value);
            lastBrushReadTime = now;
        }
    }
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
    const container = document.querySelector('#maskEditorCanvasContainer');
    if (!container || container.querySelector('#brush-tool-overlay')) return;

    if (getComputedStyle(container).position === 'static') {
        container.style.position = 'relative';
    }

    const canvases = container.querySelectorAll('canvas');
    baseCanvas  = canvases[0];
    paintCanvas = canvases[1];

    const overlay = document.createElement('canvas');
    overlay.id = 'brush-tool-overlay';
    overlay.width  = canvases[0].width;
    overlay.height = canvases[0].height;
    container.appendChild(overlay);
    brushToolOverlay = overlay;
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
    if (brushToolOverlay) {
        const ctx = brushToolOverlay.getContext('2d');
        ctx?.clearRect(0, 0, brushToolOverlay.width, brushToolOverlay.height);
    }
}

/**
 * Check whether any custom brush tool (Clone or Smudge) is currently active.
 * @returns {boolean} true if at least one custom tool is active
 */
function isAnyCustomToolActive() {
    return editorState.cloneBrush.active || editorState.smudgeBrush.active;
}

// === Clone Brush Functions ===

/**
 * Map client (CSS) coordinates to canvas pixel coordinates.
 * Necessary because the canvas display size may differ from its intrinsic pixel size
 * (e.g. the canvas is rendered at 50% scale, so CSS pixels must be multiplied by 2).
 *
 * 进行坐标映射。需要进行坐标映射是因为 Canvas 的显示尺寸与实际像素尺寸可能不一致（画布可能以缩小/放大状态显示)
 *
 * @param {HTMLCanvasElement} canvas - The target canvas element
 * @param {number} clientX - Client X coordinate from the pointer event
 * @param {number} clientY - Client Y coordinate from the pointer event
 * @returns {{cx: number, cy: number}} Canvas pixel coordinates
 */
function displayToCanvas(canvas, clientX, clientY) {
    const rect = canvas.getBoundingClientRect(); // 获取 CSS 显示区域
    return {
        // 计算鼠标在显示区域内的相对位置，再按比例转换为像素坐标
        // 例如缩小到 50%, rect.width 为 canvas.width 的一半，下面公式就相当于 offset * 2，放大回正确的像素位置
        cx: (clientX - rect.left) * canvas.width  / rect.width,
        cy: (clientY - rect.top)  * canvas.height / rect.height,
    };
}

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

    const { cx, cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

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

    const {cx, cy} = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

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

    const store = getMaskEditorStore();
    store?.canvasHistory?.saveState?.();
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
    // 步长与笔刷半径成正比, 确保至少为 1 像素, radius/3 保证相邻 stamp 有足够重叠
    const step = Math.max(1, radius / 3);
    // 两点间直线距离
    const dist  = Math.hypot(cx - editorState.cloneBrush.lastDrawX, cy - editorState.cloneBrush.lastDrawY);
    // 需要的插值步数
    const steps = Math.ceil(dist / step);

    // 线性插值填充间隙
    if (steps === 0) {
        // 没有移动距离时至少绘制一个点（仅 mouse down 的情况）
        stampClone(cx, cy, radius);
    } else {
        for (let i = 1; i <= steps; i++) {
            const t  = i / steps; // 0.0 ~ 1.0 的插值因子
            // 在 lastDraw 和 current 之间均匀分布 stamp
            const dx = editorState.cloneBrush.lastDrawX + (cx - editorState.cloneBrush.lastDrawX) * t;
            const dy = editorState.cloneBrush.lastDrawY + (cy - editorState.cloneBrush.lastDrawY) * t;
            // 在插值点执行克隆
            stampClone(dx, dy, radius);
        }
    }

    editorState.cloneBrush.lastDrawX = cx;
    editorState.cloneBrush.lastDrawY = cy;
}

/**
 * Stamp a single clone brush dab at (drawX, drawY).
 * Copies pixels from the base canvas source region to the paint canvas destination,
 * blending with a Gaussian falloff within the brush radius.
 * @param {number} drawX - Destination center X in canvas pixels
 * @param {number} drawY - Destination center Y in canvas pixels
 * @param {number} radius - Brush radius in canvas pixels
 */
function stampClone(drawX, drawY, radius) {
    const r = Math.ceil(radius);
    const sigma = radius * 0.4;

    const srcX = editorState.cloneBrush.sampleX + (drawX - editorState.cloneBrush.strokeStartX);
    const srcY = editorState.cloneBrush.sampleY + (drawY - editorState.cloneBrush.strokeStartY);

    const srcL = Math.max(0, Math.round(srcX - r));
    const srcT = Math.max(0, Math.round(srcY - r));
    const srcR = Math.min(baseCanvas.width,  Math.round(srcX + r));
    const srcB = Math.min(baseCanvas.height, Math.round(srcY + r));
    if (srcL >= srcR || srcT >= srcB) return;

    const w = srcR - srcL;
    const h = srcB - srcT;

    const dstL = Math.round(drawX - (srcX - srcL));
    const dstT = Math.round(drawY - (srcY - srcT));

    const baseCtx  = baseCanvas.getContext('2d',  { willReadFrequently: true });
    const paintCtx = paintCanvas.getContext('2d', { willReadFrequently: true });

    const srcData = baseCtx.getImageData(srcL, srcT, w, h);
    const dstData = paintCtx.getImageData(dstL, dstT, w, h);

    for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
            const dx   = (srcL + px) - srcX;
            const dy   = (srcT + py) - srcY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > radius) continue;

            const weight = Math.exp(-(dist * dist) / (2 * sigma * sigma));
            const i = (py * w + px) * 4;

            dstData.data[i]   = srcData.data[i]   * weight + dstData.data[i]   * (1 - weight);
            dstData.data[i+1] = srcData.data[i+1] * weight + dstData.data[i+1] * (1 - weight);
            dstData.data[i+2] = srcData.data[i+2] * weight + dstData.data[i+2] * (1 - weight);
            dstData.data[i+3] = Math.max(dstData.data[i+3], Math.round(srcData.data[i+3] * weight));
        }
    }

    paintCtx.putImageData(dstData, dstL, dstT);
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
        getMaskEditorStore()?.canvasHistory?.saveState?.();
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

    const { cx, cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

    // Sample composite at brush location
    const radius = getBrushRadius();
    const carried = sampleComposite(cx, cy, radius);
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

    const { cx, cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

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

    const store = getMaskEditorStore();
    store?.canvasHistory?.saveState?.();
}

/**
 * Sample the composited image (paint over base) within a circular brush region.
 * Returns a Float32Array buffer and its bounding box, used as the smudge carried buffer.
 * @param {number} centerX - Brush center X in canvas pixels
 * @param {number} centerY - Brush center Y in canvas pixels
 * @param {number} radius - Brush radius in canvas pixels
 * @returns {{data: Float32Array, width: number, height: number, offsetX: number, offsetY: number}|null}
 *   The sampled composite region, or null if the region is empty
 */
function sampleComposite(centerX, centerY, radius) {
    const r = Math.ceil(radius);
    const left   = Math.max(0, Math.round(centerX - r));
    const top    = Math.max(0, Math.round(centerY - r));
    const right  = Math.min(baseCanvas.width,  Math.round(centerX + r));
    const bottom = Math.min(baseCanvas.height, Math.round(centerY + r));
    if (left >= right || top >= bottom) return null;

    const w = right - left;
    const h = bottom - top;

    const baseCtx  = baseCanvas.getContext('2d',  { willReadFrequently: true });
    const paintCtx = paintCanvas.getContext('2d', { willReadFrequently: true });
    const baseData  = baseCtx.getImageData(left, top, w, h);
    const paintData = paintCtx.getImageData(left, top, w, h);

    // Composite: paint over base
    const composite = new Float32Array(w * h * 4);
    for (let i = 0; i < w * h * 4; i += 4) {
        const pa = paintData.data[i+3] / 255;
        composite[i]   = paintData.data[i]   * pa + baseData.data[i]   * (1 - pa);
        composite[i+1] = paintData.data[i+1] * pa + baseData.data[i+1] * (1 - pa);
        composite[i+2] = paintData.data[i+2] * pa + baseData.data[i+2] * (1 - pa);
        composite[i+3] = Math.max(paintData.data[i+3], baseData.data[i+3]);
    }

    return {
        data: composite,
        width: w,
        height: h,
        offsetX: left,
        offsetY: top
    };
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
    const dist  = Math.hypot(cx - editorState.smudgeBrush.lastDrawX, cy - editorState.smudgeBrush.lastDrawY);
    const steps = Math.ceil(dist / step);

    for (let i = 1; i <= steps; i++) {
        const t  = i / steps;
        const dx = editorState.smudgeBrush.lastDrawX + (cx - editorState.smudgeBrush.lastDrawX) * t;
        const dy = editorState.smudgeBrush.lastDrawY + (cy - editorState.smudgeBrush.lastDrawY) * t;
        stampSmudge(dx, dy, radius);
    }

    editorState.smudgeBrush.lastDrawX = cx;
    editorState.smudgeBrush.lastDrawY = cy;
}

const SMUDGE_STRENGTH = 0.8;

/**
 * Stamp a single smudge brush dab at (drawX, drawY).
 * Blends the carried buffer into the paint canvas using Gaussian falloff,
 * then progressively mixes the output back into the carried buffer to create
 * a trailing smear effect.
 * @param {number} drawX - Dab center X in canvas pixels
 * @param {number} drawY - Dab center Y in canvas pixels
 * @param {number} radius - Brush radius in canvas pixels
 */
function stampSmudge(drawX, drawY, radius) {
    const r = Math.ceil(radius);
    const sigma = radius * 0.4;

    const dstL = Math.max(0, Math.round(drawX - r));
    const dstT = Math.max(0, Math.round(drawY - r));
    const dstR = Math.min(paintCanvas.width,  Math.round(drawX + r));
    const dstB = Math.min(paintCanvas.height, Math.round(drawY + r));
    if (dstL >= dstR || dstT >= dstB) return;

    const w = dstR - dstL;
    const h = dstB - dstT;

    const baseCtx  = baseCanvas.getContext('2d',  { willReadFrequently: true });
    const paintCtx = paintCanvas.getContext('2d', { willReadFrequently: true });
    const baseData  = baseCtx.getImageData(dstL, dstT, w, h);
    const paintData = paintCtx.getImageData(dstL, dstT, w, h);

    const carried = editorState.smudgeBrush.carriedBuffer;
    if (!carried || !carried.data) return;

    const cb = carried.data;
    const cw = carried.width;
    const ch = carried.height;

    for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
            const canvasX = dstL + px;
            const canvasY = dstT + py;

            const dx = canvasX - drawX;
            const dy = canvasY - drawY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > radius) continue;

            const weight = Math.exp(-(dist * dist) / (2 * sigma * sigma));
            const s = SMUDGE_STRENGTH * weight;

            const di = (py * w + px) * 4;

            // Composite destination
            const pa = paintData.data[di+3] / 255;
            const destR = paintData.data[di]   * pa + baseData.data[di]   * (1 - pa);
            const destG = paintData.data[di+1] * pa + baseData.data[di+1] * (1 - pa);
            const destB = paintData.data[di+2] * pa + baseData.data[di+2] * (1 - pa);
            const destA = Math.max(paintData.data[di+3], baseData.data[di+3]);

            // Brush-relative coordinates: index carried buffer from brush center
            // This keeps the mapping correct regardless of how far the brush has moved
            const cbx = Math.round(dx) + r;
            const cby = Math.round(dy) + r;

            if (cbx >= 0 && cbx < cw && cby >= 0 && cby < ch) {
                const ci = (cby * cw + cbx) * 4;

                // Blend: output = carried * s + dest * (1 - s)
                const outR = cb[ci]   * s + destR * (1 - s);
                const outG = cb[ci+1] * s + destG * (1 - s);
                const outB = cb[ci+2] * s + destB * (1 - s);
                const outA = cb[ci+3] * s + destA * (1 - s);

                paintData.data[di]   = Math.round(outR);
                paintData.data[di+1] = Math.round(outG);
                paintData.data[di+2] = Math.round(outB);
                paintData.data[di+3] = Math.round(outA);

                // Update carried (progressive mixing)
                cb[ci]   = outR;
                cb[ci+1] = outG;
                cb[ci+2] = outB;
                cb[ci+3] = outA;
            }
        }
    }

    paintCtx.putImageData(paintData, dstL, dstT);
}

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
        getMaskEditorStore()?.canvasHistory?.saveState?.();
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

// === Keyboard Event Handler ===

/**
 * Handle keyboard events for brush tools.
 * Currently handles Esc to deactivate tools, Space to allow mask editor pan,
 * 'C' to toggle Clone Brush, 'S' to toggle Smudge Brush.
 * @param {KeyboardEvent} e - The keyboard event
 * @returns {boolean} true if the event was handled, false otherwise
 */
function handleBrushToolKeydown(e) {
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
            editorState.smudgeBrush.active = true;
            updateSmudgeStyle();
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
 * - brushToolOverlay element
 * - Canvas references
 */
function cleanupAllBrushTools() {
    // Cleanup individual tools
    cleanupCloneTool();
    cleanupSmudgeTool();

    // Remove overlay element
    if (brushToolOverlay) {
        brushToolOverlay.remove();
        brushToolOverlay = null;
    }

    // Clear canvas references
    baseCanvas = null;
    paintCanvas = null;
}

// === Exports ===

export {
    // State (read-only access)
    editorState,

    // Shared resources
    brushToolOverlay,
    baseCanvas,
    paintCanvas,

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

    // Button refs (for external assignment)
    cloneBtnRef,
    smudgeBtnRef,

    // Button creation
    createCloneButton,
    createSmudgeButton,

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

    // Cleanup all
    cleanupAllBrushTools,
};
