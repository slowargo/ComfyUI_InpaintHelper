# Clone Brush Feature Design (v2)

Target files: `js/maskEditorTurbo.js` + `js/maskEditorTurbo.css`

---

## Core Concept

Clone brush samples pixels from the base layer (`canvases[0]`) and paints them onto the paint layer (`canvases[1]`). An **overlay canvas** (added as a child of `#maskEditorCanvasContainer`) handles mouse event capture and renders visual feedback without interfering with the existing tools when clone mode is off.

---

## State

```javascript
const cloneState = {
    active: false,
    hasSample: false,
    sampleX: 0, sampleY: 0,         // source point, canvas pixel coords (set by Alt+Click)
    strokeStartX: 0, strokeStartY: 0, // recorded at mousedown, fixed for the whole stroke
    lastDrawX: 0, lastDrawY: 0,      // previous step position, for stroke interpolation
    brushRadius: 20,                 // canvas pixels
    isDrawing: false,
    eventsBound: false,
};
```

No `sampleCanvas`, no `altPressed`, no `brushSettings` cache, no `resizeObserver` reference.

---

## Overlay Canvas

Created once on `onEditorReady()`, removed on editor close.

```javascript
function initCloneBrushOverlay() {
    const container = document.querySelector('#maskEditorCanvasContainer');
    if (!container || container.querySelector('#clone-brush-overlay')) return;

    const base = container.querySelector('canvas');
    const overlay = document.createElement('canvas');
    overlay.id = 'clone-brush-overlay';
    overlay.width  = base.width;
    overlay.height = base.height;
    // CSS: position absolute, full coverage, pointer-events toggled by clone mode
    container.appendChild(overlay);
    cloneState.overlay = overlay;
}
```

CSS controls event pass-through:

```css
#clone-brush-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;   /* default: events reach native tools */
    z-index: 50;
    cursor: crosshair;
}
#clone-brush-overlay.active {
    pointer-events: all;    /* clone mode: capture all mouse events */
}
```

No Blur Mode CSS needed — the overlay is inside the dialog and moves with it automatically.

---

## Coordinate Mapping

Display (CSS) coordinates → canvas pixel coordinates:

```javascript
function displayToCanvas(canvas, clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    return {
        cx: (clientX - rect.left) * canvas.width  / rect.width,
        cy: (clientY - rect.top)  * canvas.height / rect.height,
    };
}
```

No `devicePixelRatio` needed — it cancels out algebraically.

---

## Event Handling

Mounted on `#maskEditorCanvasContainer` in **capture phase**. No dependency on which tool the mask editor thinks is active.

```javascript
function initCloneToolEvents() {
    if (cloneState.eventsBound) return;
    const container = document.querySelector('#maskEditorCanvasContainer');
    container.addEventListener('mousedown', onCloneMouseDown, true);
    container.addEventListener('mousemove', onCloneMouseMove, true);
    container.addEventListener('mouseup',   onCloneMouseUp,   true);
    cloneState.eventsBound = true;
}

function onCloneMouseDown(e) {
    if (!cloneState.active) return;
    e.stopImmediatePropagation();
    e.preventDefault(); // suppress browser context menu (Firefox Alt+click)

    const canvases = document.querySelectorAll('#maskEditorCanvasContainer canvas');
    const { cx, cy } = displayToCanvas(canvases[1], e.clientX, e.clientY);

    if (e.altKey) {
        cloneState.hasSample  = true;
        cloneState.sampleX    = cx;
        cloneState.sampleY    = cy;
        renderOverlay(cx, cy);
        return;
    }

    if (!cloneState.hasSample) return;

    cloneState.isDrawing    = true;
    cloneState.strokeStartX = cx;
    cloneState.strokeStartY = cy;
    cloneState.lastDrawX    = cx;
    cloneState.lastDrawY    = cy;
    applyCloneStroke(cx, cy); // paint first dot immediately
}

function onCloneMouseMove(e) {
    if (!cloneState.active) return;

    const canvases = document.querySelectorAll('#maskEditorCanvasContainer canvas');
    const { cx, cy } = displayToCanvas(canvases[1], e.clientX, e.clientY);

    if (cloneState.isDrawing) {
        e.stopImmediatePropagation();
        applyCloneStroke(cx, cy);
    }
    renderOverlay(cx, cy); // always update visual feedback
}

function onCloneMouseUp(e) {
    if (!cloneState.active || !cloneState.isDrawing) return;
    e.stopImmediatePropagation();
    cloneState.isDrawing = false;

    // Single saveState per stroke: GPU sync + undo checkpoint
    const store = getMaskEditorStore();
    store?.canvasHistory?.saveState?.();
}
```

---

## Drawing Algorithm (Parallel Tracking Clone)

The key design: **do not pre-capture a sample patch**. Sample directly from the base layer at each draw step using the accumulated offset from stroke start.

```
source position at step P = samplePoint + (P - strokeStart)
```

This produces standard clone brush parallel-tracking behavior: dragging further samples further in the same direction from the original source point.

```javascript
function applyCloneStroke(cx, cy) {
    const canvases = document.querySelectorAll('#maskEditorCanvasContainer canvas');
    const baseCanvas  = canvases[0];
    const paintCanvas = canvases[1];

    // Walk from lastDraw to current in steps (avoids gaps in fast strokes)
    const step = Math.max(1, cloneState.brushRadius / 3);
    const dist  = Math.hypot(cx - cloneState.lastDrawX, cy - cloneState.lastDrawY);
    const steps = Math.ceil(dist / step);

    for (let i = 1; i <= steps; i++) {
        const t  = i / steps;
        const dx = cloneState.lastDrawX + (cx - cloneState.lastDrawX) * t;
        const dy = cloneState.lastDrawY + (cy - cloneState.lastDrawY) * t;
        stampClone(baseCanvas, paintCanvas, dx, dy);
    }

    cloneState.lastDrawX = cx;
    cloneState.lastDrawY = cy;
    // No saveState here — called once on mouseup
}

function stampClone(baseCanvas, paintCanvas, drawX, drawY) {
    const r = Math.ceil(cloneState.brushRadius);
    const sigma = cloneState.brushRadius * 0.4;

    // Source position: samplePoint + offset from stroke start
    const srcX = cloneState.sampleX + (drawX - cloneState.strokeStartX);
    const srcY = cloneState.sampleY + (drawY - cloneState.strokeStartY);

    // Clamp source region to base canvas bounds
    const srcL = Math.max(0, Math.round(srcX - r));
    const srcT = Math.max(0, Math.round(srcY - r));
    const srcR = Math.min(baseCanvas.width,  Math.round(srcX + r));
    const srcB = Math.min(baseCanvas.height, Math.round(srcY + r));
    if (srcL >= srcR || srcT >= srcB) return;

    const w = srcR - srcL;
    const h = srcB - srcT;

    // Corresponding destination region (same size, offset by draw - src delta)
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
            if (dist > cloneState.brushRadius) continue;

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
```

---

## Overlay Visual Feedback

All feedback is drawn on the overlay canvas (never modifies the actual layers).

```javascript
function renderOverlay(mouseX, mouseY) {
    if (!cloneState.overlay) return;
    const ctx = cloneState.overlay.getContext('2d');
    ctx.clearRect(0, 0, cloneState.overlay.width, cloneState.overlay.height);

    if (!cloneState.hasSample) return;

    // 1. Source point crosshair (red)
    drawCrosshair(ctx, cloneState.sampleX, cloneState.sampleY, '#ff4444');

    // 2. Current sample source indicator (while drawing: tracked position)
    if (cloneState.isDrawing) {
        const trackedX = cloneState.sampleX + (mouseX - cloneState.strokeStartX);
        const trackedY = cloneState.sampleY + (mouseY - cloneState.strokeStartY);
        drawCrosshair(ctx, trackedX, trackedY, 'rgba(255, 68, 68, 0.5)');

        // Dashed line from tracked source to brush position
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(trackedX, trackedY);
        ctx.lineTo(mouseX, mouseY);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // 3. Brush circle at cursor
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, cloneState.brushRadius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawCrosshair(ctx, x, y, color) {
    const size = 8;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x - size, y); ctx.lineTo(x + size, y);
    ctx.moveTo(x, y - size); ctx.lineTo(x, y + size);
    ctx.stroke();
}
```

---

## Undo / Redo

`saveState()` is called **once per stroke** on `mouseup`. This creates a single undo checkpoint per drag, consistent with the mask editor's native brush behavior.

Do not call `saveState()` inside the draw loop — this would create a checkpoint for every interpolation step, making Ctrl+Z undo only tiny increments.

Expected behavior: Ctrl+Z undoes the entire last clone stroke. Likely works automatically since `canvasHistory` snapshots all painted layers. Verify by testing after implementation.

---

## Keyboard Shortcuts

Added inside the existing `keydown` handler, only when `cloneState.active`:

- `[` — `brushRadius = Math.max(5, brushRadius - 5)`
- `]` — `brushRadius = Math.min(300, brushRadius + 5)`

---

## UI Button

Added in `addFastForwardToggleButton()`. Button order:

```
[FF] [↺Mask] [↺All] [⊕Clone] [👁]
```

Clone button reuses `.reload-mask-button` base style with an `.active` modifier when clone mode is on.

```javascript
const cloneBtn = document.createElement('button');
cloneBtn.className = 'reload-mask-button';
cloneBtn.title = 'Clone Brush: Alt+Click to set source, drag to paint';

cloneBtn.addEventListener('click', () => {
    cloneState.active = !cloneState.active;
    cloneBtn.classList.toggle('active', cloneState.active);
    if (cloneState.overlay) {
        cloneState.overlay.classList.toggle('active', cloneState.active);
    }
    if (!cloneState.active) {
        // Clear overlay and reset sample when deactivating
        const ctx = cloneState.overlay?.getContext('2d');
        ctx?.clearRect(0, 0, cloneState.overlay.width, cloneState.overlay.height);
        cloneState.hasSample = false;
    }
});
```

---

## Lifecycle

| Event | Action |
|-------|--------|
| `onEditorReady()` | `initCloneBrushOverlay()` then `initCloneToolEvents()` |
| Editor closed (`cleanupFastForwardUI`) | remove `#clone-brush-overlay`; remove capture listeners; reset `cloneState` |
| `toggleEditorBlur()` enters blur | no special handling needed — overlay moves with dialog |

Cleanup:

```javascript
function cleanupCloneTool() {
    if (cloneState.isDrawing) {
        getMaskEditorStore()?.canvasHistory?.saveState?.();
        cloneState.isDrawing = false;
    }

    if (cloneState.eventsBound) {
        const container = document.querySelector('#maskEditorCanvasContainer');
        container?.removeEventListener('mousedown', onCloneMouseDown, true);
        container?.removeEventListener('mousemove', onCloneMouseMove, true);
        container?.removeEventListener('mouseup',   onCloneMouseUp,   true);
        cloneState.eventsBound = false;
    }

    cloneState.overlay?.remove();
    cloneState.overlay    = null;
    cloneState.active     = false;
    cloneState.hasSample  = false;
    cloneState.isDrawing  = false;
}
```

---

## CSS Additions

```css
/* Clone button active state */
.reload-mask-button.active {
    background: rgba(255, 140, 0, 0.9);
    box-shadow: 0 0 8px rgba(255, 140, 0, 0.8);
}

/* Overlay — see pointer-events toggled via .active class in JS */
#clone-brush-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 50;
    cursor: crosshair;
}

#clone-brush-overlay.active {
    pointer-events: all;
}
```

No Blur Mode CSS for the overlay — it lives inside the dialog and inherits its transform.

---

## File Changes

| File | Lines added (estimate) |
|------|------------------------|
| `maskEditorTurbo.js` | ~130 |
| `maskEditorTurbo.css` | ~20 |

---

## Issues Resolved vs v1

| v1 Issue | Resolution |
|----------|------------|
| Clone algorithm was fixed-stamp, not tracking | `stampClone` computes `samplePoint + (drawPos - strokeStart)` each step |
| Preview offset used `lastDrawX` (wrong) | Preview now uses `strokeStartX` for offset calculation |
| `saveState()` called inside draw loop | Moved to `mouseup` only |
| `isPenToolActive()` unnecessary coupling | Removed; capture-phase interception is sufficient |
| `ResizeObserver` for static dialog | Removed; size synced once on init |
| GPU history depth `states.shift()` | Removed; never mutate internal Pinia state |
| `altPressed` dead field | Removed; use `e.altKey` directly |
| `devicePixelRatio` math was redundant | Simplified to `canvas.width / rect.width` |
| Blur Mode CSS double-transform bug | Removed; overlay inherits dialog transform |
| `rescaleSampleCanvas` sampled from stale data | No pre-capture at all; sample from base layer live each step |
| `saveCanvasState()` undefined | Consistently use `store.canvasHistory.saveState()` directly |
