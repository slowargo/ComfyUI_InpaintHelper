# Clone Brush & Smudge Brush 实现文档

基于设计文档与实际代码生成。

**目标文件：**
- `js/maskEditorTurbo.js` - 主实现 (1289 行)
- `js/maskEditorTurbo.css` - 样式 (137 行)

---

## 架构概览

Mask Editor Turbo 模块扩展了 ComfyUI 的遮罩编辑器，添加了 Fast Forward Mode、Clone Brush 和 Smudge Brush 工具。

### 状态管理

```javascript
// 编辑器状态对象 (第 8-37 行)
const editorState = {
    // Fast Forward Mode
    cycleInProgress: false,
    fastForwardModeOn: true,
    sourceNodeId: null,

    // 编辑器模糊模式
    isBlurred: false,

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
        carriedBuffer: null,
        eventsBound: false,
    }
};
```

### 模块级共享资源

```javascript
// 第 39-42 行
let brushToolOverlay = null;  // 两个工具共享的覆盖层画布
let baseCanvas = null;        // canvases[0] - 基础图像层
let paintCanvas = null;       // canvases[1] - 绘制/遮罩层
```

---

## 共享基础设施

### 笔刷大小管理

```javascript
// 第 46-58 行
let globalBrushRadius = 20;

function getBrushRadius() {
    const rangeInput = document.querySelector('input.maskEditor_sidePanelBrushRange');
    if (rangeInput?.value) {
        return parseFloat(rangeInput.value);
    }
    return globalBrushRadius;
}

function setBrushRadius(radius) {
    globalBrushRadius = Math.max(5, Math.min(300, radius));
}
```

**注意：** 与设计文档中分别设置 `cloneBrushRadius` 和 `smudgeBrushRadius` 不同，实现中使用了统一的 `getBrushRadius()`，与原生笔刷大小滑块同步。

### 覆盖层画布初始化

```javascript
// 第 60-79 行
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
}
```

**实现说明：** 覆盖层 ID 从设计文档的 `#tool-overlay` 改为 `#brush-tool-overlay`，更清晰。

### 工具互斥

```javascript
// 第 104-122 行
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

function isAnyCustomToolActive() {
    return editorState.cloneBrush.active || editorState.smudgeBrush.active;
}
```

---

## Clone Brush 实现

### 事件处理

**设计 vs 实现差异：**
- **设计：** 事件绑定到 `#maskEditorCanvasContainer`，捕获阶段
- **实现：** 事件绑定到 `document`，捕获阶段，配合 `isPointerInBrushArea()` 检查

```javascript
// 第 145-152 行
function initCloneToolEvents() {
    if (editorState.cloneBrush.eventsBound) return;
    document.addEventListener('pointerdown', onCloneMouseDown, true);
    document.addEventListener('pointermove', onCloneMouseMove, true);
    document.addEventListener('pointerup',   onCloneMouseUp,   true);
    editorState.cloneBrush.eventsBound = true;
}

// 第 81-86 行 - 笔刷区域检测辅助函数
function isPointerInBrushArea(e) {
    if (!brushToolOverlay) return false;
    const rect = brushToolOverlay.getBoundingClientRect();
    return e.clientX >= rect.left && e.clientX <= rect.right &&
           e.clientY >= rect.top  && e.clientY <= rect.bottom;
}
```

### 鼠标事件处理器

```javascript
// 第 154-201 行
function onCloneMouseDown(e) {
    if (!editorState.cloneBrush.active) return;
    if (!isPointerInBrushArea(e)) return;

    e.stopImmediatePropagation();
    e.preventDefault();

    const { cx, cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

    if (e.altKey) {
        // Alt+点击：设置源采样点
        editorState.cloneBrush.hasSample  = true;
        editorState.cloneBrush.sampleX    = cx;
        editorState.cloneBrush.sampleY    = cy;
        renderCloneOverlay(cx, cy);
        return;
    }

    if (!editorState.cloneBrush.hasSample) return;

    // 开始绘制笔触
    editorState.cloneBrush.isDrawing    = true;
    editorState.cloneBrush.strokeStartX = cx;
    editorState.cloneBrush.strokeStartY = cy;
    editorState.cloneBrush.lastDrawX    = cx;
    editorState.cloneBrush.lastDrawY    = cy;
    applyCloneStroke(cx, cy);
}

function onCloneMouseMove(e) {
    if (!editorState.cloneBrush.active) return;
    if (!isPointerInBrushArea(e)) return;

    const {cx, cy} = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

    if (editorState.cloneBrush.isDrawing) {
        e.stopImmediatePropagation();
        applyCloneStroke(cx, cy);
    }
    renderCloneOverlay(cx, cy);
}

function onCloneMouseUp(e) {
    if (!editorState.cloneBrush.active || !editorState.cloneBrush.isDrawing) return;
    e.stopImmediatePropagation();
    editorState.cloneBrush.isDrawing = false;

    const store = getMaskEditorStore();
    store?.canvasHistory?.saveState?.();  // 每次笔触一个撤销检查点
}
```

### 笔触插值

```javascript
// 第 203-220 行
function applyCloneStroke(cx, cy) {
    if (!baseCanvas || !paintCanvas) return;

    const radius = getBrushRadius();
    const step = Math.max(1, radius / 3);
    const dist  = Math.hypot(cx - editorState.cloneBrush.lastDrawX, cy - editorState.cloneBrush.lastDrawY);
    const steps = Math.ceil(dist / step);

    for (let i = 1; i <= steps; i++) {
        const t  = i / steps;
        const dx = editorState.cloneBrush.lastDrawX + (cx - editorState.cloneBrush.lastDrawX) * t;
        const dy = editorState.cloneBrush.lastDrawY + (cy - editorState.cloneBrush.lastDrawY) * t;
        stampClone(dx, dy, radius);
    }

    editorState.cloneBrush.lastDrawX = cx;
    editorState.cloneBrush.lastDrawY = cy;
}
```

### 核心克隆算法（平行追踪）

```javascript
// 第 222-265 行
function stampClone(drawX, drawY, radius) {
    const r = Math.ceil(radius);
    const sigma = radius * 0.4;

    // 源位置：samplePoint + (currentPos - strokeStart)
    const srcX = editorState.cloneBrush.sampleX + (drawX - editorState.cloneBrush.strokeStartX);
    const srcY = editorState.cloneBrush.sampleY + (drawY - editorState.cloneBrush.strokeStartY);

    // 限制在画布边界内
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

            // Alpha 混合克隆
            dstData.data[i]   = srcData.data[i]   * weight + dstData.data[i]   * (1 - weight);
            dstData.data[i+1] = srcData.data[i+1] * weight + dstData.data[i+1] * (1 - weight);
            dstData.data[i+2] = srcData.data[i+2] * weight + dstData.data[i+2] * (1 - weight);
            dstData.data[i+3] = Math.max(dstData.data[i+3], Math.round(srcData.data[i+3] * weight));
        }
    }

    paintCtx.putImageData(dstData, dstL, dstT);
}
```

### 视觉反馈

```javascript
// 第 267-305 行
function renderCloneOverlay(mouseX, mouseY) {
    if (!brushToolOverlay || !editorState.cloneBrush.hasSample) return;

    const ctx = brushToolOverlay.getContext('2d');
    ctx.clearRect(0, 0, brushToolOverlay.width, brushToolOverlay.height);

    const radius = getBrushRadius();

    // 源十字准星（红色）
    drawCrosshair(ctx, editorState.cloneBrush.sampleX, editorState.cloneBrush.sampleY, '#ff6666', radius);

    // 绘制时的追踪源指示器
    if (editorState.cloneBrush.isDrawing) {
        const trackedX = editorState.cloneBrush.sampleX + (mouseX - editorState.cloneBrush.strokeStartX);
        const trackedY = editorState.cloneBrush.sampleY + (mouseY - editorState.cloneBrush.strokeStartY);
        drawCrosshair(ctx, trackedX, trackedY, 'rgba(255, 102, 102, 0.7)', radius);

        // 源到光标的虚线
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(trackedX, trackedY);
        ctx.lineTo(mouseX, mouseY);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // 笔刷圆圈
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawCrosshair(ctx, x, y, color, size = 8) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - size, y); ctx.lineTo(x + size, y);
    ctx.moveTo(x, y - size); ctx.lineTo(x, y + size);
    ctx.stroke();
}
```

---

## Smudge Brush 实现

### 状态

```javascript
// editorState 的一部分（第 29-36 行）
smudgeBrush: {
    active: false,
    isDrawing: false,
    lastDrawX: 0, lastDrawY: 0,
    carriedBuffer: null,      // Float32Array 携带采样像素
    eventsBound: false,
}
```

**简化：** 实现中将整个 `carriedBuffer` 对象（包含 data、width、height、offsetX、offsetY）存储在状态中，而不是将字段扁平化。

### 事件处理

```javascript
// 第 327-381 行
function initSmudgeToolEvents() {
    if (editorState.smudgeBrush.eventsBound) return;
    document.addEventListener('pointerdown', onSmudgeMouseDown, true);
    document.addEventListener('pointermove', onSmudgeMouseMove, true);
    document.addEventListener('pointerup',   onSmudgeMouseUp,   true);
    editorState.smudgeBrush.eventsBound = true;
}

function onSmudgeMouseDown(e) {
    if (!editorState.smudgeBrush.active) return;
    if (!isPointerInBrushArea(e)) return;

    e.stopImmediatePropagation();
    e.preventDefault();

    const { cx, cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

    // 在笔刷位置采样复合图像
    const radius = getBrushRadius();
    const carried = sampleComposite(cx, cy, radius);
    if (!carried) return;

    editorState.smudgeBrush.carriedBuffer = carried;
    editorState.smudgeBrush.isDrawing = true;
    editorState.smudgeBrush.lastDrawX = cx;
    editorState.smudgeBrush.lastDrawY = cy;

    applySmudgeStroke(cx, cy);
}

function onSmudgeMouseMove(e) {
    if (!editorState.smudgeBrush.active) return;
    if (!isPointerInBrushArea(e)) return;

    const { cx, cy } = displayToCanvas(brushToolOverlay, e.clientX, e.clientY);

    if (editorState.smudgeBrush.isDrawing) {
        e.stopImmediatePropagation();
        applySmudgeStroke(cx, cy);
    }
    renderSmudgeOverlay(cx, cy);
}

function onSmudgeMouseUp(e) {
    if (!editorState.smudgeBrush.active || !editorState.smudgeBrush.isDrawing) return;
    e.stopImmediatePropagation();
    editorState.smudgeBrush.isDrawing = false;

    editorState.smudgeBrush.carriedBuffer = null;

    const store = getMaskEditorStore();
    store?.canvasHistory?.saveState?.();
}
```

### 复合采样

```javascript
// 第 383-416 行
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

    // 复合：paint 层叠加在 base 层之上，使用 alpha 混合
    const composite = new Float32Array(w * h * 4);
    for (let i = 0; i < w * h * 4; i += 4) {
        const pa = paintData.data[i+3] / 255;
        composite[i]   = paintData.data[i]   * pa + baseData.data[i]   * (1 - pa);
        composite[i+1] = paintData.data[i+1] * pa + baseData.data[i+1] * (1 - pa);
        composite[i+2] = paintData.data[i+2] * pa + baseData.data[i+2] * (1 - pa);
        composite[i+3] = Math.max(paintData.data[i+3], baseData.data[i+3]);
    }

    return { data: composite, width: w, height: h, offsetX: left, offsetY: top };
}
```

### 笔触插值

```javascript
// 第 418-435 行
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
```

### 核心涂抹算法

**重要实现差异：**
- **设计：** 通过绝对画布坐标映射携带缓冲区
- **实现：** 通过笔刷相对坐标映射携带缓冲区（以笔刷为中心）

这确保涂抹效果自然跟随笔刷移动，而不是停留在固定的画布位置。

```javascript
// 第 437-515 行
const SMUDGE_STRENGTH = 0.8;  // 从设计的 0.5 增加，效果更强

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

            // 复合目标像素
            const pa = paintData.data[di+3] / 255;
            const destR = paintData.data[di]   * pa + baseData.data[di]   * (1 - pa);
            const destG = paintData.data[di+1] * pa + baseData.data[di+1] * (1 - pa);
            const destB = paintData.data[di+2] * pa + baseData.data[di+2] * (1 - pa);
            const destA = Math.max(paintData.data[di+3], baseData.data[di+3]);

            // 笔刷相对坐标用于携带缓冲区查找
            const cbx = Math.round(dx) + r;
            const cby = Math.round(dy) + r;

            if (cbx >= 0 && cbx < cw && cby >= 0 && cby < ch) {
                const ci = (cby * cw + cbx) * 4;

                // 混合：output = carried * s + dest * (1 - s)
                const outR = cb[ci]   * s + destR * (1 - s);
                const outG = cb[ci+1] * s + destG * (1 - s);
                const outB = cb[ci+2] * s + destB * (1 - s);
                const outA = cb[ci+3] * s + destA * (1 - s);

                paintData.data[di]   = Math.round(outR);
                paintData.data[di+1] = Math.round(outG);
                paintData.data[di+2] = Math.round(outB);
                paintData.data[di+3] = Math.round(outA);

                // 更新携带缓冲区（渐进混合）
                cb[ci]   = outR;
                cb[ci+1] = outG;
                cb[ci+2] = outB;
                cb[ci+3] = outA;
            }
        }
    }

    paintCtx.putImageData(paintData, dstL, dstT);
}
```

### 视觉反馈

```javascript
// 第 517-531 行
function renderSmudgeOverlay(mouseX, mouseY) {
    if (!brushToolOverlay) return;

    const ctx = brushToolOverlay.getContext('2d');
    ctx.clearRect(0, 0, brushToolOverlay.width, brushToolOverlay.height);

    const radius = getBrushRadius();

    // 橙色笔刷圆圈（与克隆的白色区分）
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 165, 0, 0.9)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
}
```

---

## UI 实现

### 按钮容器

```javascript
// 第 971-981 行
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
```

### 按钮布局

```
[FF] [Mask] [All] [Clone] [Smudge] [Blur/Eye]
```

```javascript
// 第 885-1059 行
// 按钮在容器中的顺序：
buttonContainer.appendChild(toggleBtn);          // FF Mode
buttonContainer.appendChild(reloadMaskOnlyBtn);  // Mask
buttonContainer.appendChild(reloadAllBtn);       // All
buttonContainer.appendChild(cloneBtn);           // Clone
buttonContainer.appendChild(smudgeBtn);          // Smudge
buttonContainer.appendChild(blurBtn);            // Blur/Eye
```

### Clone 按钮

```javascript
// 第 997-1021 行
const cloneBtn = document.createElement("button");
cloneBtn.className = "fast-forward-mode-toggle";
cloneBtn.id = "clone-brush-button";
cloneBtn.title = "Clone Brush: Alt+Click to set source, drag to paint";
const cloneIcon = document.createElement("i");
cloneIcon.className = "pi pi-clone";
cloneBtn.appendChild(cloneIcon);
const cloneText = document.createElement("span");
cloneText.textContent = "Clone";
cloneBtn.appendChild(cloneText);
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
});
```

### Smudge 按钮

```javascript
// 第 1023-1047 行
const smudgeBtn = document.createElement("button");
smudgeBtn.className = "fast-forward-mode-toggle";
smudgeBtn.id = "smudge-brush-button";
smudgeBtn.title = "Smudge Brush: Drag to smudge pixels";
const smudgeIcon = document.createElement("i");
smudgeIcon.className = "pi pi-arrow-right-arrow-left";
smudgeBtn.appendChild(smudgeIcon);
const smudgeText = document.createElement("span");
smudgeText.textContent = "Smudge";
smudgeBtn.appendChild(smudgeText);
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
});
```

---

## 键盘快捷键

### 笔刷大小调整

```javascript
// 第 1072-1108 行
if (isAnyCustomToolActive()) {
    const rangeInput = document.querySelector('input.maskEditor_sidePanelBrushRange');
    if (e.key === '[') {
        e.preventDefault();
        const currentRadius = getBrushRadius();
        const newRadius = Math.max(5, currentRadius - 5);
        setBrushRadius(newRadius);
        if (rangeInput) {
            rangeInput.value = newRadius;
            rangeInput.dispatchEvent(new Event('input', { bubbles: true }));
            rangeInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
        // 用新大小重新渲染覆盖层
        if (editorState.cloneBrush.active) {
            renderCloneOverlay(editorState.cloneBrush.lastDrawX, editorState.cloneBrush.lastDrawY);
        } else if (editorState.smudgeBrush.active) {
            renderSmudgeOverlay(editorState.smudgeBrush.lastDrawX, editorState.smudgeBrush.lastDrawY);
        }
        return;
    }
    // ... ']' 增加类似
}
```

---

## 生命周期管理

### 编辑器打开

```javascript
// 第 1190-1211 行
function onEditorReady(dialog) {
    initialized = true;
    currentDialog = dialog;
    restoreColorAndAddToggle();
    initBrushToolOverlay();
    initCloneToolEvents();
    initSmudgeToolEvents();

    // 模糊模式点击处理器
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
```

### 编辑器关闭清理

```javascript
// 第 1240-1269 行
if (!dialog && initialized) {
    initialized = false;

    // 清理 clone 工具
    cleanupCloneTool();

    // 清理 smudge 工具
    cleanupSmudgeTool();

    // 清理共享覆盖层资源
    brushToolOverlay?.remove();
    brushToolOverlay = null;
    baseCanvas = null;
    paintCanvas = null;

    // 移除点击监听器
    if (currentDialog && dialogClickHandler) {
        currentDialog.removeEventListener('click', dialogClickHandler);
        dialogClickHandler = null;
    }

    // 清理 UI
    cleanupFastForwardUI(currentDialog);

    // 积极的 GC 提示
    currentDialog.querySelectorAll('canvas').forEach(c => {
        c.width = 0; c.height = 0; c.parentNode?.removeChild(c);
    });
    currentDialog.querySelectorAll('img').forEach(c => {
        c.src = ''; c.parentNode?.removeChild(c);
    });
    currentDialog.innerHTML = "";
    currentDialog = null;
}
```

### 工具清理函数

```javascript
// 第 307-323 行
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

// 第 533-549 行
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
```

---

## CSS 样式

### 覆盖层

```css
#brush-tool-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 50;
    cursor: crosshair;
}

#brush-tool-overlay.active {
    pointer-events: all;
}
```

### 按钮样式

```css
.fast-forward-mode-toggle {
    background: rgba(30, 144, 255, 0.5);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    cursor: pointer;
    font-size: 11px;
    font-weight: bold;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 3px;
}

.fast-forward-mode-toggle.enabled {
    background: rgba(30, 144, 255, 0.9);
    box-shadow: 0 0 8px rgba(30, 144, 255, 0.8);
}

.fast-forward-mode-toggle.disabled {
    background: rgba(30, 144, 255, 0.4);
    box-shadow: none;
}
```

---

## 设计与实现对比

| 方面 | 设计 | 实现 | 说明 |
|------|------|------|------|
| 覆盖层 ID | `#tool-overlay` | `#brush-tool-overlay` | 更具描述性的名称 |
| 事件目标 | `#maskEditorCanvasContainer` | `document` + 区域检查 | 全局捕获 + 边界检查 |
| 笔刷大小 | 分别设置 clone/smudge 半径 | 通过原生滑块统一 | 与内置笔刷共享 |
| 涂抹强度 | 0.5 | 0.8 | 更强的效果 |
| 携带缓冲区映射 | 绝对画布坐标 | 笔刷相对坐标 | 像素随笔刷移动 |
| 十字准星大小 | 固定 8px | 可配置（默认 8px） | 添加 radius 参数 |
| Clone 笔刷颜色 | 白色 (`#fff`) | 浅红色 (`#ff6666`) | 更好的可见性 |
| 按钮样式 | `.reload-mask-button` 基础 | `.fast-forward-mode-toggle` | 与 FF 按钮一致 |

---

## 关键实现决策

1. **统一笔刷大小：** 两个工具共享原生笔刷大小滑块，确保一致的用户体验。

2. **文档上的指针事件：** 事件绑定到 `document` 而非容器以确保可靠捕获，`isPointerInBrushArea()` 提供空间过滤。

3. **涂抹的笔刷相对坐标：** 携带缓冲区通过笔刷相对坐标索引，使涂抹效果自然跟随笔刷移动。

4. **更高的涂抹强度：** 从 0.5 增加到 0.8 以获得更明显的效果。

5. **模块级共享资源：** 覆盖层和画布引用是模块级的而非状态对象中，简化了访问模式。

6. **编辑器关闭时清理：** 包括将画布尺寸归零和清除图像源在内的积极清理，以帮助垃圾回收。
