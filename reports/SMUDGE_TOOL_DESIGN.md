# Smudge Tool Design Plan

## Context

Mask Editor Turbo 已实现 Clone Brush 工具，现在需要新增 Smudge（涂抹）工具。Smudge 模拟"手指涂抹"效果——拖动时将起始位置的像素沿笔触路径"推涂"，像素颜色随拖动距离逐渐与目标混合。

**设计决策**：
- **目标图层**：从 Base + Paint 复合视图读取像素，写入 Paint 层
- **强度参数**：固定 0.5，不提供 UI 控件

## Algorithm

### Core Smudge Operation

```
On mousedown at P:
  1. Compute composite of base + paint at brush region around P
  2. Store into carriedBuffer (ImageData)

On each interpolated step at Q:
  1. Read composite at brush region around Q → destBuffer
  2. For each pixel within radius:
     - w = gaussian_weight(distance_from_center)
     - s = STRENGTH * w                          // effective strength
     - output = carried * s + dest * (1 - s)     // blend carried into dest
     - carried = output                           // update carried (progressive mixing)
  3. Write output pixels to Paint layer at Q

On mouseup:
  1. canvasHistory.saveState() for undo
  2. Clear carriedBuffer
```

### Compositing Formula (per pixel)

```javascript
// paint layer alpha (0-255) → [0,1]
const a = paintData[i+3] / 255;
composite_r = paintData[i]   * a + baseData[i]   * (1 - a);
composite_g = paintData[i+1] * a + baseData[i+1] * (1 - a);
composite_b = paintData[i+2] * a + baseData[i+2] * (1 - a);
composite_a = Math.max(paintData[i+3], baseData[i+3]);
```

### Gaussian Falloff

与 Clone Brush 相同：`sigma = radius * 0.4`，`weight = exp(-(dist²) / (2σ²))`

### Stroke Interpolation

与 Clone Brush 相同：`step = Math.max(1, radius / 3)`，线性插值避免间隙

## Architecture Changes

### 1. Shared Overlay Refactor

**目的**：Clone Brush 和 Smudge 共享同一个 overlay canvas，避免 DOM 重复。

**改动**：
- 将 `initCloneBrushOverlay()` 拆分为通用的 `initToolOverlay()`
- 提取 `toolOverlay`、`baseCanvas`、`paintCanvas` 为模块级变量
- Clone Brush 的 `editorState.cloneBrush.overlay/baseCanvas/paintCanvas` 改为引用模块级变量
- overlay ID 从 `#clone-brush-overlay` 改为 `#tool-overlay`
- CSS 选择器同步更新

```javascript
// Module-level shared resources
let toolOverlay = null;
let baseCanvas = null;
let paintCanvas = null;

function initToolOverlay() {
    const container = document.querySelector('#maskEditorCanvasContainer');
    if (!container || container.querySelector('#tool-overlay')) return;

    if (getComputedStyle(container).position === 'static') {
        container.style.position = 'relative';
    }

    const canvases = container.querySelectorAll('canvas');
    baseCanvas  = canvases[0];
    paintCanvas = canvases[1];

    const overlay = document.createElement('canvas');
    overlay.id = 'tool-overlay';
    overlay.width  = canvases[0].width;
    overlay.height = canvases[0].height;
    container.appendChild(overlay);
    toolOverlay = overlay;
    console.log("[slowargo.js] Tool overlay initialized");
}
```

### 2. Tool Exclusivity

任何时刻只能有一个自定义工具激活。激活 Smudge 时自动关闭 Clone，反之亦然。

```javascript
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
    // Clear overlay visual feedback
    const ctx = toolOverlay?.getContext('2d');
    ctx?.clearRect(0, 0, toolOverlay.width, toolOverlay.height);
}

function isAnyCustomToolActive() {
    return editorState.cloneBrush.active || editorState.smudgeBrush.active;
}
```

按钮点击时：
```javascript
cloneBtn.addEventListener("click", () => {
    const willActivate = !editorState.cloneBrush.active;
    deactivateAllCustomTools();
    if (willActivate) {
        editorState.cloneBrush.active = true;
        updateCloneStyle();
    }
    toolOverlay?.classList.toggle('active', isAnyCustomToolActive());
});

smudgeBtn.addEventListener("click", () => {
    const willActivate = !editorState.smudgeBrush.active;
    deactivateAllCustomTools();
    if (willActivate) {
        editorState.smudgeBrush.active = true;
        updateSmudgeStyle();
    }
    toolOverlay?.classList.toggle('active', isAnyCustomToolActive());
});
```

### 3. State Object

```javascript
// Add to editorState alongside cloneBrush:
smudgeBrush: {
    active: false,
    isDrawing: false,
    lastDrawX: 0,
    lastDrawY: 0,
    carriedBuffer: null,      // Float32Array — carried pixel data (RGBA per pixel)
    bufferWidth: 0,           // Width of carried buffer region
    bufferHeight: 0,          // Height of carried buffer region
    bufferOffsetX: 0,         // Canvas X of buffer top-left corner
    bufferOffsetY: 0,         // Canvas Y of buffer top-left corner
    eventsBound: false,
}
```

### 4. New Functions

| Function | Description |
|----------|-------------|
| `initSmudgeToolEvents()` | Bind pointerdown/move/up on document (capture phase) |
| `onSmudgeMouseDown(e)` | Sample composite → carriedBuffer; begin stroke |
| `onSmudgeMouseMove(e)` | Apply smudge stroke + render overlay |
| `onSmudgeMouseUp(e)` | End stroke, saveState() |
| `applySmudgeStroke(cx, cy)` | Interpolate steps from lastDraw → current, call stampSmudge per step |
| `stampSmudge(cx, cy, radius)` | Core per-stamp blending (composite read → carry blend → paint write → carry update) |
| `sampleComposite(cx, cy, radius)` | Read base+paint at region, return composited Float32Array |
| `renderSmudgeOverlay(mouseX, mouseY)` | Draw brush circle on shared overlay |
| `cleanupSmudgeTool()` | Remove event listeners, reset state, clear carriedBuffer |

### 5. Core Functions Detail

#### sampleComposite(centerX, centerY, radius)

```javascript
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

    // Use Float32Array for carried buffer to avoid rounding accumulation
    const composite = new Float32Array(w * h * 4);

    for (let i = 0; i < w * h * 4; i += 4) {
        const pa = paintData.data[i+3] / 255;  // paint alpha
        composite[i]   = paintData.data[i]   * pa + baseData.data[i]   * (1 - pa);
        composite[i+1] = paintData.data[i+1] * pa + baseData.data[i+1] * (1 - pa);
        composite[i+2] = paintData.data[i+2] * pa + baseData.data[i+2] * (1 - pa);
        composite[i+3] = Math.max(paintData.data[i+3], baseData.data[i+3]);
    }

    return { data: composite, width: w, height: h, offsetX: left, offsetY: top };
}
```

#### stampSmudge(drawX, drawY, radius)

```javascript
const SMUDGE_STRENGTH = 0.5;

function stampSmudge(drawX, drawY, radius) {
    const r = Math.ceil(radius);
    const sigma = radius * 0.4;

    // Destination region (clamped to canvas bounds)
    const dstL = Math.max(0, Math.round(drawX - r));
    const dstT = Math.max(0, Math.round(drawY - r));
    const dstR = Math.min(paintCanvas.width,  Math.round(drawX + r));
    const dstB = Math.min(paintCanvas.height, Math.round(drawY + r));
    if (dstL >= dstR || dstT >= dstB) return;

    const w = dstR - dstL;
    const h = dstB - dstT;

    // Read destination composite
    const baseCtx  = baseCanvas.getContext('2d',  { willReadFrequently: true });
    const paintCtx = paintCanvas.getContext('2d', { willReadFrequently: true });
    const baseData  = baseCtx.getImageData(dstL, dstT, w, h);
    const paintData = paintCtx.getImageData(dstL, dstT, w, h);

    const carried = editorState.smudgeBrush.carriedBuffer;
    const cb = carried.data;
    const cw = carried.width;
    const ch = carried.height;
    const cox = carried.offsetX;
    const coy = carried.offsetY;

    for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
            const canvasX = dstL + px;
            const canvasY = dstT + py;

            // Distance from brush center
            const dx = canvasX - drawX;
            const dy = canvasY - drawY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > radius) continue;

            // Gaussian weight
            const weight = Math.exp(-(dist * dist) / (2 * sigma * sigma));
            const s = SMUDGE_STRENGTH * weight;

            const di = (py * w + px) * 4;

            // Composite destination pixel
            const pa = paintData.data[di+3] / 255;
            const destR = paintData.data[di]   * pa + baseData.data[di]   * (1 - pa);
            const destG = paintData.data[di+1] * pa + baseData.data[di+1] * (1 - pa);
            const destB = paintData.data[di+2] * pa + baseData.data[di+2] * (1 - pa);
            const destA = Math.max(paintData.data[di+3], baseData.data[di+3]);

            // Map to carried buffer coordinates
            const cbx = canvasX - cox;
            const cby = canvasY - coy;

            if (cbx >= 0 && cbx < cw && cby >= 0 && cby < ch) {
                const ci = (cby * cw + cbx) * 4;

                // Blend: output = carried * s + dest * (1 - s)
                const outR = cb[ci]   * s + destR * (1 - s);
                const outG = cb[ci+1] * s + destG * (1 - s);
                const outB = cb[ci+2] * s + destB * (1 - s);
                const outA = cb[ci+3] * s + destA * (1 - s);

                // Write to paint layer
                paintData.data[di]   = Math.round(outR);
                paintData.data[di+1] = Math.round(outG);
                paintData.data[di+2] = Math.round(outB);
                paintData.data[di+3] = Math.round(outA);

                // Update carried buffer (progressive mixing)
                cb[ci]   = outR;
                cb[ci+1] = outG;
                cb[ci+2] = outB;
                cb[ci+3] = outA;
            }
        }
    }

    paintCtx.putImageData(paintData, dstL, dstT);

    // Update carried buffer position to track brush movement
    carried.offsetX = dstL;
    carried.offsetY = dstT;
}
```

### 6. UI — Toolbar Button

在 `addFastForwardToggleButton()` 中，Clone 按钮之后添加 Smudge 按钮：

```
┌─────────────────────────────────────────────────────────┐
│ [FF] [↺Mask] [↺All] [⊕Clone] [☝Smudge] [👁]  │
└─────────────────────────────────────────────────────────┘
```

- 图标：`pi pi-arrow-right-arrow-left`（或其他合适的 PrimeIcons 图标）
- 文本：`Smudge`
- 样式：复用 `.fast-forward-mode-toggle` class（与 Clone 一致）
- 点击行为：toggle smudge active 状态 + 互斥关闭 clone

### 7. Keyboard Shortcut Integration

Smudge 工具复用 `[` / `]` 键调整笔刷大小（与 Clone Brush 共享逻辑，通过 `getCloneBrushRadius()` / 原生 slider 同步）。需要将 `[`/`]` 的响应条件从 `editorState.cloneBrush.active` 扩展为 `isAnyCustomToolActive()`。

### 8. Visual Feedback

Smudge 比 Clone 简单得多——只需要画笔刷圆圈（白色描边），无需十字准星或虚线。

```javascript
function renderSmudgeOverlay(mouseX, mouseY) {
    if (!toolOverlay) return;
    const ctx = toolOverlay.getContext('2d');
    ctx.clearRect(0, 0, toolOverlay.width, toolOverlay.height);

    const radius = getCloneBrushRadius();

    // Brush circle
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 200, 100, 0.8)';  // Orange tint to distinguish from clone
    ctx.lineWidth = 1.5;
    ctx.stroke();
}
```

## Files to Modify

| File | Changes |
|------|---------|
| `js/maskEditorTurbo.js` | 主要改动：overlay 重构、smudge 状态/函数/事件/UI/清理 |
| `js/maskEditorTurbo.css` | 更新 `#clone-brush-overlay` → `#tool-overlay` 选择器 |

## Implementation Steps

1. **Refactor Overlay**：提取共享 overlay 和 canvas 引用到模块级变量，重命名 overlay ID
2. **Update Clone Brush**：将 clone brush 中对 `editorState.cloneBrush.overlay/baseCanvas/paintCanvas` 的引用改为模块级变量
3. **Add State**：在 `editorState` 中增加 `smudgeBrush` 对象
4. **Implement Core Algorithm**：`sampleComposite()` + `stampSmudge()` + `applySmudgeStroke()`
5. **Add Event Handlers**：`onSmudgeMouseDown/Move/Up` + `initSmudgeToolEvents()`
6. **Add Visual Feedback**：`renderSmudgeOverlay()` 画笔刷圆圈
7. **Add UI Button**：在 toolbar 中添加 Smudge 按钮 + 互斥逻辑 + `deactivateAllCustomTools()`
8. **Update Keyboard Shortcuts**：扩展 `[`/`]` 响应条件为 `isAnyCustomToolActive()`
9. **Add Cleanup**：`cleanupSmudgeTool()` + 集成到编辑器关闭流程中的 `onEditorReady()` 和 observer cleanup
10. **Update CSS**：`#clone-brush-overlay` → `#tool-overlay`

## Verification

1. 打开 Mask Editor，确认工具栏出现 Smudge 按钮
2. 点击 Smudge 按钮，确认激活（蓝色高亮）且 Clone 自动关闭
3. 在已绘制区域拖动，确认像素被涂抹/拖拽
4. 在 Base 图层可见区域拖动，确认能涂抹原始图像内容到 Paint 层
5. 测试 `[`/`]` 键调整笔刷大小
6. 测试 Ctrl+Z 撤销一整个涂抹笔触
7. 确认点击 Clone 按钮后 Smudge 自动关闭
8. 确认关闭编辑器后资源正确清理（无内存泄漏）
