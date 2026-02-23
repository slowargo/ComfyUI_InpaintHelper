# Clone Brush (克隆笔刷) 设计方案

## 1. 功能概述

在 Mask Editor 中实现仿制图章工具（Clone Stamp），允许用户：
- 按住 Alt 键取样
- 在其他位置绘制复制的像素
- 实时预览取样位置和画笔效果

## 2. 交互流程

```
1. 点击工具栏 "Clone" 按钮激活工具
   ↓ 自动切换到 Pen Tool
2. Alt + 点击画布 → 设置取样点（显示红色十字 ⓧ）
   ↓
3. 松开 Alt，拖动鼠标 → 从取样点复制像素
   ↓ 实时显示：取样源位置 + 画笔预览
4. 点击 Mask/Pen/Clone 按钮 或 关闭编辑器 → 退出
```

## 3. UI 布局

### 3.1 按钮顺序
```
[FF] [Mask] [All] [Clone] [Blur]
           ↑
        新增按钮
```

### 3.2 实时预览元素

```
取样点: ⓧ (红色十字标记，固定在取样位置)
         ↓
        连线 (虚线连接取样点和画笔中心)
         ↓
画笔中心: ◯ (显示当前取样源的预览)
         ↓
       [实际绘制到 paint layer]
```

## 4. 核心实现

### 4.1 状态管理 (`maskEditorTurbo.js`)

```javascript
const cloneState = {
    active: false,              // 是否激活 Clone 工具
    hasSample: false,           // 是否已设置取样点
    sampleX: 0, sampleY: 0,     // 原始取样点坐标
    lastDrawX: 0, lastDrawY: 0, // 上次绘制位置（用于计算偏移）
    sampleSize: 20,             // 取样时的画笔大小
    sampleCanvas: null,         // 取样像素缓存（复用）
    previewCanvas: null,        // 实时预览层
    isDrawing: false,           // 是否正在绘制
    eventsBound: false,         // 事件是否已绑定（防止重复添加）
    altPressed: false,          // Alt 键状态
    brushSettings: null,        // 缓存的画笔设置
};
```

### 4.2 关键函数

**activateCloneTool()**
- 检查 `eventsBound`，防止重复添加事件监听
- 设置 `cloneState.active = true`
- 自动切换到 Pen Tool（通过点击 pen tool 按钮）
- 创建预览层 canvas，同步尺寸
- 监听窗口 resize，自动调整预览层大小
- 显示取样点标记（如果有）

**deactivateCloneTool()**
- 如果 `isDrawing=true`，先结束当前绘制并保存状态
- 设置 `cloneState.active = false`
- 清除 `hasSample`
- 移除预览层，清理内存
- 移除 resize 监听
- 隐藏取样点标记

**setCloneSample(x, y)**
- 边界检查：确保取样区域不越界
- 从 base layer (canvas[0]) 捕获像素到 `sampleCanvas`
- 记录实际取样中心点（可能偏离点击位置）
- 记录当前画笔大小到 `sampleSize`
- 初始化 `lastDrawX/Y = sampleX/Y`
- 显示红色十字标记

**drawCloneBrush(x, y)**
- 检查画笔大小是否变更，如变更则重新取样或缩放
- 使用线段插值，在起点和终点之间均匀绘制
- 计算取样偏移：跟随画笔移动（对齐模式）
- 应用硬度遮罩实现软硬边缘
- 边界检查，限制绘制在画布内
- 绘制到 paint layer (canvas[1])
- 更新 `lastDrawX/Y`
- 同步 GPU 状态

**updatePreview(x, y)**
- 使用 `requestAnimationFrame` 节流
- 更新画笔中心预览（显示当前取样源内容）
- 更新连线位置
- 无需保存历史（预览不写入）

**getCanvasCoordinates(e, canvas)**
- 处理 CSS 缩放
- 处理 devicePixelRatio
- 返回正确的 canvas 像素坐标

### 4.3 事件处理（Capture 阶段拦截）

```javascript
// 在 canvas container 上 capture 阶段监听，阻止 Pen Tool 事件
function initCloneToolEvents() {
    if (cloneState.eventsBound) return; // 防止重复绑定

    const container = document.querySelector('#maskEditorCanvasContainer');
    container.addEventListener('mousedown', onCloneMouseDown, true);
    container.addEventListener('mousemove', onCloneMouseMove, true);
    container.addEventListener('mouseup', onCloneMouseUp, true);

    // 窗口失焦时重置 Alt 状态
    window.addEventListener('blur', onWindowBlur);

    cloneState.eventsBound = true;
}

function onWindowBlur() {
    cloneState.altPressed = false;
}

function onCloneMouseDown(e) {
    if (!cloneState.active || !isPenToolActive()) return;

    e.stopImmediatePropagation(); // 阻止 Pen Tool 处理
    e.preventDefault(); // 阻止浏览器菜单（Firefox）

    const coords = getCanvasCoordinates(e, getPaintLayerCanvas());

    if (e.altKey) {
        setCloneSample(coords.x, coords.y);
    } else if (cloneState.hasSample) {
        cloneState.isDrawing = true;
        cloneState.brushSettings = getBrushSettings(); // 缓存画笔设置
        cloneState.lastDrawX = coords.x;
        cloneState.lastDrawY = coords.y;
        saveCanvasState(); // Undo 起点
        drawCloneBrush(coords.x, coords.y); // 立即绘制一个点
    }
}

function onCloneMouseMove(e) {
    if (!cloneState.active || !isPenToolActive()) return;

    const coords = getCanvasCoordinates(e, getPaintLayerCanvas());

    if (cloneState.isDrawing) {
        e.stopImmediatePropagation();
        drawCloneBrush(coords.x, coords.y);
    } else if (cloneState.hasSample) {
        // 仅更新预览，不阻止事件
        updatePreview(coords.x, coords.y);
    }
}

function onCloneMouseUp(e) {
    if (!cloneState.active || !cloneState.isDrawing) return;

    e.stopImmediatePropagation();
    cloneState.isDrawing = false;
    saveCanvasState(); // Undo 终点
}
```

### 4.4 实时预览实现

```javascript
function createPreviewLayer() {
    const previewCanvas = document.createElement('canvas');
    previewCanvas.className = 'clone-preview-layer';
    previewCanvas.style.position = 'absolute';
    previewCanvas.style.top = '0';
    previewCanvas.style.left = '0';
    previewCanvas.style.pointerEvents = 'none';
    previewCanvas.style.zIndex = '50';

    // 同步尺寸
    syncPreviewCanvasSize(previewCanvas);

    // 监听尺寸变化
    const container = document.querySelector('#maskEditorCanvasContainer');
    cloneState.resizeObserver = new ResizeObserver(() => {
        syncPreviewCanvasSize(previewCanvas);
    });
    cloneState.resizeObserver.observe(container);

    container.appendChild(previewCanvas);
    return previewCanvas;
}

function syncPreviewCanvasSize(previewCanvas) {
    const container = document.querySelector('#maskEditorCanvasContainer');
    const mainCanvas = container.querySelector('canvas');
    if (mainCanvas && previewCanvas) {
        previewCanvas.width = mainCanvas.width;
        previewCanvas.height = mainCanvas.height;
    }
}

let previewRafId = null;
function updatePreview(mouseX, mouseY) {
    if (!cloneState.hasSample || !cloneState.previewCanvas) return;
    if (previewRafId) return; // 节流

    previewRafId = requestAnimationFrame(() => {
        renderPreview(mouseX, mouseY);
        previewRafId = null;
    });
}

function renderPreview(mouseX, mouseY) {
    const ctx = cloneState.previewCanvas.getContext('2d');
    const { size } = getBrushSettings();

    ctx.clearRect(0, 0, cloneState.previewCanvas.width, cloneState.previewCanvas.height);

    // 1. 绘制取样点标记（红色十字）
    drawSampleMarker(ctx, cloneState.sampleX, cloneState.sampleY);

    // 2. 绘制连线（取样点 -> 画笔中心）
    drawDashedLine(ctx,
        cloneState.sampleX, cloneState.sampleY,
        mouseX, mouseY
    );

    // 3. 绘制画笔中心预览圆圈
    const offsetX = mouseX - cloneState.lastDrawX;
    const offsetY = mouseY - cloneState.lastDrawY;
    const srcX = cloneState.sampleX + offsetX;
    const srcY = cloneState.sampleY + offsetY;

    ctx.save();
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, size/2, 0, Math.PI * 2);
    ctx.clip();

    ctx.drawImage(cloneState.sampleCanvas,
        0, 0, size, size,
        mouseX - size/2, mouseY - size/2, size, size);

    // 绘制圆圈边框
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.restore();
}
```

### 4.5 绘制实现

```javascript
function drawCloneBrush(x, y) {
    if (!cloneState.hasSample) return;

    const { size, hardness, opacity } = cloneState.brushSettings || getBrushSettings();

    // 画笔大小变更处理
    if (size !== cloneState.sampleSize) {
        rescaleSampleCanvas(size);
        cloneState.sampleSize = size;
    }

    const paintCanvas = getPaintLayerCanvas();
    const ctx = paintCanvas.getContext('2d', { willReadFrequently: true });

    // 线段插值
    const { lastDrawX, lastDrawY } = cloneState;
    const dist = Math.hypot(x - lastDrawX, y - lastDrawY);
    const step = size / 4;

    ctx.globalAlpha = opacity;

    for (let d = 0; d < dist; d += step) {
        const t = d / dist;
        const ix = lastDrawX + (x - lastDrawX) * t;
        const iy = lastDrawY + (y - lastDrawY) * t;
        drawStamp(ctx, ix, iy, size, hardness);
    }

    drawStamp(ctx, x, y, size, hardness);

    cloneState.lastDrawX = x;
    cloneState.lastDrawY = y;

    // GPU 同步
    const store = getMaskEditorStore();
    if (store?.canvasHistory?.saveState) {
        store.canvasHistory.saveState();
    }
}

function drawStamp(ctx, x, y, size, hardness) {
    const canvas = ctx.canvas;
    const half = size / 2;

    // 边界限制
    const startX = Math.max(0, x - half);
    const startY = Math.max(0, y - half);
    const endX = Math.min(canvas.width, x + half);
    const endY = Math.min(canvas.height, y + half);

    if (startX >= endX || startY >= endY) return;

    // 计算取样偏移
    const offsetX = x - cloneState.sampleX;
    const offsetY = y - cloneState.sampleY;

    // 应用硬度遮罩
    ctx.save();

    // 创建临时 canvas 应用硬度
    if (hardness < 1) {
        const maskCanvas = createHardnessMask(size, hardness);
        ctx.globalCompositeOperation = 'source-over';
        // 使用 mask...
    }

    ctx.drawImage(cloneState.sampleCanvas,
        startX - x + half, startY - y + half, endX - startX, endY - startY,
        startX, startY, endX - startX, endY - startY
    );

    ctx.restore();
}

function rescaleSampleCanvas(newSize) {
    if (!cloneState.sampleCanvas) return;

    const oldCanvas = cloneState.sampleCanvas;
    const newCanvas = document.createElement('canvas');
    newCanvas.width = newSize;
    newCanvas.height = newSize;

    const ctx = newCanvas.getContext('2d');
    ctx.drawImage(oldCanvas, 0, 0, newSize, newSize);

    cloneState.sampleCanvas = newCanvas;
}
```

### 4.6 坐标转换

```javascript
function getCanvasCoordinates(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // 考虑 CSS 缩放和 devicePixelRatio
    const scaleX = (canvas.width / dpr) / rect.width;
    const scaleY = (canvas.height / dpr) / rect.height;

    return {
        x: (e.clientX - rect.left) * scaleX * dpr,
        y: (e.clientY - rect.top) * scaleY * dpr
    };
}
```

### 4.7 画笔设置获取

```javascript
function getBrushSettings() {
    const store = getMaskEditorStore();

    // 多层防御，兼容不同 ComfyUI 版本
    return {
        size: store?.brushSize ?? store?.brush?.size ?? 20,
        hardness: store?.brushHardness ?? store?.brush?.hardness ?? 0.5,
        opacity: store?.brushOpacity ?? store?.brush?.opacity ?? 1.0
    };
}
```

### 4.8 清理函数

```javascript
function cleanupCloneTool() {
    // 结束当前绘制
    if (cloneState.isDrawing) {
        saveCanvasState();
        cloneState.isDrawing = false;
    }

    // 移除事件监听
    if (cloneState.eventsBound) {
        const container = document.querySelector('#maskEditorCanvasContainer');
        container.removeEventListener('mousedown', onCloneMouseDown, true);
        container.removeEventListener('mousemove', onCloneMouseMove, true);
        container.removeEventListener('mouseup', onCloneMouseUp, true);
        window.removeEventListener('blur', onWindowBlur);
        cloneState.eventsBound = false;
    }

    // 移除 resize 监听
    if (cloneState.resizeObserver) {
        cloneState.resizeObserver.disconnect();
        cloneState.resizeObserver = null;
    }

    // 移除预览层
    if (cloneState.previewCanvas) {
        cloneState.previewCanvas.remove();
        cloneState.previewCanvas = null;
    }

    // 释放 canvas 内存
    if (cloneState.sampleCanvas) {
        cloneState.sampleCanvas.width = 0;
        cloneState.sampleCanvas.height = 0;
        cloneState.sampleCanvas = null;
    }

    cloneState.active = false;
    cloneState.hasSample = false;
    cloneState.brushSettings = null;
}
```

## 5. 关键问题与解决方案

### 5.1 事件劫持 → 事件拦截

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 无法劫持 Vue 组件内部事件 | Pen Tool 是 ComfyUI 内部 Vue 组件 | 使用 `capture: true` 在顶层拦截，调用 `stopImmediatePropagation()` |
| 事件监听器重复添加 | 多次激活工具 | 使用 `eventsBound` 标志位检查 |

### 5.2 浏览器兼容性

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Alt 键触发浏览器菜单 | Firefox 默认行为 | `e.preventDefault()` 阻止默认行为 |
| Alt+Tab 后状态错乱 | 窗口失焦时 Alt 未释放 | `window.addEventListener('blur')` 重置状态 |

### 5.3 边界处理

```javascript
function setCloneSample(x, y) {
    const { size } = getBrushSettings();
    const half = size / 2;
    const baseCanvas = getBaseLayerCanvas();

    // 计算安全的取样区域
    const srcX = Math.max(0, Math.min(x - half, baseCanvas.width - size));
    const srcY = Math.max(0, Math.min(y - half, baseCanvas.height - size));

    // 记录实际取样中心点（可能偏离点击位置）
    cloneState.sampleX = srcX + half;
    cloneState.sampleY = srcY + half;
    cloneState.lastDrawX = cloneState.sampleX;
    cloneState.lastDrawY = cloneState.sampleY;
    cloneState.sampleSize = size;

    // 捕获像素
    captureSampleArea(srcX, srcY, size);
}
```

### 5.4 GPU 同步

每次绘制操作后必须同步 GPU：

```javascript
function afterDraw() {
    const store = getMaskEditorStore();
    if (store?.canvasHistory?.saveState) {
        store.canvasHistory.saveState();

        // 可选：限制历史深度防止内存泄漏
        if (store.canvasHistory.states?.length > 20) {
            store.canvasHistory.states.shift();
        }
    }
}
```

### 5.5 性能优化

| 优化项 | 实现 |
|--------|------|
| 预览节流 | 使用 `requestAnimationFrame` |
| 绘制插值 | 在移动距离内均匀分布绘制点 |
| 画笔设置缓存 | 在 `mouseDown` 时缓存，避免每帧读取 store |
| Canvas 复用 | 复用 `sampleCanvas`，变更大小时重新采样 |

### 5.6 高 DPI 支持

```javascript
function getCanvasCoordinates(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const scaleX = (canvas.width / dpr) / rect.width;
    const scaleY = (canvas.height / dpr) / rect.height;

    return {
        x: (e.clientX - rect.left) * scaleX * dpr,
        y: (e.clientY - rect.top) * scaleY * dpr
    };
}
```

### 5.7 Blur Mode 兼容

Blur Mode 下编辑器模糊且偏移，预览层需要同步：

```javascript
function updatePreviewPosition() {
    if (!cloneState.previewCanvas) return;

    if (editorState.isBlurred) {
        cloneState.previewCanvas.style.transform = 'translateX(-80%)';
        cloneState.previewCanvas.style.filter = 'blur(1px)';
    } else {
        cloneState.previewCanvas.style.transform = '';
        cloneState.previewCanvas.style.filter = '';
    }
}
```

## 6. Undo/Redo 支持

| 时机 | 操作 |
|------|------|
| mousedown (开始绘制) | `saveCanvasState()` - 保存操作前状态 |
| mouseup (结束绘制) | `saveCanvasState()` - 保存操作后状态 |

效果：一次拖动操作 = 一个 Undo 单元

## 7. 图层操作

| 操作 | Canvas | 索引 | 说明 |
|------|--------|------|------|
| 取样源 | base layer | 0 | 读取原始图像像素 |
| 绘制目标 | paint layer | 1 | 写入克隆结果 |
| 预览层 | preview layer | - | 新建的 overlay canvas |

## 8. CSS 样式 (`maskEditorTurbo.css`)

```css
/* Clone 按钮 */
.clone-tool-button {
    background: rgba(30, 144, 255, 0.8);
    color: white;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    cursor: pointer;
    font-size: 11px;
    font-weight: bold;
    transition: all 0.2s;
}

.clone-tool-button:hover {
    background: rgba(30, 144, 255, 1);
}

.clone-tool-button.active {
    background: rgba(30, 144, 255, 1);
    box-shadow: 0 0 8px rgba(30, 144, 255, 0.8);
}

/* 取样点标记 */
.clone-sample-marker {
    position: absolute;
    width: 20px;
    height: 20px;
    pointer-events: none;
    z-index: 51;
}

.clone-sample-marker::before,
.clone-sample-marker::after {
    content: '';
    position: absolute;
    background: red;
}

.clone-sample-marker::before {
    width: 100%;
    height: 2px;
    top: 50%;
    left: 0;
    transform: translateY(-50%);
}

.clone-sample-marker::after {
    width: 2px;
    height: 100%;
    left: 50%;
    top: 0;
    transform: translateX(-50%);
}

/* 预览层 */
.clone-preview-layer {
    pointer-events: none;
}

/* Blur Mode 适配 */
.mask-editor-dialog.editor-blurred .clone-preview-layer {
    transform: translateX(-80%);
    filter: blur(1px);
    transition: transform 0.3s ease, filter 0.3s ease;
}
```

## 9. 实现文件

- `js/maskEditorTurbo.js` - 核心逻辑
- `js/maskEditorTurbo.css` - 样式

## 10. 退出条件

- 点击 Mask Tool 按钮
- 点击 Pen Tool 按钮
- 再次点击 Clone 按钮
- 关闭 Mask Editor（自动调用 `cleanupCloneTool`）

## 11. 潜在问题汇总

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| 事件监听器重复添加 | ✅ 已解决 | `eventsBound` 标志位检查 |
| Alt 键状态丢失 | ✅ 已解决 | `window.addEventListener('blur')` 重置 |
| 快速点击不绘制 | ✅ 已解决 | `mouseDown` 立即绘制一个点 |
| 预览层尺寸不匹配 | ✅ 已解决 | `ResizeObserver` 自动同步 |
| 画笔大小变更 | ✅ 已解决 | `rescaleSampleCanvas` 重新采样 |
| GPU 内存泄漏 | ✅ 已解决 | 限制历史记录深度 |
| 边界外绘制 | ✅ 已解决 | `Math.max/min` 限制绘制区域 |
| 绘制中切换工具 | ✅ 已解决 | `deactivateCloneTool` 先结束绘制 |
| Blur Mode 冲突 | ✅ 已解决 | 同步 CSS transform/filter |
| Retina/DPI 屏幕 | ✅ 已解决 | `devicePixelRatio` 处理 |
| 坐标系变换 | ✅ 已解决 | `getCanvasCoordinates` 统一处理 |
| Store API 兼容 | ✅ 已解决 | 多层防御性读取 |
| 浏览器菜单快捷键 | ✅ 已解决 | `e.preventDefault()` |
| 性能（高频绘制） | ✅ 已解决 | `requestAnimationFrame` + 线段插值 |

---

*设计方案版本: 3.0*
*更新日期: 2026-02-22*
