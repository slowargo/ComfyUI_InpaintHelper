# Transform Tool 设计文档

## 概述

Transform 是一个 paint 层内容变换工具，支持框选 paint 层区域并进行透视变形、拉伸、旋转和移动操作。

## 核心特点

- **单一入口**：仅 Q 键激活，无模式切换
- **智能取样**：paint 层有内容直接变换；空白则自动从 base 层复制
- **即时应用**：无 Enter 确认，每次操作直接更新 paint 层
- **单选区**：同时只维护一个选区，新建选区时自动丢弃旧选区

## 交互流程

```
按 Q 激活工具
    │
    ▼
┌─────────────────────────────────────┐
│           无选区状态                 │
│      （base 层正常，等待框选）        │
└──────────────┬──────────────────────┘
               │ 鼠标按下（在选区外）
               ▼
        开始框选新选区
        base 层变暗（50% 透明度）
        显示动态虚线框
               │
               ▼ 鼠标松开
        检测选区内容
               │
               ├─ paint 层有内容 ──► 进入 Transform 模式
               │
               └─ paint 层空白 ────► 从 base 自动复制
                        Toast: "Empty selection, auto-copied from base layer"
                        进入 Transform 模式
               │
               ▼
┌─────────────────────────────────────┐
│          Transform 模式              │
│  显示四角句柄、边线、旋转手柄         │
│  可拖动进行各种变换                  │
└──────────────┬──────────────────────┘
               │
    ├─ 在选区内按下 ────┤
    ├─ 在句柄上按下 ────┤  进入拖动状态
    │                   │  MouseUp 时应用变换
    │                   │  直接更新 paint 层
    │                   │
    ▼                   ▼
  拖动移动          拖动四角/边/旋转
  （改变位置）      （变形/拉伸/旋转）
               │
               ▼ 鼠标在选区外按下
        放弃当前选区
        回到"无选区状态"
        （已应用的变换保留在 paint 层）
```

## 状态管理

```javascript
transformTool: {
    // 当前阶段
    stage: 'idle' | 'selecting' | 'transforming',

    // 选区数据（仅在 selecting/transforming 时存在）
    selection: {
        // 选区矩形（画布像素坐标，始终是原始框选区域）
        rect: { x, y, width, height },

        // 源图像缓存（框选时从 paint 层"剪切"的像素）
        // 使用 document.createElement('canvas') 创建，兼容性优于 OffscreenCanvas
        sourceCanvas: HTMLCanvasElement,

        // 是否从 base 层自动复制（用于撤销时判断是否需要清除 paint 层）
        isFromBase: boolean,

        // 上一次 applyTransform 写入 paint 层的区域（用于下次应用前清除残影）
        lastAppliedBounds: { x, y, width, height } | null,

        // paint 层在 lastAppliedBounds 区域的原始快照（用于清除变换残影时恢复底层像素，
        // 避免 clearPaintRect 误伤该区域内的其他绘制内容）
        paintSnapshot: ImageData | null,

        // 是否已应用过至少一次变换（用于放弃选区时决定是否还原像素）
        hasTransformed: boolean,
    },

    // 变换状态（仅在 transforming 时存在）
    transform: {
        // 四角最终屏幕坐标，所有操作（包括旋转）直接修改此数组
        corners: [
            { x, y }, // 左上（索引 0）
            { x, y }, // 右上（索引 1）
            { x, y }, // 右下（索引 2）
            { x, y }  // 左下（索引 3）
        ],
    },

    // 拖动状态
    drag: {
        active: boolean,
        type: 'move' | 'corner' | 'edge' | 'rotate',
        targetIndex: number,      // 角点索引(0-3)或边索引(0-3)
        startMouse: { x, y },     // 拖动起始鼠标位置
        startTransform: {         // 拖动开始时的变换状态快照
            corners: [...]
        }
    },

    // 可复用的临时画布（applyTransform 渲染用，避免每次 mouseUp 都创建/销毁）
    tempCanvas: HTMLCanvasElement | null,
}
```

## 鼠标按下检测逻辑

句柄检测的距离阈值均以**屏幕像素（screen space）**为基准。检测前需将鼠标位置和句柄位置
都转换到屏幕坐标后再计算距离，确保在画布缩放/平移状态下手感一致。

```javascript
function onPointerDown(e) {
    const pos = getCanvasPos(e);

    if (state.stage === 'transforming') {
        // 1. 检测是否在选区内部
        if (isPointInQuad(pos, state.transform.corners)) {
            startDrag('move', pos);
            return;
        }

        // 2. 检测是否在手柄上
        const handle = detectHandle(pos, state.transform);
        if (handle) {
            startDrag(handle.type, pos, handle.index);
            return;
        }

        // 3. 在选区外：放弃旧选区，开始新框选
        clearSelection();  // 清空 overlay，保留 paint 层内容
        state.stage = 'selecting';
        startSelection(pos);
    } else {
        // 无选区状态：开始框选
        state.stage = 'selecting';
        startSelection(pos);
    }
}
```

## 句柄检测

### 优先级

所有距离阈值均基于**屏幕像素**，不受画布缩放影响。

1. **四角**（最高优先级）- 距离阈值 15px（屏幕像素）
   - 视觉：8px 方形，悬停 10px
   - 操作：透视变形（任意移动角点）
   - 光标：`nwse-resize` / `nesw-resize`（根据角点位置动态选择）

2. **旋转手柄** - 距离阈值 10px（屏幕像素）
   - 位置：顶边中心上方 30px
   - 视觉：6px 圆形 + 连接线
   - 操作：绕中心旋转
   - 光标：`grab`，拖动中 `grabbing`

3. **四边** - 点到线段距离阈值 10px（屏幕像素）
   - 视觉：无边框，悬停时光标变化
   - 操作：单轴拉伸（Shift = 等比缩放）
   - 光标：`ns-resize` / `ew-resize`（根据边方向）

4. **选区内部**（最低优先级）
   - 射线法点包含检测
   - 操作：移动位置
   - 光标：`move`

### 框选阶段光标

- 框选中：`crosshair`

## 框选阶段

### 视觉反馈

```javascript
function renderSelecting(ctx, startPos, currentPos) {
    const rect = normalizeRect(startPos, currentPos);

    // 清空 overlay
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // 绘制虚线选区框
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = 1;
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);

    // 半透明填充
    ctx.fillStyle = 'rgba(74, 158, 255, 0.15)';
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
}
```

### Base 层变暗

```javascript
function setBaseLayerDimmed(dimmed) {
    if (dimmed) {
        baseCanvas.style.opacity = '0.4';
        // 或添加遮罩层
        dimOverlay.style.display = 'block';
    } else {
        baseCanvas.style.opacity = '1';
        dimOverlay.style.display = 'none';
    }
}
```

### 完成框选

最小选区尺寸：width 和 height 均须 ≥ 5px，否则视为误触，直接取消本次框选。

```javascript
function finalizeSelection() {
    const rect = getSelectionRect();

    // 过滤过小的选区
    if (rect.width < 5 || rect.height < 5) {
        cancelSelection();
        return;
    }

    // 从 paint 层检测选区内容（比例阈值：不透明像素数 < 选区面积的 0.5% 或少于 16 个则视为空）
    const paintData = samplePaintLayer(rect);
    const minPixels = Math.max(16, Math.floor(rect.width * rect.height * 0.005));
    const isEmpty = isImageDataEmpty(paintData, minPixels);

    if (isEmpty) {
        // 从 base 层检测选区内容（若 base 层也为空则取消选区）
        const baseData = sampleBaseLayer(rect);
        const isBaseEmpty = isImageDataEmpty(baseData, minPixels);
        if (isBaseEmpty) {
            showToast('Both layers are empty in this area', { duration: 2000 });
            cancelSelection();
            return;
        }

        // 从 base 层复制到 paint 层，作为变换源（复用上面已采样的 baseData）
        copyToPaintLayer(rect, baseData);

        // 从 paint 层剪切像素到 sourceCanvas（此时 paint 层写入的是 base 数据）
        state.selection.sourceCanvas = createSourceCanvas(baseData);
        state.selection.isFromBase = true;
        state.selection.paintSnapshot = null;

        // Toast 提示
        showToast('Empty selection, auto-copied from base layer (Ctrl+Z to undo)', {
            duration: 3000
        });
    } else {
        // 将 paint 层选区像素剪切到 sourceCanvas
        state.selection.sourceCanvas = createSourceCanvas(paintData);
        state.selection.isFromBase = false;
        state.selection.paintSnapshot = null;
    }

    // 剪切：从 paint 层清除选区像素（源已保存到 sourceCanvas）
    clearPaintRect(rect);

    state.selection.rect = rect;
    state.selection.lastAppliedBounds = null;
    state.selection.hasTransformed = false;

    // 初始化变换状态（默认矩形，corners 即为最终屏幕坐标）
    state.transform = {
        corners: [
            { x: rect.x, y: rect.y },
            { x: rect.x + rect.width, y: rect.y },
            { x: rect.x + rect.width, y: rect.y + rect.height },
            { x: rect.x, y: rect.y + rect.height }
        ],
    };

    // 恢复 base 层亮度
    setBaseLayerDimmed(false);

    // 进入 transform 模式
    state.stage = 'transforming';
}
```

## Transform 模式

### 渲染（非实时透视）

```javascript
function renderTransforming(ctx, state) {
    const { selection, transform } = state;

    // 清空 overlay
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // corners 即为最终屏幕坐标，旋转操作已直接修改 corners，无需额外变换
    const corners = transform.corners;

    // 绘制变换后的图像（仅在非拖动状态或移动操作时实时渲染像素）
    if (!state.drag.active || state.drag.type === 'move') {
        // 移动操作可以实时渲染
        drawTransformedImage(ctx, selection.sourceCanvas, corners);
    } else {
        // 透视变形/旋转/缩放拖动中：仅绘制边框轮廓，MouseUp 时再渲染像素
        drawTransformOutline(ctx, corners);
    }

    // 绘制句柄
    drawHandles(ctx, corners, state.drag);
}

function drawTransformOutline(ctx, corners) {
    // 绘制四边形边框
    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < 4; i++) {
        ctx.lineTo(corners[i].x, corners[i].y);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);
}
```

### 应用变换

```javascript
function applyTransform() {
    const { selection, transform } = state;
    const corners = transform.corners;

    // 1. 清除上一次变换写入的区域（使用快照恢复，避免误伤该区域内的其他绘制内容）
    if (selection.lastAppliedBounds && selection.paintSnapshot) {
        paintCtx.putImageData(selection.paintSnapshot,
            selection.lastAppliedBounds.x, selection.lastAppliedBounds.y);
    }

    // 2. 计算本次变换的 bounding box
    const bounds = getBounds(corners);

    // 3. 保存 paint 层在新 bounds 区域的快照（在写入变换结果之前）
    selection.paintSnapshot = paintCtx.getImageData(
        bounds.x, bounds.y, bounds.width, bounds.height);

    // 4. 渲染到临时画布（复用 tempCanvas，大小仅覆盖 bounds 区域）
    if (!state.tempCanvas) {
        state.tempCanvas = document.createElement('canvas');
    }
    state.tempCanvas.width = bounds.width;
    state.tempCanvas.height = bounds.height;
    const tempCtx = state.tempCanvas.getContext('2d');
    tempCtx.clearRect(0, 0, bounds.width, bounds.height);

    // 将 corners 偏移到 tempCanvas 的本地坐标系
    const localCorners = corners.map(c => ({
        x: c.x - bounds.x,
        y: c.y - bounds.y
    }));
    const localSrcCorners = getSourceCorners(selection.rect);

    drawPerspectiveQuad(tempCtx, selection.sourceCanvas, localSrcCorners, localCorners);

    // 5. 合成到 paint 层
    paintCtx.drawImage(state.tempCanvas, bounds.x, bounds.y);

    // 6. 记录本次写入区域，供下次应用前恢复
    selection.lastAppliedBounds = bounds;
    selection.hasTransformed = true;
}

function onMouseUp(e) {
    if (state.drag.active) {
        // 应用变换
        applyTransform();

        // 重置拖动状态
        state.drag.active = false;

        // 重新渲染（显示最终图像）
        renderTransforming(overlayCtx, state);
        
        // 调用 saveState() 保存历史
        // ...
    }
}
```

## 选区放弃逻辑

```javascript
function clearSelection() {
    const { selection } = state;

    if (selection) {
        if (!selection.hasTransformed) {
            // 用户框选后未做任何变换就放弃——还原像素到 paint 层原位
            paintCtx.drawImage(selection.sourceCanvas,
                selection.rect.x, selection.rect.y);
        }
        // 保存历史（仅在确实发生过变换时才有意义）
        if (selection.hasTransformed) {
            getMaskEditorStore()?.canvasHistory?.saveState?.();
        }
    }

    // 清空 overlay
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    // 恢复 base 层亮度
    setBaseLayerDimmed(false);

    // 重置状态（保留 paint 层已应用的内容）
    state.stage = 'idle';
    state.selection = null;
    state.transform = null;
    state.drag = { active: false };
}
```

## 性能优化

1. **透视渲染延迟**
   - 拖动四角/边/旋转时：仅绘制边框轮廓
   - MouseUp 时：计算并应用完整透视变换（统一用网格细分，不区分仿射/透视）

2. **移动操作实时**
   - 拖动选区内部移动：可以实时渲染（使用 drawImage）

3. **缓存源图像**
   - 框选完成后立即创建 `HTMLCanvasElement` 缓存（`document.createElement('canvas')`）
   - 避免每次渲染都调用 `getImageData`

4. **网格细分**
   - 透视变换使用 20x20 网格细分
   - 平衡质量与性能

5. **临时画布复用**
   - `applyTransform` 使用挂在 `state.tempCanvas` 上的可复用画布
   - 大小仅覆盖 corners 的 bounding box，非全画布大小
   - 避免每次 mouseUp 都创建/销毁临时 canvas

6. **快照恢复代替全清除**
   - 使用 `paintSnapshot`（ImageData）恢复上一次变换区域的底层像素
   - 比 `clearPaintRect` 更精准，且不会误伤该区域内的其他绘制内容

## 退出方式

与其他 brush tools 一致：

| 操作 | 结果 |
|------|------|
| 按 Escape | 退出工具，保留已应用的变换，清空 overlay |
| 按 Q | 切换工具开关（on/off） |
| 切换其他工具（Clone/Smudge） | 退出 Transform，激活其他工具 |
| 关闭 Mask Editor | 自动退出 |

## 键盘快捷键

| 按键 | 功能 |
|------|------|
| `Q` | 激活/关闭 Transform 工具 |
| `Shift`（按住） | 边拖动时约束为等比缩放 |
| `Escape` | 退出工具 |
| `Ctrl+Z` | 撤销（调用编辑器内置撤销） |
其他未提及的按键与其他工具行为保持一致

### 撤销粒度说明

与 Clone/Smudge 的逐步撤销（每次 mouseUp 保存）相同

## 与现有工具的整合

Transform 工具需接入 `maskEditorBrushTools.js` 的工具管理体系：

### 互斥机制

- `deactivateAllCustomTools()` 需增加 Transform 的停用逻辑（调用 `cleanupTransform()`）
- `isAnyCustomToolActive()` 需包含 Transform 的激活状态检查
- 快捷键 Q / C / S 之间互斥：按 Q 时先 `deactivateAllCustomTools()`，再激活 Transform；
  按 C / S 时同理，先停用 Transform 再激活对应工具

### 需要从 maskEditorBrushTools.js 导入的资源

- `brushToolOverlay` — 共享 overlay canvas（Transform 的框选和句柄绘制在此 overlay 上）
- `baseCanvas` / `paintCanvas` — 画布引用
- `displayToCanvas()` — 坐标映射函数
- `deactivateAllCustomTools()` — 互斥控制

### 需要从 maskEditorBrushToolsTransform.js 导出的接口

- `toggleTransform()` — 供按钮和快捷键调用
- `cleanupTransform()` — 供 `deactivateAllCustomTools()` 和 `cleanupAllBrushTools()` 调用
- `isTransformActive()` — 供 `isAnyCustomToolActive()` 调用
- `handleTransformKeydown(e)` — 处理 R（重置）等 Transform 专属按键

## 文件结构

```
js/
├── maskEditorBrushTools.js              # 修改：添加 Transform 状态导出
├── maskEditorBrushToolsTransform.js     # 新建：Transform 工具实现
│   ├── State                            # transformToolState 对象
│   ├── Event Handlers                   # onPointerDown/Move/Up
│   ├── Selection Logic                  # 框选逻辑、内容检测
│   ├── Transform Logic                  # 拖动计算、变换应用
│   ├── Rendering                        # 覆盖层绘制
│   │   ├── renderSelecting()
│   │   ├── renderTransforming()
│   │   └── drawPerspectiveQuad()
│   └── Integration                      # toggleTransform(), cleanup()
└── maskEditorTurbo.js                   # 修改：添加按钮创建和 Q 键绑定
```

## 依赖

**零外部依赖**。所有计算使用原生 JavaScript 和 Canvas 2D API。

需实现的数学函数：
- `isPointInQuad(point, corners)` - 射线法点包含检测
- `pointToSegmentDistance(point, a, b)` - 点到线段距离
- `detectHandle(pos, transform)` - 句柄检测
- `drawPerspectiveQuad(ctx, src, srcQuad, dstQuad)` - 透视渲染（网格细分）
- `bilinearInterpolate(p0, p1, p2, p3, u, v)` - 双线性插值
- `applyTriangleTransform(ctx, s0, s1, s2, d0, d1, d2)` - 三角形仿射变换
- `isImageDataEmpty(imageData, minOpaquePixels)` - 统计 alpha > 0 的像素数，小于阈值则视为空（阈值按选区面积比例计算：`max(16, area * 0.005)`）
