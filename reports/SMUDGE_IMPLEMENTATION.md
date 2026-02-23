# Smudge Tool Implementation Summary

## Overview
成功实现了 Smudge（涂抹）工具，与现有的 Clone Brush 共享同一套架构基础。

---

## Key Changes

### 1. Shared Overlay Architecture (maskEditorTurbo.js)

#### Module-level Resources
```javascript
let brushToolOverlay = null;
let baseCanvas = null;
let paintCanvas = null;
let globalBrushRadius = 20;

// Button references for module-scope style updates
let cloneBtnRef = null;
let smudgeBtnRef = null;
```

#### Unified Overlay Initialization
- 函数：`initBrushToolOverlay()`
- Overlay ID：从 `#clone-brush-overlay` 改为 `#brush-tool-overlay`
- 支持多个工具共享同一个 overlay canvas

#### Tool Utility Functions
- `getBrushRadius()` / `setBrushRadius(radius)` - 笔刷大小读写
- `isPointerInBrushArea(e)` - 检测指针是否在画布区域
- `updateCloneStyle()` / `updateSmudgeStyle()` - 按钮样式更新（**模块级**，通过 `cloneBtnRef`/`smudgeBtnRef` 操作）
- `deactivateAllCustomTools()` - 互斥关闭所有自定义工具
- `isAnyCustomToolActive()` - 检查是否有工具激活

### 2. Clone Brush Refactoring

#### State Changes
- 移除 `editorState.cloneBrush.overlay`（改用模块级 `brushToolOverlay`）
- 移除 `editorState.cloneBrush.baseCanvas`（改用模块级 `baseCanvas`）
- 移除 `editorState.cloneBrush.paintCanvas`（改用模块级 `paintCanvas`）
- 移除 `editorState.cloneBrush.brushRadius`（改用全局 `globalBrushRadius`）

#### Function Updates
- `renderOverlay()` → 重命名为 `renderCloneOverlay()`
- `initCloneBrushOverlay()` → 已删除（改为 `initBrushToolOverlay()`）
- `isPointerInCanvasArea()` → 改为 `isPointerInBrushArea()`
- Clone 按钮事件：使用 `deactivateAllCustomTools()` 实现工具互斥

### 3. Smudge Tool Implementation

#### State in editorState
```javascript
smudgeBrush: {
    active: false,
    isDrawing: false,
    lastDrawX: 0, lastDrawY: 0,
    carriedBuffer: null,      // { data: Float32Array, width, height, offsetX, offsetY }
    eventsBound: false,
}
```

#### Core Smudge Functions

1. **Event Handlers**
   - `initSmudgeToolEvents()` - 绑定文档级捕获阶段监听器
   - `onSmudgeMouseDown(e)` - 采样起始复合像素 → carriedBuffer
   - `onSmudgeMouseMove(e)` - 应用涂抹笔触 + 渲染 overlay
   - `onSmudgeMouseUp(e)` - 完成笔触，保存 undo 检查点，清空 carriedBuffer

2. **Sampling & Compositing**
   - `sampleComposite(cx, cy, radius)` - 从 Base + Paint 采样合成像素，返回 Float32Array

3. **Smudge Algorithm**
   - `applySmudgeStroke(cx, cy)` - 插值笔触路径（同 Clone Brush）
   - `stampSmudge(drawX, drawY, radius)` - 核心涂抹操作：
     - 强度（strength）：固定 `SMUDGE_STRENGTH = 0.5`
     - 高斯衰减：`sigma = radius * 0.4`
     - **笔刷相对坐标**：`cbx = Math.round(dx) + r`（dx = canvasX - drawX）
     - 递进混合：写回 carried buffer，颜色随拖动逐渐衰减

4. **Visualization**
   - `renderSmudgeOverlay(mouseX, mouseY)` - 绘制橙色笔刷圆圈，**始终显示**（不受 isDrawing 限制）

5. **Cleanup**
   - `cleanupSmudgeTool()` - 释放 carriedBuffer，移除事件监听，重置状态

### 4. UI & Keyboard Integration

#### Toolbar Button
- 位置：Clone 按钮之后，Blur 按钮之前
- 图标：`pi pi-arrow-right-arrow-left`
- 文本：`Smudge`
- 样式：复用 `.fast-forward-mode-toggle` class（enabled/disabled 状态）
- `cloneBtnRef = cloneBtn` / `smudgeBtnRef = smudgeBtn` 在创建时赋值

#### Keyboard Shortcuts ([ 和 ])
- 条件改为 `isAnyCustomToolActive()`，支持 Clone 和 Smudge 共用
- 增量调整笔刷大小（±5px），同步更新 UI slider 和全局变量

#### Initialize & Cleanup
- 编辑器打开：`initBrushToolOverlay()` + `initCloneToolEvents()` + `initSmudgeToolEvents()`
- 编辑器关闭：`cleanupCloneTool()` + `cleanupSmudgeTool()` + 清理共享 overlay/canvas 引用

### 5. CSS Updates

#### maskEditorTurbo.css
- 选择器：`#clone-brush-overlay` → `#brush-tool-overlay`（注释正确闭合）

---

## Algorithm Details

### Smudge Blending

```
On mousedown at P(cx, cy):
  carriedBuffer = sampleComposite(cx, cy, radius)
  // carriedBuffer[relX][relY] = composite color at (cx - r + relX, cy - r + relY)

For each interpolated step at Q(drawX, drawY):
  For each pixel at (canvasX, canvasY) within radius of Q:
    dx = canvasX - drawX
    dy = canvasY - drawY
    dist = sqrt(dx² + dy²)
    w = gaussian(dist, sigma=radius*0.4)
    s = SMUDGE_STRENGTH * w   // = 0.5 * w

    // Brush-relative index into carried buffer (always valid regardless of brush position)
    cbx = round(dx) + r        // range [0, 2r]
    cby = round(dy) + r

    dest = composite(baseData, paintData) at (canvasX, canvasY)
    output = carried[cbx][cby] * s + dest * (1 - s)
    paint[canvasX][canvasY] = output
    carried[cbx][cby] = output  // progressive decay

On mouseup:
  canvasHistory.saveState()   // undo checkpoint
  carriedBuffer = null
```

### Compositing Formula
```javascript
const pa = paintAlpha / 255;
composite_rgb = paint_rgb * pa + base_rgb * (1 - pa);
composite_a   = max(paint_a, base_a);
```

---

## Bug Fixes Applied (Post-Review)

| # | 问题 | 修复 |
|---|------|------|
| 1 | CSS `/* ... ===` 缺少 `*/`，overlay 样式规则全被注释掉，工具完全失效 | 补全 `*/` |
| 2 | `updateCloneStyle`/`updateSmudgeStyle` 定义在 `addFastForwardToggleButton` 闭包内，`deactivateAllCustomTools`（模块级）调用时 ReferenceError | 提升为模块级函数，通过 `cloneBtnRef`/`smudgeBtnRef` 访问按钮 |
| 3 | `stampSmudge` 用绝对坐标 `canvasX - cox` 索引 carried buffer；笔刷移动超过 `2r` 后 `cbx >= cw`，涂抹效果消失 | 改为笔刷相对坐标 `Math.round(dx) + r` |
| 4 | `renderSmudgeOverlay` guard `!isDrawing` 导致悬浮时不显示笔刷圆圈 | 移除 guard |
| 5 | 死代码：`getCloneBrushRadius` wrapper、`initCloneBrushOverlay` stub、state 中未用的 `bufferWidth/Height/OffsetX/OffsetY` | 全部删除 |

---

## Testing Checklist

- [ ] Mask Editor 打开，确认工具栏显示 Smudge 按钮
- [ ] 鼠标悬浮在画布上，确认橙色笔刷圆圈显示（无需按下）
- [ ] 点击 Smudge，确认激活（蓝色高亮），Clone 自动关闭
- [ ] 在已绘制区域短距离拖动，确认像素涂抹效果
- [ ] **长笔触测试**：拖动距离 > 2×radius，确认涂抹效果贯穿全程（验证 Bug 3 修复）
- [ ] 在 Base 层拖动，确认原始图像像素被涂抹到 Paint 层
- [ ] `[` / `]` 键调整笔刷大小，确认 Clone 和 Smudge 均响应
- [ ] Ctrl+Z 撤销整个涂抹笔触（一次 undo 回退一整笔）
- [ ] 点击 Clone 后 Smudge 自动关闭，Clone 样式正确（验证 Bug 2 修复）
- [ ] 关闭 Mask Editor，重新打开，确认无残留状态

## Files Modified

| File | Changes |
|------|---------|
| `js/maskEditorTurbo.js` | 工具架构重构、Smudge 实现、5 项 bug 修复 |
| `js/maskEditorTurbo.css` | Overlay selector 更新，修复注释闭合 |

## Known Limitations & Future Work

1. **Strength 参数**：当前固定为 0.5，可考虑未来添加 UI 控件
2. **多层支持**：目前仅在 Paint 层写入，涂抹结果不影响 Mask 层
3. **撤销粒度**：按笔触粒度（mouseup）保存 undo，整笔回退
4. **性能**：大笔刷 + 快速拖动时，每步都有两次 `getImageData`（base + paint）；可考虑缓存 base layer 数据
5. **视觉反馈**：可考虑在涂抹方向添加方向箭头或运动模糊指示
