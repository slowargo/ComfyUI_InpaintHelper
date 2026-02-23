# Smudge Tool Implementation Summary

## Overview
成功实现了 Smudge（涂抹）工具，与现有的 Clone Brush 共享同一套架构基础。

## Key Changes

### 1. Shared Overlay Architecture (maskEditorTurbo.js)

#### Module-level Resources
```javascript
let brushToolOverlay = null;
let baseCanvas = null;
let paintCanvas = null;
let globalBrushRadius = 20;
```

#### Unified Overlay Initialization
- 函数：`initBrushToolOverlay()`
- Overlay ID：从 `#clone-brush-overlay` 改为 `#brush-tool-overlay`
- 支持多个工具共享同一个 overlay canvas

#### Tool Utility Functions
- `getBrushRadius()` - 获取笔刷大小（从 slider 或全局变量）
- `setBrushRadius(radius)` - 设置笔刷大小
- `isPointerInBrushArea(e)` - 检测指针是否在画布区域
- `deactivateAllCustomTools()` - 互斥关闭所有自定义工具
- `isAnyCustomToolActive()` - 检查是否有工具激活

### 2. Clone Brush Refactoring

#### State Changes
- 移除 `editorState.cloneBrush.overlay`（改用模块级 `brushToolOverlay`）
- 移除 `editorState.cloneBrush.baseCanvas`（改用模块级 `baseCanvas`）
- 移除 `editorState.cloneBrush.paintCanvas`（改用模块级 `paintCanvas`）
- 移除 `editorState.cloneBrush.brushRadius`（改用全局 `globalBrushRadius`）

#### Function Updates
- `getCloneBrushRadius()` → 委托给 `getBrushRadius()`
- `renderOverlay()` → 重命名为 `renderCloneOverlay()`
- `initCloneBrushOverlay()` → 改为调用 `initBrushToolOverlay()`
- `isPointerInCanvasArea()` → 改为 `isPointerInBrushArea()`
- Clone 按钮事件：使用 `deactivateAllCustomTools()` 实现工具互斥

### 3. Smudge Tool Implementation

#### New State in editorState
```javascript
smudgeBrush: {
    active: false,
    isDrawing: false,
    lastDrawX: 0,
    lastDrawY: 0,
    carriedBuffer: null,      // Float32Array
    bufferWidth: 0,
    bufferHeight: 0,
    bufferOffsetX: 0,
    bufferOffsetY: 0,
    eventsBound: false,
}
```

#### Core Smudge Functions

1. **Event Handlers**
   - `initSmudgeToolEvents()` - 绑定事件监听器
   - `onSmudgeMouseDown(e)` - 采样起始像素
   - `onSmudgeMouseMove(e)` - 应用涂抹笔触
   - `onSmudgeMouseUp(e)` - 完成笔触，保存 undo 检查点

2. **Sampling & Compositing**
   - `sampleComposite(cx, cy, radius)` - 从 Base + Paint 采样合成像素
   - 返回 Float32Array 以避免累积舍入误差

3. **Smudge Algorithm**
   - `applySmudgeStroke(cx, cy)` - 插值笔触路径
   - `stampSmudge(cx, cy, radius)` - 每个步骤的核心涂抹操作
     - 强度（strength）：固定 0.5
     - 高斯衰减：sigma = radius * 0.4
     - 递进混合：carried 像素不断混合，产生衰减效果

4. **Visualization**
   - `renderSmudgeOverlay(mouseX, mouseY)` - 绘制橙色笔刷圆圈（区分于 Clone 的白色）

5. **Cleanup**
   - `cleanupSmudgeTool()` - 释放资源，移除事件监听

### 4. UI & Keyboard Integration

#### Toolbar Button
- 位置：Clone 按钮之后，Blur 按钮之前
- 图标：`pi pi-arrow-right-arrow-left`
- 文本：`Smudge`
- 样式：复用 `.fast-forward-mode-toggle` class（enabled/disabled 状态）
- 点击行为：互斥激活（点击 Smudge 时关闭 Clone，反之亦然）

#### Keyboard Shortcuts ([ 和 ])
- 支持所有激活的自定义工具（不限于 Clone）
- 增量调整笔刷大小（±5px）
- 同步更新 UI slider 和全局变量

#### Initialize & Cleanup
- 编辑器打开时：`initBrushToolOverlay()` + `initCloneToolEvents()` + `initSmudgeToolEvents()`
- 编辑器关闭时：`cleanupCloneTool()` + `cleanupSmudgeTool()` + 清理共享资源

### 5. CSS Updates

#### maskEditorTurbo.css
- 选择器更新：`#clone-brush-overlay` → `#brush-tool-overlay`
- 保持原有功能：pointer-events toggle、z-index、cursor 等

## Algorithm Details

### Smudge Blending
```
On mousedown at P:
  1. Sample composite (base + paint) at P → carriedBuffer

For each interpolated step at Q:
  1. Read composite at Q → destBuffer
  2. For each pixel within radius R:
     - w = gaussian(distance, radius) with σ = radius * 0.4
     - s = STRENGTH * w = 0.5 * w
     - output = carried * s + dest * (1 - s)
     - carried = output (update for next step)
  3. Write output to Paint layer

On mouseup:
  - Save undo checkpoint via canvasHistory.saveState()
  - Clear carriedBuffer
```

### Compositing Formula
```javascript
// Composite pixel from Paint + Base
alpha_norm = paintAlpha / 255
composited_rgb = paint_rgb * alpha_norm + base_rgb * (1 - alpha_norm)
composited_alpha = max(paint_alpha, base_alpha)
```

## Testing Checklist

- [ ] Mask Editor 打开，确认工具栏显示 Smudge 按钮
- [ ] 点击 Smudge，确认激活（蓝色高亮）
- [ ] Smudge 激活时，Clone 自动关闭
- [ ] 在已绘制区域拖动，确认像素涂抹效果
- [ ] 在 Base 层拖动，确认原始图像像素被涂抹到 Paint 层
- [ ] `[` / `]` 键调整笔刷大小，确认同时工作于 Clone 和 Smudge
- [ ] Ctrl+Z 撤销整个涂抹笔触
- [ ] 点击 Clone，Smudge 自动关闭
- [ ] 关闭 Mask Editor，确认无内存泄漏
- [ ] 快速切换工具，确认状态管理正确

## Files Modified

| File | Changes |
|------|---------|
| `js/maskEditorTurbo.js` | 工具架构重构、Smudge 实现、UI 更新、事件处理 |
| `js/maskEditorTurbo.css` | Overlay selector 更新 |

## Architecture Benefits

1. **可扩展性**：新增工具只需实现 init + event handlers + stamp + render + cleanup
2. **代码复用**：Overlay canvas、坐标转换、笔刷大小、keyboard handler 等共享
3. **工具互斥**：`deactivateAllCustomTools()` 避免多工具冲突
4. **性能**：单个 overlay canvas、合理的采样和混合策略
5. **用户体验**：视觉反馈（不同颜色区分工具）、自然的涂抹效果

## Known Limitations & Future Work

1. **Strength 参数**：当前固定为 0.5，可考虑未来添加 UI 控件
2. **多层支持**：目前仅在 Paint 层写入，可考虑支持 Mask 层
3. **撤销粒度**：按笔触粒度（mouseup）保存 undo，可考虑更细粒度
4. **性能优化**：大笔刷 + 快速拖动时的 GPU 优化
5. **视觉反馈**：可考虑显示涂抹方向箭头或强度指示器
