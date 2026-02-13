# Editor Blur Mode - 设计方案

## 功能概述

**Editor Blur Mode** 是 ComfyUI_InpaintHelper 的一项新功能，提供了一种更高效的 mask 编辑和主界面操作并行工作的方式。该模式允许用户在一个统一的视图中同时看到 mask editor（左半边）和主界面（右半边）。

### 核心特性

- **默认模糊化**：Mask editor 打开时自动模糊化并移至左半边，保持半透明不可操作状态
- **灵活切换**：点击左半边任何区域即可切换 editor 状态（清晰↔模糊）
- **快捷键禁用**：模糊状态下禁用所有键盘快捷键，防止误操作
- **右侧畅通**：右半边主界面完全可操作，不受 editor 影响
- **无冲突交互**：点击不同区域即可在两个界面之间自由切换

---

## 设计方案

### 1. 状态管理

```javascript
const editorBlurState = {
    isBlurred: true,  // 默认模糊化
}
```

**初始状态**：`isBlurred: true`
- Mask editor 打开时自动应用模糊化样式
- 用户需主动点击左半边来激活编辑

**状态转换**：
- 模糊 → 清晰：点击左半边任何区域
- 清晰 → 模糊：再次点击左半边（或关闭 editor）

### 2. 交互流程

```
┌─────────────────────────────────────┐
│  Mask Editor 打开                   │
│  自动应用模糊化样式                 │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  [模糊态] 左半边               │ 右半边  │
│  不可操作，可点击              │ 可操作   │
│  50% 宽度，blur(4px) opacity(0.4) │ 主界面  │
└─────────────────────────────────────┘
         ↓ 点击左半边
┌─────────────────────────────────────┐
│  [清晰态] 左半边（全屏或展开）      │
│  可编辑，所有快捷键启用            │
│  绘制 mask、使用 FF Mode           │
└─────────────────────────────────────┘
         ↓ 再次点击左半边/关闭
回到模糊态或编辑器关闭
```

### 3. 技术实现

#### 3.1 状态切换函数

**文件**: `js/maskEditorTurbo.js`

```javascript
function toggleEditorBlur() {
    editorBlurState.isBlurred = !editorBlurState.isBlurred;
    const editor = document.querySelector("#global-mask-editor");
    if (editor) {
        if (editorBlurState.isBlurred) {
            editor.classList.add("editor-blurred");
        } else {
            editor.classList.remove("editor-blurred");
        }
    }
}
```

**作用**：
- 切换 `isBlurred` 状态
- 动态添加/移除 `editor-blurred` CSS class
- 控制模糊↔清晰的视觉转换

#### 3.2 键盘快捷键禁用

**文件**: `js/maskEditorTurbo.js` - `initFastForwardMode()` 中的 keydown 监听

```javascript
window.addEventListener('keydown', async function(e) {
    if (!ComfyApp.maskeditor_is_opended()) return;

    // 模糊状态下禁用所有快捷键
    if (editorBlurState.isBlurred) {
        e.preventDefault();
        e.stopImmediatePropagation();
        return;
    }

    // 清晰状态下，快捷键正常工作
    // ...
}, true);
```

**禁用的快捷键**（仅在模糊状态下）：
- `Ctrl+Z` - 撤销
- `Ctrl+Y` - 重做
- `Ctrl+L` - 加载 clipspace
- `Enter` - Fast Forward 执行
- 其他所有键盘输入

#### 3.3 点击事件监听

**文件**: `js/maskEditorTurbo.js` - MutationObserver 中

```javascript
const editor = document.querySelector("#global-mask-editor");
if (editor && !editor.dataset.blurListenerAdded) {
    editor.addEventListener('click', (e) => {
        if (editorBlurState.isBlurred) {
            e.preventDefault();
            e.stopImmediatePropagation();
            toggleEditorBlur();
        }
    });
    editor.dataset.blurListenerAdded = 'true';
}
```

**作用**：
- 监听整个 editor 的点击事件
- 模糊状态下点击时切换为清晰态
- 防止事件冒泡，避免触发其他操作

#### 3.4 初始状态应用

**文件**: `js/maskEditorTurbo.js` - `restoreColorAndAddToggle()` 中

```javascript
function restoreColorAndAddToggle() {
    // ...

    // 应用初始模糊状态到编辑器
    const editor = document.querySelector("#global-mask-editor");
    if (editor) {
        if (editorBlurState.isBlurred) {
            editor.classList.add("editor-blurred");
        } else {
            editor.classList.remove("editor-blurred");
        }
    }

    // ...
}
```

**作用**：
- 编辑器打开时，自动应用初始的模糊化样式

#### 3.5 CSS 样式

**文件**: `js/maskEditorTurbo.css`

```css
/* Editor Blur Mode */
#global-mask-editor.editor-blurred {
    position: fixed !important;
    left: 0 !important;
    top: 0 !important;
    width: 50% !important;
    height: 100% !important;
    filter: blur(4px) opacity(0.4);
    pointer-events: auto;
    z-index: 50;
    cursor: pointer;
}

#global-mask-editor.editor-blurred * {
    pointer-events: none;
}
```

**样式说明**：
- **position: fixed** - 固定位置，不随滚动移动
- **left: 0, top: 0, width: 50%, height: 100%** - 占据左半边屏幕
- **filter: blur(4px)** - 4px 模糊效果
- **opacity: 0.4** - 40% 透明度，显示后方内容
- **pointer-events: auto** - editor 本身可接收点击
- **pointer-events: none** 在子元素 - 子元素不可交互，点击冒泡到父元素
- **z-index: 50** - 层级低于弹窗，不遮挡关键 UI
- **cursor: pointer** - 鼠标指针变为手指，提示可点击

---

## 使用场景

### 场景 1：同时进行 mask 编辑和参数调整

1. 打开某个节点的 mask editor
2. Editor 自动模糊化，显示在左边
3. 用户在右边添加/修改节点参数
4. 点击左边激活 editor，进行 mask 绘制
5. 绘制完成后，点击左边切回模糊态，继续调整右边节点

### 场景 2：使用 Fast Forward Mode 进行快速迭代

1. 编辑器以模糊态打开
2. 点击左边激活编辑器
3. 使用 Ctrl+L 加载 clipspace
4. 按 Enter 执行 Fast Forward 循环
5. 循环完成后，编辑器仍处于激活态
6. 可继续绘制或点击左边切为模糊态观察右边的执行结果

### 场景 3：对比效果与编辑

1. 编辑器模糊显示在左边
2. 右边显示节点输出预览和参数
3. 用户可以实时对比编辑前后的效果
4. 需要修改时点击左边激活编辑器

---

## 技术细节

### 键盘事件处理策略

采用 **capture phase** 的事件监听（第三个参数为 `true`）：

```javascript
window.addEventListener('keydown', ..., true);  // capture phase
```

**优势**：
- 在 bubbling phase 之前捕获事件
- 能够 preventDefault 所有后续处理
- 确保快捷键完全被禁用

### 点击事件处理策略

利用 **事件冒泡** 和 **pointer-events**：

1. 用户点击编辑器任何地方
2. 子元素的 `pointer-events: none` 让点击穿过
3. 点击冒泡到 `#global-mask-editor`
4. `#global-mask-editor` 的 `pointer-events: auto` 接收点击
5. 监听器触发，调用 `toggleEditorBlur()`

### 状态持久化

当前设计中，**编辑器关闭后状态重置**：
- 每次打开编辑器时，`isBlurred` 重新设置为 `true`（默认模糊）
- 这种设计保证用户每次打开编辑器都处于清晰的起始状态

---

## 副作用和考虑

### 1. 键盘快捷键

| 快捷键 | 清晰态 | 模糊态 | 说明 |
|--------|--------|--------|------|
| Ctrl+Z | ✅ 启用 | ❌ 禁用 | 撤销 |
| Ctrl+Y | ✅ 启用 | ❌ 禁用 | 重做 |
| Ctrl+L | ✅ 启用 | ❌ 禁用 | 加载 clipspace |
| Enter | ✅ 启用 | ❌ 禁用 | Fast Forward 执行 |
| Esc | 在编辑器中无特殊含义 | - | - |
| 其他键 | ✅ 启用 | ❌ 禁用 | 所有输入被拦截 |

### 2. 视觉影响

- 左半边占 50% 宽度可能会遮挡部分主界面
- 模糊 + 半透明设计使用户仍能看到后方内容
- 适合宽屏显示（16:9 或更宽）

### 3. GPU 资源

- 模糊效果会消耗一定的 GPU 资源
- 半透明渲染需要合成两个图层
- 在低端设备上可能有轻微性能影响

### 4. 编辑器内容保留

- 模糊→清晰转换不会丢失编辑内容
- Brush color 通过 localStorage 保留（Fast Forward Mode 已实现）
- Undo/Redo 历史在转换过程中保留

---

## 文件改动清单

### 修改的文件

1. **js/maskEditorTurbo.js**
   - 添加 `editorBlurState` 状态对象
   - 添加 `toggleEditorBlur()` 函数
   - 修改 `initFastForwardMode()` keydown 监听逻辑
   - 修改 `restoreColorAndAddToggle()` 应用初始状态
   - 在 MutationObserver 中添加点击监听
   - 导出 `toggleEditorBlur` 函数

2. **js/maskEditorTurbo.css**
   - 添加 `.editor-blurred` 样式类

### 新增代码量

- JavaScript: ~40 行
- CSS: ~13 行
- 总计: ~53 行

---

## 后续扩展可能

1. **状态持久化**
   - 记住用户上次的编辑器状态（模糊/清晰）
   - 打开时恢复上次状态

2. **快捷键自定义**
   - 添加快捷键来切换模糊态
   - 允许用户自定义 blur 强度

3. **响应式设计**
   - 在小屏幕上自动调整宽度比例
   - 移动设备适配

4. **动画效果**
   - 模糊↔清晰的平滑过渡
   - 位置改变时的动画

5. **高级模式**
   - "展开" 按钮让编辑器暂时全屏
   - 拖动分割线调整宽度比例

---

## 总结

Editor Blur Mode 通过将 mask editor 模糊化并移至屏幕左侧，创造了一个高效的多任务工作环境。用户可以同时看到编辑器和主界面，通过简单的点击切换两种模式。这个设计充分利用了屏幕空间，提高了工作效率，特别是在需要频繁在 mask 编辑和参数调整之间切换的场景中。
