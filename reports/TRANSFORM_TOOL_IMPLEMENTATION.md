# Transform Tool 实现文档

## 概述

Transform 工具是一个 paint 层内容变换工具，支持框选区域并进行透视变形、拉伸、旋转和移动操作。本文档详细说明了各个函数的实现目的、主要逻辑和功能关联。

## 架构概览

Transform 工具采用状态机驱动的架构，主要包含以下核心模块：

```
Transform Tool Architecture
├── 状态管理 (transformToolState)
├── 事件处理系统 (Event Handlers)
├── 选区管理 (Selection Management)
├── 变换操作 (Transform Operations)
├── 透视变换核心 (Perspective Transform)
├── 渲染系统 (Rendering System)
└── 辅助功能 (Utilities)
```

### 工具流程状态机

```
idle → selecting → transforming
  ↑                      ↓
  ←─────── cleanup ←─────┘
```

## 1. 核心状态管理

### transformToolState
**作用**: 维护 Transform 工具的完整状态
**实现目的**: 集中管理工具的所有状态，包括当前阶段、选区数据、变换状态、拖动状态等
**使用场景**: 所有 Transform 相关操作都依赖此状态对象

**主要逻辑**:
- `stage`: 控制工具流程（idle → selecting → transforming）
- `selection`: 保存选区矩形、源图像缓存、变换历史等
- `transform`: 存储四角坐标，所有变换操作直接修改此数组
- `drag`: 跟踪拖动状态和起始位置

**功能关联**: 对应设计文档中的"状态管理"部分，是整个工具的数据中心

## 2. 共享资源管理

### 画布引用管理
| 函数 | 作用 | 实现目的 |
|------|------|----------|
| `setSharedOverlay(overlay)` | 设置共享的 overlay 画布引用 | 当 overlay 重新创建时同步更新引用 |
| `getSharedBaseCanvas()` | 获取 base 层画布引用 | 延迟获取画布引用，避免初始化时序问题 |
| `getSharedPaintCanvas()` | 获取 paint 层画布引用 | 延迟获取画布引用，避免初始化时序问题 |

### 坐标转换
**screenToCanvasLength(screenLength)**
**作用**: 将屏幕像素长度转换为画布坐标长度
**实现目的**: 确保句柄检测阈值在画布缩放状态下保持一致的手感
**主要逻辑**: 获取画布缩放比例，使用平均缩放值转换长度
**功能关联**: 实现设计文档中"句柄检测距离阈值均以屏幕像素为基准"的要求

## 3. 数学工具函数

### 几何检测算法
| 函数 | 算法 | 用途 | 阈值 |
|------|------|------|------|
| `isPointInQuad(point, corners)` | 射线法 | 检测点是否在四边形内部 | - |
| `pointToSegmentDistance(point, a, b)` | 点到线段距离 | 检测鼠标是否在四边附近 | 10px |
| `detectHandle(pos, transform)` | 优先级检测 | 检测鼠标位置对应的句柄类型 | 四角15px, 旋转10px, 四边10px |

### 边界计算
**getBounds(corners)**
**作用**: 计算四边形的边界框
**实现目的**: 确定变换后图像的绘制区域，优化渲染性能
**主要逻辑**: 找出四角的最小/最大 x,y 坐标，并限制在画布范围内

## 4. 图像数据处理

### 内容检测与采样
```
图像数据处理流程:
选区框选 → 检测内容 → 智能取样 → 缓存源图像
    ↓           ↓          ↓          ↓
sampleLayer → isEmpty → copyFromBase → createCanvas
```

| 函数组 | 功能 | 实现逻辑 |
|--------|------|----------|
| `isImageDataEmpty()` | 检测图像数据是否为空 | 统计 alpha > 0 的像素数 |
| `samplePaintLayer()` / `sampleBaseLayer()` | 从指定图层采样像素 | 使用 getImageData，设置 willReadFrequently 优化 |
| `createSourceCanvas()` | 创建源图像画布 | 从 ImageData 创建 HTMLCanvasElement |
| `clearPaintRect()` / `copyToPaintLayer()` | 清除或复制像素 | 实现剪切操作和自动复制 |

**智能取样逻辑**:
1. 检测 paint 层内容是否足够
2. 如果 paint 层为空，自动从 base 层复制
3. 缓存源图像避免重复采样

## 5. 透视变换核心

### 数学基础
**bilinearInterpolate(p0, p1, p2, p3, u, v)**
**作用**: 双线性插值计算四边形内任意点的坐标
**实现目的**: 将矩形网格映射到任意四边形，实现透视变换
**主要逻辑**: 先在 u 方向插值得到两个点，再在 v 方向插值得到最终点

### 变换渲染
```
透视变换渲染流程:
源四边形 → 网格细分 → 三角形变换 → 目标四边形
    ↓          ↓           ↓           ↓
srcQuad → gridDivision → triangleTransform → dstQuad
```

| 函数 | 作用 | 算法 |
|------|------|------|
| `applyTriangleTransform()` | 计算三角形仿射变换矩阵 | 3点确定仿射变换 |
| `drawPerspectiveQuad()` | 绘制透视四边形变换 | 网格细分 + 三角形渲染 |

## 6. 事件处理系统

### 事件绑定与分发
```
事件处理流程:
initEvents → pointerDown → pointerMove → pointerUp
     ↓           ↓            ↓            ↓
   绑定监听    检测句柄      更新变换      完成操作
```

| 事件处理器 | 职责 | 状态转换 |
|------------|------|----------|
| `initTransformToolEvents()` | 绑定全局事件监听器 | - |
| `onTransformPointerDown()` | 处理鼠标按下，决定操作类型 | idle/selecting → transforming |
| `onTransformPointerMove()` | 处理鼠标移动，更新变换 | - |
| `onTransformPointerUp()` | 处理鼠标释放，完成操作 | transforming → idle |

### 操作类型检测优先级
1. **四角句柄** (15px) - 透视变形
2. **旋转手柄** (10px) - 旋转操作
3. **四边** (10px) - 拉伸操作
4. **选区内部** - 移动操作

## 7. 选区管理

### 选区生命周期
```
选区管理流程:
startSelection → updateSelection → finalizeSelection
      ↓               ↓                 ↓
   记录起始点      实时绘制框选       检测内容创建选区
                                        ↓
                              restorePreviousSelection (失败时)
```

| 函数 | 阶段 | 功能 |
|------|------|------|
| `startSelection()` | 开始 | 记录起始点，开始框选 |
| `updateSelection()` | 进行中 | 实时绘制虚线框和半透明填充 |
| `finalizeSelection()` | 完成 | 验证选区，处理空选区逻辑 |
| `restorePreviousSelection()` | 回退 | 恢复之前的有效选区 |

**选区验证逻辑**:
1. 检测选区大小是否有效
2. 检测选区内容是否足够
3. 处理空选区的自动复制
4. 创建变换状态或回退到之前选区

## 8. 变换操作

### 变换类型与算法
| 变换类型 | 触发条件 | 算法 | 实时性 |
|----------|----------|------|--------|
| **移动** | 选区内部拖动 | 平移变换 | 实时图像 |
| **透视变形** | 四角句柄拖动 | 自由四边形变换 | 轮廓预览 |
| **拉伸** | 四边句柄拖动 | 约束变形 | 轮廓预览 |
| **旋转** | 旋转句柄拖动 | 绕中心旋转 | 轮廓预览 |

### 变换执行流程
```
变换操作流程:
startDrag → updateDrag → endDrag → applyTransform
    ↓          ↓          ↓           ↓
 保存起始    计算变换    应用变换    写入paint层
```

| 函数 | 职责 | 关键逻辑 |
|------|------|----------|
| `startDrag()` | 开始拖动操作 | 保存拖动起始状态 |
| `updateDrag()` | 更新拖动变换 | 根据类型计算不同变换算法 |
| `endDrag()` | 结束拖动操作 | 调用 applyTransform 写入像素 |
| `applyTransform()` | 应用变换到 paint 层 | 清除→计算边界→透视渲染→写入 |

## 9. 渲染系统

### 渲染策略
**性能优化策略**:
- **移动操作**: 实时显示完整图像（性能开销小）
- **透视/拉伸/旋转**: 显示轮廓预览（避免实时透视计算）

### 渲染函数
| 函数 | 用途 | 渲染内容 |
|------|------|----------|
| `renderTransforming()` | 主渲染函数 | 清空overlay → 绘制图像/轮廓 → 绘制句柄 |
| `drawTransformOutline()` | 轮廓渲染 | 虚线边框 |
| `drawTransformedImage()` | 图像渲染 | 透视变换后的完整图像 |
| `drawHandles()` | 句柄渲染 | 四角方形句柄 + 旋转圆形句柄 |

### 句柄视觉规格
- **四角句柄**: 8px 方形，白色填充，黑色边框
- **旋转句柄**: 6px 圆形，位于顶边中点上方

## 10. 光标管理

### 光标样式映射
| 句柄类型 | 光标样式 | 视觉提示 |
|----------|----------|----------|
| 四角句柄 | `nw-resize`, `ne-resize`, `sw-resize`, `se-resize` | 对角调整 |
| 旋转句柄 | `grab` / `grabbing` | 旋转操作 |
| 四边句柄 | `n-resize`, `s-resize`, `e-resize`, `w-resize` | 边缘调整 |
| 选区内部 | `move` | 移动操作 |
| 其他区域 | `crosshair` | 框选模式 |

## 11. 工具生命周期

### 生命周期管理
```
工具生命周期:
激活 → 使用 → 清理
 ↓      ↓      ↓
init → work → cleanup
```

| 函数 | 阶段 | 操作 |
|------|------|------|
| `initTransformToolEvents()` | 激活 | 绑定事件监听器 |
| `isTransformActive()` | 查询 | 检查工具状态 |
| `handleTransformKeydown()` | 使用 | 处理键盘快捷键 (Escape) |
| `cleanupTransform()` | 清理 | 清除选区→移除事件→重置状态→恢复光标 |

### 退出机制
- **Escape 键**: 立即退出工具
- **工具切换**: 自动清理当前工具状态
- **互斥机制**: 确保同时只有一个工具激活

## 12. 辅助功能

### 视觉反馈
**setBaseLayerDimmed(dimmed)**
**作用**: 控制 base 层的亮度显示
**实现目的**: 框选时突出显示选区，提供视觉层次
**主要逻辑**: 通过 CSS opacity 属性控制 base 画布透明度

### 几何工具
**normalizeRect(start, end)**
**作用**: 标准化矩形坐标，确保宽高为正
**实现目的**: 处理反向拖动的选区，统一矩形表示
**主要逻辑**: 计算最小/最大坐标，返回标准化的矩形对象

## 函数关联关系图

```
Transform Tool 函数关联图:

transformToolState (状态中心)
    ├── 事件处理
    │   ├── initTransformToolEvents()
    │   ├── onTransformPointerDown() ──→ detectHandle()
    │   ├── onTransformPointerMove() ──→ updateSelection() / updateDrag()
    │   └── onTransformPointerUp() ──→ finalizeSelection() / endDrag()
    │
    ├── 选区管理
    │   ├── startSelection() ──→ setBaseLayerDimmed()
    │   ├── updateSelection() ──→ normalizeRect()
    │   ├── finalizeSelection() ──→ samplePaintLayer() ──→ isImageDataEmpty()
    │   │                      └──→ sampleBaseLayer() ──→ copyToPaintLayer()
    │   │                      └──→ createSourceCanvas()
    │   └── restorePreviousSelection()
    │
    ├── 变换操作
    │   ├── startDrag() ──→ 保存起始状态
    │   ├── updateDrag() ──→ 计算变换 ──→ renderTransforming()
    │   ├── endDrag() ──→ applyTransform() ──→ drawPerspectiveQuad()
    │   └── applyTransform() ──→ getBounds() ──→ clearPaintRect()
    │
    ├── 渲染系统
    │   ├── renderTransforming() ──→ drawTransformedImage() / drawTransformOutline()
    │   │                       └──→ drawHandles()
    │   ├── drawPerspectiveQuad() ──→ bilinearInterpolate()
    │   │                        └──→ applyTriangleTransform()
    │   └── updateCursor() ──→ getCursorForHandle()
    │
    └── 工具管理
        ├── isTransformActive()
        ├── handleTransformKeydown()
        └── cleanupTransform() ──→ setBaseLayerDimmed(false)

共享资源:
    ├── setSharedOverlay() ──→ 重置画布缓存
    ├── getSharedBaseCanvas() / getSharedPaintCanvas()
    └── screenToCanvasLength() ──→ 坐标转换

数学工具:
    ├── isPointInQuad() ──→ 射线法检测
    ├── pointToSegmentDistance() ──→ 距离计算
    └── detectHandle() ──→ 优先级检测系统
```

## 总结

Transform 工具的实现完全遵循设计文档的架构，通过以下关键特性实现了高性能的图像变换功能：

### 核心特性
1. **状态机驱动**: 清晰的 idle → selecting → transforming 流程
2. **智能取样**: 自动检测内容并从合适的图层获取像素
3. **性能优化**: 移动实时渲染，透视操作轮廓预览
4. **精确交互**: 屏幕坐标基准的句柄检测，优先级明确

### 架构优势
- **模块化设计**: 每个函数职责单一，便于维护和扩展
- **事件驱动**: 响应式的用户交互处理
- **资源管理**: 智能的画布引用和缓存机制
- **数学精确**: 基于双线性插值的透视变换算法

### 实现质量
- **完整性**: 覆盖了设计文档中的所有功能需求
- **健壮性**: 包含错误处理和状态回退机制
- **用户体验**: 直观的视觉反馈和光标提示
- **性能**: 针对不同操作类型的渲染优化策略

整个实现构成了一个完整、高效、易用的图像变换工具系统。