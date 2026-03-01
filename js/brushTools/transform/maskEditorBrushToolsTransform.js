import { displayToCanvas, showToast, getBrushOpacity } from "../../utils.js";
import { saveCanvasHistory } from "../common/helpers.js";
import {
    getSharedOverlay,
    getSharedBaseCanvas,
    getSharedPaintCanvas,
} from "../common/sharedCanvasRefs.js";
import {
    detectHandle,
    getBounds,
    getSourceCorners,
    hasCornersChanged,
    isPointInQuad,
    normalizeRect,
    screenToCanvasLength,
    snapRectToCanvasPixels,
} from "./math.js";
import {
    autoSelectPaintObjectFromTinyRect,
    clearPaintByMask,
    clearPaintRect,
    createSourceCanvas,
    isImageDataEmpty,
    releaseCanvasBuffer,
    releaseSelectionResources,
    restoreUntransformedPaintSelection,
    sampleBaseLayer,
    samplePaintLayer,
    setBaseLayerDimmed,
} from "./layer.js";
import { drawPerspectiveQuad } from "./perspective.js";
import {
    createCustomCursor,
    destroyCustomCursor,
    updateCustomCursor,
    showCustomCursor,
    getCursorForHandle,
} from "./cursor.js";

// === Transform Tool State ===
const transformToolState = {
    // 当前阶段。idle 为初始状态。
    stage: 'idle', // 'idle' | 'selecting' | 'transforming'

    // 选区数据（仅在 selecting/transforming 时存在）
    selection: null,

    // 变换状态（corner 等。仅在 transforming 时存在）
    transform: null,

    // 拖动状态
    drag: {
        active: false,
        type: null, // 'move' | 'corner' | 'edge' | 'rotate'
        targetIndex: -1,
        startMouse: { x: 0, y: 0 },
        startTransform: null
    },

    // 事件绑定状态
    eventsBound: false,

    // 最后的鼠标位置
    lastMousePos: null,

    // 保存的旧选区会话（用于新选区无效时恢复）
    // shouldRestoreCutPixels: 旧会话是否属于“仅剪切未变换”的 paint 选区。
    // 注意该判断必须在 clearSelection 前做，因为 clearSelection 可能会提交并重置 dirty 标记。
    previousSession: null,

    // 准实时预览控制
    preview: {
        throttleMs: 200,      // 刷新间隔（毫秒）
        lastRenderTime: 0,    // 上次渲染时间
        pendingRender: false, // 是否有待渲染请求
        timerId: null         // setTimeout ID
    }
};

// 透视绘制使用的可复用画布；不放入状态树，避免状态对象承担大块像素缓存。
let reusableTransformCanvas = null;

/**
 * 获取当前光标类型（用于自定义光标）
 * @returns {string} cursor type
 */
function getCurrentCursorType(pos) {
    if (transformToolState.stage === 'selecting') {
        return 'crosshair';
    }

    if (transformToolState.stage === 'transforming') {
        // 如果有活动拖动，根据拖动类型设置光标
        if (transformToolState.drag.active) {
            const drag = transformToolState.drag;
            if (drag.type === 'rotate') {
                return 'grabbing';
            } else if (drag.type === 'move') {
                return 'move';
            } else if (drag.type === 'corner') {
                const cornerCursors = ['nwse-resize', 'nesw-resize', 'nwse-resize', 'nesw-resize'];
                return cornerCursors[drag.targetIndex] || 'nwse-resize';
            } else if (drag.type === 'edge') {
                const edgeCursors = ['ns-resize', 'ew-resize', 'ns-resize', 'ew-resize'];
                return edgeCursors[drag.targetIndex] || 'move';
            }
        }

        // 检测句柄
        const handle = detectHandle(pos, transformToolState.transform);
        if (handle) {
            return getCursorForHandle(handle);
        }

        // 检测是否在选区内部
        if (isPointInQuad(pos, transformToolState.transform.corners)) {
            return 'move';
        }

        return 'default';
    }

    return 'default';
}

/**
 * 更新自定义光标（在 pointermove 中调用）
 */
function updateCustomCursorAt(clientX, clientY) {
    const overlay = getSharedOverlay();
    if (!overlay) return;

    // 检查是否在画布区域内
    const rect = overlay.getBoundingClientRect();
    const isOverCanvas = clientX >= rect.left && clientX <= rect.right &&
        clientY >= rect.top && clientY <= rect.bottom;

    if (!isOverCanvas) {
        showCustomCursor(false);
        return;
    }

    const pos = displayToCanvas(overlay, clientX, clientY);
    const cursorType = getCurrentCursorType(pos);

    // Transform 工具激活时总是显示光标
    // idle 状态显示 crosshair（准备框选），其他状态根据操作显示对应光标
    if (transformToolState.stage === 'idle') {
        updateCustomCursor(clientX, clientY, 'crosshair');
        showCustomCursor(true);
    } else if (cursorType !== 'default') {
        updateCustomCursor(clientX, clientY, cursorType);
        showCustomCursor(true);
    } else {
        showCustomCursor(false);
    }
}

/**
 * 重置光标为默认
 */
function resetCursor() {
    showCustomCursor(false);
}

// === Event Handlers ===

/**
 * 绑定 Transform 工具事件
 */
function initTransformToolEvents() {
    if (transformToolState.eventsBound) return;
    document.addEventListener('pointerdown', onTransformPointerDown, true);
    document.addEventListener('pointermove', onTransformPointerMove, true);
    document.addEventListener('pointerup', onTransformPointerUp, true);
    transformToolState.eventsBound = true;
    // 创建自定义光标元素并设置初始样式
    createCustomCursor();
    // 立即设置初始光标为 crosshair（框选模式）
    updateCustomCursor(0, 0, 'crosshair');
}

/**
 * 处理指针按下事件
 */
function onTransformPointerDown(e) {
    if (!isTransformActive()) return;

    // 获取共享资源
    const overlay = getSharedOverlay();
    if (!overlay) return;

    // 检查是否在画布区域内
    const rect = overlay.getBoundingClientRect();
    if (e.clientX < rect.left || e.clientX > rect.right ||
        e.clientY < rect.top || e.clientY > rect.bottom) return;

    e.stopImmediatePropagation();
    e.preventDefault();

    const pos = displayToCanvas(overlay, e.clientX, e.clientY);

    if (transformToolState.stage === 'transforming') {
        // transforming -> selecting, 提交变换

        // 检测句柄
        const handle = detectHandle(pos, transformToolState.transform);
        if (handle) {
            startDrag(handle.type, pos, handle.index);
            return;
        }

        // 检测是否在选区内部
        if (isPointInQuad(pos, transformToolState.transform.corners)) {
            startDrag('move', pos);
            return;
        }

        // 在选区外：保存旧选区状态，开始新框选（不还原像素，以便失败时恢复）
        const shouldRestoreCutPixels = Boolean(
            transformToolState.selection?.cutFromPaint &&
            !transformToolState.selection?.hasAppliedTransform &&
            !transformToolState.selection?.pendingTransform &&
            !transformToolState.selection?.paintDirtySinceLastSave
        );
        transformToolState.previousSession = {
            selection: transformToolState.selection,
            transform: transformToolState.transform,
            shouldRestoreCutPixels,
        };
        // 这里允许 clearSelection 内部按 dirty 标记落盘（若此前有实际变换提交），
        // 但禁止回填像素，并保留资源供后续失败回滚。
        clearSelection(false, true); // 不还原像素，并保留资源用于回滚
        transformToolState.stage = 'selecting';
        startSelection(pos);
    } else if (transformToolState.stage === 'selecting') {
        // selecting -> selecting

        // 如果正在框选中（已按下未松开），继续；否则开始新框选
        // 正常情况下应该不会走到这里。（鼠标左键按下触发选框，之后按下右键就会再次触发PointerDown走到这)
        if (!selectionStart) {
            startSelection(pos);
        }
    } else if (transformToolState.stage === 'idle') {
        // idle -> selecting 工具激活后开始绘制选区

        transformToolState.stage = 'selecting';
        startSelection(pos);
    }
    /*
      回归用例（围绕外部点击触发 clearSelection(false, true) 的状态流）:

      1) 未变换切换选区
         步骤: paint 画一块 -> 框选 -> 不拖动 -> 点击外部新建选区。
         预期: 原像素不丢失（旧选区可按需回填）；不新增无意义 undo 点。

      2) 变换后切换选区
         步骤: 框选 -> 移动/缩放一次 -> 点击外部新建选区。
         预期: 旧选区结果保留在新位置；不会回填到初始剪切位置；仅新增一次有效 history。

      3) 失败回滚后停用工具（关键链路）
         步骤: 从 paint 创建选区 -> 移动 -> 点击空白触发无效新选区并恢复旧选区 -> 立即停用 Transform。
         预期: 不回填到最初创建选区位置；paint 仅保留已提交结果；undo 行为稳定。

      4) 无变换失败回滚
         步骤: 从 paint 创建选区但不拖动 -> 点击空白触发失败回滚 -> 停用 Transform。
         预期: 允许把初始剪切内容回填到原位（等价取消）；不新增 history。

      5) 同一选区多次拖动后一次清选
         步骤: 同一选区连续拖动 2-3 次 -> Esc/切换工具清选。
         预期: clearSelection 前会先提交 pendingTransform；整个会话只落一次 history。

      6) Erase 介入后再次 Transform 提交（destination-out 残留防护）
         步骤: 从 paint 创建选区 -> 移动 -> 退出 Transform -> 用 erase 做任意擦除 ->
              重新激活 Transform 并创建新选区 -> 移动 -> 停用 Transform。
         预期: 停用时会把当前选区正常绘制到 paint（source-over），
              不会因残留 destination-out 导致“未绘制/被擦除”。
    */
}

/**
 * 处理指针移动事件
 */
function onTransformPointerMove(e) {
    if (!isTransformActive()) return;

    const overlay = getSharedOverlay();
    if (!overlay) return;

    const pos = displayToCanvas(overlay, e.clientX, e.clientY);
    transformToolState.lastMousePos = pos;

    // 更新自定义光标
    updateCustomCursorAt(e.clientX, e.clientY);

    if (transformToolState.stage === 'selecting') {
        updateSelection(pos);
    } else if (transformToolState.stage === 'transforming') {
        if (transformToolState.drag.active) {
            updateDrag(pos);
        }
        renderTransforming();
    }
}

/**
 * 处理指针释放事件
 */
function onTransformPointerUp(e) {
    if (!isTransformActive()) return;

    if (transformToolState.stage === 'selecting') {
        finalizeSelection();
    } else if (transformToolState.stage === 'transforming' && transformToolState.drag.active) {
        endDrag();
    }
}

// === Selection Logic ===

let selectionStart = null;

/**
 * 开始框选
 */
function startSelection(pos) {
    selectionStart = { x: pos.x, y: pos.y };
    setBaseLayerDimmed(true);
}

/**
 * 更新框选
 */
function updateSelection(pos) {
    if (!selectionStart) return;

    const overlay = getSharedOverlay();
    if (!overlay) return;

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const rect = normalizeRect(selectionStart, pos);

    // 绘制虚线选区框（使用屏幕坐标）
    const lineWidth = screenToCanvasLength(1);
    const dashSize = screenToCanvasLength(4);
    ctx.setLineDash([dashSize, dashSize]);
    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);

    // 半透明填充
    ctx.fillStyle = 'rgba(74, 158, 255, 0.15)';
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);

    ctx.setLineDash([]);
}

/**
 * 完成框选
 */
function finalizeSelection() {
    if (!selectionStart) return;

    const paintCanvas = getSharedPaintCanvas();
    const baseCanvas = getSharedBaseCanvas();
    const canvasWidth = paintCanvas?.width || baseCanvas?.width || 0;
    const canvasHeight = paintCanvas?.height || baseCanvas?.height || 0;

    const currentPos = transformToolState.lastMousePos || selectionStart;
    const rawRect = normalizeRect(selectionStart, currentPos);
    let rect = snapRectToCanvasPixels(rawRect, canvasWidth, canvasHeight);
    let sourceData = null;
    let cutFromPaint = true;
    let useMaskClear = false;

    // 过滤过小的选区
    if (rawRect.width < 5 || rawRect.height < 5) {
        // 小框选优先尝试自动识别 paint 连通对象（点击选对象）
        const autoSelection = autoSelectPaintObjectFromTinyRect(rawRect);
        if (!autoSelection) {
            restorePreviousSelection();
            return;
        }

        rect = snapRectToCanvasPixels(autoSelection.rect, canvasWidth, canvasHeight);
        if (rect.width <= 0 || rect.height <= 0) {
            restorePreviousSelection();
            return;
        }
        sourceData = autoSelection.imageData;
        cutFromPaint = true;
        useMaskClear = true;
    }

    if (!sourceData) {
        if (rect.width <= 0 || rect.height <= 0) {
            restorePreviousSelection();
            return;
        }
        // 检测选区内容
        const paintData = samplePaintLayer(rect);
        const minPixels = Math.max(16, Math.floor(rect.width * rect.height * 0.005));
        const isEmpty = isImageDataEmpty(paintData, minPixels);

        sourceData = paintData;
        cutFromPaint = true;
        // paint 层为空时，使用 base 层内容作为变换源
        if (isEmpty) {
            // 从 base 层检测选区内容
            const baseData = sampleBaseLayer(rect);
            const isBaseEmpty = isImageDataEmpty(baseData, minPixels);
            if (isBaseEmpty) {
                // should not reach here. just in case
                showToast('Both layers are empty in this area', { duration: 2000 });
                restorePreviousSelection();
                return;
            }
            sourceData = baseData;
            cutFromPaint = false;
        }
    }

    transformToolState.selection = {
        rect: rect,
        sourceCanvas: createSourceCanvas(sourceData),
        // true 表示该选区创建时对 paint 层做过剪切；后续 clearSelection 可能需要回填。
        cutFromPaint,
        // true 表示该选区至少一次把变换结果提交到 paint 层。
        // 该状态用于防止 history 落盘后 dirty 被清零时误触发“初始位置回填”。
        hasAppliedTransform: false,
        // true 表示该选区已对 paint 做过“有效提交”；仅该标记为 true 时允许落一次 history。
        paintDirtySinceLastSave: false,
        pendingTransform: false
    };

    // 剪切：仅当源来自 paint 层时，才从 paint 层清除选区像素。
    // 注意：这一步是“进入浮动选区编辑态”，不是最终提交，不写 history。
    if (cutFromPaint) {
        if (useMaskClear) {
            clearPaintByMask(rect, sourceData);
        } else {
            clearPaintRect(rect);
        }
    }

    // 初始化变换状态
    transformToolState.transform = {
        corners: [
            { x: rect.x, y: rect.y },
            { x: rect.x + rect.width, y: rect.y },
            { x: rect.x + rect.width, y: rect.y + rect.height },
            { x: rect.x, y: rect.y + rect.height }
        ]
    };

    // 恢复 base 层亮度
    setBaseLayerDimmed(false);

    // 进入 transform 模式
    transformToolState.stage = 'transforming';
    selectionStart = null;

    // 新选区创建成功：提交/释放旧选区
    if (transformToolState.previousSession?.shouldRestoreCutPixels) {
        restoreUntransformedPaintSelection(transformToolState.previousSession.selection);
    }
    releaseSelectionResources(transformToolState.previousSession?.selection);
    transformToolState.previousSession = null;
}

/**
 * 恢复之前的选区（新选区无效时调用）
 */
function restorePreviousSelection() {
    selectionStart = null;
    setBaseLayerDimmed(false);

    // 清空 overlay
    const overlay = getSharedOverlay();
    if (overlay) {
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    // 恢复旧选区
    if (transformToolState.previousSession?.selection && transformToolState.previousSession?.transform) {
        transformToolState.selection = transformToolState.previousSession.selection;
        transformToolState.transform = transformToolState.previousSession.transform;
        transformToolState.stage = 'transforming';
        // 重绘
        renderTransforming();
        showToast('New selection invalid, restored previous selection', { duration: 2000 });
    } else {
        transformToolState.stage = 'idle';
        resetCursor();
    }

    // 清空保存的状态
    transformToolState.previousSession = null;
}

// === Transform Logic ===

/**
 * 开始拖动
 */
function startDrag(type, pos, targetIndex = -1) {
    transformToolState.drag = {
        active: true,
        type: type,
        targetIndex: targetIndex,
        startMouse: { x: pos.x, y: pos.y },
        startTransform: {
            corners: transformToolState.transform.corners.map(c => ({ x: c.x, y: c.y }))
        }
    };
}

/**
 * 更新拖动
 */
function updateDrag(pos) {
    const drag = transformToolState.drag;
    if (!drag.active) return;

    const dx = pos.x - drag.startMouse.x;
    const dy = pos.y - drag.startMouse.y;

    if (drag.type === 'move') {
        // 移动整个选区
        for (let i = 0; i < 4; i++) {
            transformToolState.transform.corners[i].x = drag.startTransform.corners[i].x + dx;
            transformToolState.transform.corners[i].y = drag.startTransform.corners[i].y + dy;
        }
    } else if (drag.type === 'corner') {
        // 移动角点
        transformToolState.transform.corners[drag.targetIndex].x = drag.startTransform.corners[drag.targetIndex].x + dx;
        transformToolState.transform.corners[drag.targetIndex].y = drag.startTransform.corners[drag.targetIndex].y + dy;
    } else if (drag.type === 'edge') {
        // 边拖动：沿边的法向量方向缩放
        const edgeIndex = drag.targetIndex;
        const nextIndex = (edgeIndex + 1) % 4;

        // 计算边的中点和法向量
        const edge = {
            x: drag.startTransform.corners[nextIndex].x - drag.startTransform.corners[edgeIndex].x,
            y: drag.startTransform.corners[nextIndex].y - drag.startTransform.corners[edgeIndex].y
        };
        const edgeLength = Math.hypot(edge.x, edge.y);
        const normal = { x: -edge.y / edgeLength, y: edge.x / edgeLength };

        // 计算沿法向量的投影距离
        const projection = dx * normal.x + dy * normal.y;

        // 移动对应的两个角点
        transformToolState.transform.corners[edgeIndex].x = drag.startTransform.corners[edgeIndex].x + projection * normal.x;
        transformToolState.transform.corners[edgeIndex].y = drag.startTransform.corners[edgeIndex].y + projection * normal.y;
        transformToolState.transform.corners[nextIndex].x = drag.startTransform.corners[nextIndex].x + projection * normal.x;
        transformToolState.transform.corners[nextIndex].y = drag.startTransform.corners[nextIndex].y + projection * normal.y;
    } else if (drag.type === 'rotate') {
        // 旋转：计算中心点和旋转角度
        const center = {
            x: (drag.startTransform.corners[0].x + drag.startTransform.corners[1].x +
                drag.startTransform.corners[2].x + drag.startTransform.corners[3].x) / 4,
            y: (drag.startTransform.corners[0].y + drag.startTransform.corners[1].y +
                drag.startTransform.corners[2].y + drag.startTransform.corners[3].y) / 4
        };

        // 计算旋转角度
        const startAngle = Math.atan2(drag.startMouse.y - center.y, drag.startMouse.x - center.x);
        const currentAngle = Math.atan2(pos.y - center.y, pos.x - center.x);
        const deltaAngle = currentAngle - startAngle;

        // 应用旋转到所有角点
        const cos = Math.cos(deltaAngle);
        const sin = Math.sin(deltaAngle);

        for (let i = 0; i < 4; i++) {
            const relX = drag.startTransform.corners[i].x - center.x;
            const relY = drag.startTransform.corners[i].y - center.y;

            transformToolState.transform.corners[i].x = center.x + relX * cos - relY * sin;
            transformToolState.transform.corners[i].y = center.y + relX * sin + relY * cos;
        }
    }

    // 请求节流渲染（准实时预览）
    requestThrottledRender();
}

/**
 * 请求节流的实时预览渲染
 * 限制刷新频率为每 200ms 一次，避免性能问题
 */
function requestThrottledRender() {
    const preview = transformToolState.preview;
    const now = performance.now();
    const elapsed = now - preview.lastRenderTime;

    // 如果距离上次渲染已超过节流间隔，立即渲染
    if (elapsed >= preview.throttleMs) {
        if (preview.timerId) {
            clearTimeout(preview.timerId);
            preview.timerId = null;
        }
        preview.pendingRender = false;
        preview.lastRenderTime = now;
        renderTransforming();
        return;
    }

    // 否则，如果已有待渲染请求，不重复设置
    if (preview.pendingRender) {
        return;
    }

    // 设置定时器在剩余时间后渲染
    preview.pendingRender = true;
    const remaining = preview.throttleMs - elapsed;
    preview.timerId = setTimeout(() => {
        preview.pendingRender = false;
        preview.timerId = null;
        preview.lastRenderTime = performance.now();
        renderTransforming();
    }, remaining);
}

/**
 * 取消待执行的预览渲染
 */
function cancelPendingRender() {
    const preview = transformToolState.preview;
    if (preview.timerId) {
        clearTimeout(preview.timerId);
        preview.timerId = null;
    }
    preview.pendingRender = false;
}

/**
 * 结束拖动
 */
function endDrag() {
    const { drag, transform } = transformToolState;
    if (!drag.active) return;

    // 取消待执行的预览渲染，确保立即渲染最终状态
    cancelPendingRender();

    const didGeometryChange = hasCornersChanged(
        drag.startTransform?.corners,
        transform?.corners
    );
    if (didGeometryChange) {
        // 延迟提交：拖动结束只更新预览，不立即写入 paint 层
        if (transformToolState.selection) {
            transformToolState.selection.pendingTransform = true;
            // 几何变化意味着后续 applyTransform 会实际改动 paint，
            // 因此提前打脏标，确保 clearSelection 时只落一次 history。
            transformToolState.selection.paintDirtySinceLastSave = true;
        }
    }

    transformToolState.drag.active = false;
    renderTransforming();
}

/**
 * 应用变换
 */
function applyTransform() {
    const { selection, transform } = transformToolState;
    if (!selection || !transform) return;

    const paintCanvas = getSharedPaintCanvas();
    if (!paintCanvas) return;

    const corners = transform.corners;

    // 计算本次变换的 bounding box
    const bounds = getBounds(corners);

    // 如果变换后的区域无效，跳过
    if (bounds.width <= 0 || bounds.height <= 0) return;

    // 渲染到临时画布（模块级复用，避免每次申请新画布）
    if (!reusableTransformCanvas) {
        reusableTransformCanvas = document.createElement('canvas');
    }
    reusableTransformCanvas.width = bounds.width;
    reusableTransformCanvas.height = bounds.height;
    const tempCtx = reusableTransformCanvas.getContext('2d');
    tempCtx.clearRect(0, 0, bounds.width, bounds.height);

    const paintCtx = paintCanvas.getContext('2d', { willReadFrequently: true });

    // 计算目标四边形（相对于临时画布的本地坐标）
    const localCorners = corners.map(c => ({
        x: c.x - bounds.x,
        y: c.y - bounds.y
    }));

    // 源四边形（sourceCanvas 的完整区域）
    const srcQuad = getSourceCorners(selection.rect);

    // 使用透视变换渲染
    drawPerspectiveQuad(tempCtx, selection.sourceCanvas, srcQuad, localCorners, 20);

    // 合成到 paint 层（应用 brush opacity）
    const opacity = getBrushOpacity();
    // Eraser may leave destination-out on context; force normal compositing for transform apply.
    paintCtx.save();
    paintCtx.globalCompositeOperation = 'source-over';
    paintCtx.globalAlpha = opacity;
    paintCtx.drawImage(reusableTransformCanvas, bounds.x, bounds.y);
    paintCtx.restore();

    // 记录：该选区已对 paint 产生有效提交
    selection.hasAppliedTransform = true;
    selection.pendingTransform = false;
    selection.paintDirtySinceLastSave = true;
    // 本次提交已经“结算”了初始剪切，不允许后续 clearSelection 再回填到初始位置。
    selection.cutFromPaint = false;
}

// === Rendering ===

/**
 * 渲染 Transform 模式
 */
function renderTransforming() {
    const overlay = getSharedOverlay();
    if (!overlay || transformToolState.stage !== 'transforming') return;

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const corners = transformToolState.transform.corners;

    // 绘制变换后的图像（准实时预览，所有操作类型都支持）
    drawTransformedImage(ctx, corners);

    // 绘制句柄
    drawHandles(ctx, corners);
}

/**
 * 绘制变换轮廓
 */
function drawTransformOutline(ctx, corners) {
    const lineWidth = screenToCanvasLength(1);
    const dashSize = screenToCanvasLength(4);
    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = lineWidth;
    ctx.setLineDash([dashSize, dashSize]);
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < 4; i++) {
        ctx.lineTo(corners[i].x, corners[i].y);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);
}

/**
 * 绘制变换后的图像
 */
function drawTransformedImage(ctx, corners) {
    const { selection } = transformToolState;
    if (!selection || !selection.sourceCanvas) {
        drawTransformOutline(ctx, corners);
        return;
    }

    // 源四边形（sourceCanvas 的完整区域）
    const srcQuad = getSourceCorners(selection.rect);

    // 应用 brush opacity 到预览
    const opacity = getBrushOpacity();
    ctx.save();
    ctx.globalAlpha = opacity;

    // 使用透视变换渲染到 overlay
    drawPerspectiveQuad(ctx, selection.sourceCanvas, srcQuad, corners, 20);

    ctx.restore();

    // 绘制轮廓线（不受 opacity 影响）
    drawTransformOutline(ctx, corners);
}

/**
 * 绘制句柄
 */
function drawHandles(ctx, corners) {
    // 绘制四角句柄
    ctx.fillStyle = '#4a9eff';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;

    const cornerSize = screenToCanvasLength(4);  // 4px 屏幕半边长 -> 8px 总边长
    for (let i = 0; i < 4; i++) {
        ctx.fillRect(corners[i].x - cornerSize, corners[i].y - cornerSize, cornerSize * 2, cornerSize * 2);
        ctx.strokeRect(corners[i].x - cornerSize, corners[i].y - cornerSize, cornerSize * 2, cornerSize * 2);
    }

    // 绘制旋转手柄
    const handleLength = screenToCanvasLength(30);  // 30px 屏幕长度
    const topCenter = {
        x: (corners[0].x + corners[1].x) / 2,
        y: (corners[0].y + corners[1].y) / 2
    };
    const rotateHandle = {
        x: topCenter.x,
        y: topCenter.y - handleLength
    };

    // 连接线
    ctx.strokeStyle = '#4a9eff';
    ctx.beginPath();
    ctx.moveTo(topCenter.x, topCenter.y);
    ctx.lineTo(rotateHandle.x, rotateHandle.y);
    ctx.stroke();

    // 旋转手柄圆圈
    const handleRadius = screenToCanvasLength(6);  // 6px 屏幕半径
    ctx.beginPath();
    ctx.arc(rotateHandle.x, rotateHandle.y, handleRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
}

// === Tool Management ===

/**
 * 清除选区
 * @param {boolean} restorePixels - 是否在未变换时还原剪切自 paint 层的像素到 paint 层（默认 true）
 * @param {boolean} preserveSelectionResources - 是否保留选区资源供失败回滚（默认 false）
 */
function clearSelection(restorePixels = true, preserveSelectionResources = false) {
    const { selection } = transformToolState;

    if (selection) {
        // 如果有未提交的变换预览，在清除选区前一次性提交到 paint 层
        if (selection.pendingTransform) {
            applyTransform();
        }

        // 如果选区从 paint 层创建，一开始会剪切 paint 层内容（从而实现后续的移动/变换）
        // 当没有进行移动/变换时，applyTransform没有触发，这里应该把内容还给 paint 层，避免内容丢失
        // 如果选区从 base 层创建，paint 层一直是空，不需要还原
        // 回填仅用于撤销“进入选区时的剪切”，属于 no-op 还原，不应计入 history。
        if (
            restorePixels &&
            selection.cutFromPaint &&
            !selection.hasAppliedTransform &&
            !selection.paintDirtySinceLastSave
        ) {
            restoreUntransformedPaintSelection(selection);
        }

        // 统一 history 策略：
        // 仅在 paint 已发生“有效提交”时保存一次；纯剪切/回填不保存。
        if (selection.paintDirtySinceLastSave) {
            saveCanvasHistory();
            selection.paintDirtySinceLastSave = false;
        }

        if (!preserveSelectionResources) {
            releaseSelectionResources(selection);
        }
    }

    // 清空 overlay
    const overlay = getSharedOverlay();
    if (overlay) {
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    // 恢复 base 层亮度
    setBaseLayerDimmed(false);

    // 重置光标
    resetCursor();

    // 重置状态
    transformToolState.stage = 'idle';
    transformToolState.selection = null;
    transformToolState.transform = null;
    transformToolState.drag = { active: false };

    // 注意：不清除 previousSession
    // previousSession 用于在创建新选区失败时恢复旧选区
    // 在 restorePreviousSelection 或 finalizeSelection 成功后会清除
}


/**
 * 清理 Transform 工具
 */
function cleanupTransform() {
    const previousSelectionToDispose = transformToolState.previousSession?.selection;

    // 取消待执行的预览渲染
    cancelPendingRender();

    // 清除选区（保留已应用的变换）
    clearSelection();
    if (previousSelectionToDispose) {
        releaseSelectionResources(previousSelectionToDispose);
    }

    // 移除事件监听器
    if (transformToolState.eventsBound) {
        document.removeEventListener('pointerdown', onTransformPointerDown, true);
        document.removeEventListener('pointermove', onTransformPointerMove, true);
        document.removeEventListener('pointerup', onTransformPointerUp, true);
        transformToolState.eventsBound = false;
    }

    // 销毁自定义光标
    destroyCustomCursor();

    // 重置光标
    resetCursor();

    // 重置状态
    transformToolState.stage = 'idle';
    transformToolState.selection = null;
    transformToolState.transform = null;
    transformToolState.drag = { active: false };

    // 完全退出工具时清除所有状态
    transformToolState.previousSession = null;
    releaseCanvasBuffer(reusableTransformCanvas);
    reusableTransformCanvas = null;

    // 重置预览状态
    transformToolState.preview.lastRenderTime = 0;
    transformToolState.preview.pendingRender = false;
}

/**
 * 检查 Transform 工具是否激活
 */
function isTransformActive() {
    return transformToolState.eventsBound;
}


// === Exports ===

export {
    // 工具管理
    initTransformToolEvents,
    cleanupTransform,
    isTransformActive,
};
