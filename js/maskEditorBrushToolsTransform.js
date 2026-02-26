import { getMaskEditorStore, displayToCanvas, getCanvasScale, showToast } from "./utils.js";

// === Shared Resources Access ===
let sharedOverlay = null;
let sharedBaseCanvas = null;
let sharedPaintCanvas = null;

/**
 * 设置共享的 overlay 画布（当 overlay 重新创建时调用）
 */
function setSharedOverlay(overlay) {
    sharedOverlay = overlay;
    sharedBaseCanvas = null;
    sharedPaintCanvas = null;
}

/**
 * 获取共享的 base 画布
 */
function getSharedBaseCanvas() {
    if (!sharedBaseCanvas) {
        const canvases = document.querySelectorAll('#maskEditorCanvasContainer canvas');
        sharedBaseCanvas = canvases[0];
    }
    return sharedBaseCanvas;
}

/**
 * 获取共享的 paint 画布
 */
function getSharedPaintCanvas() {
    if (!sharedPaintCanvas) {
        const canvases = document.querySelectorAll('#maskEditorCanvasContainer canvas');
        sharedPaintCanvas = canvases[1];
    }
    return sharedPaintCanvas;
}

/**
 * 将屏幕长度转换为 canvas 坐标长度
 * @param {number} screenLength - 屏幕上的长度（像素）
 * @returns {number} canvas 坐标系中的长度
 */
function screenToCanvasLength(screenLength) {
    const overlay = sharedOverlay;
    if (!overlay) return screenLength;
    const scale = getCanvasScale(overlay);
    // 使用平均缩放比例，保持圆形不变形
    return screenLength * (scale.x + scale.y) / 2;
}

// === Custom Cursor Element ===
let customCursorEl = null;

/**
 * 创建自定义光标元素
 */
function createCustomCursor() {
    if (customCursorEl) return;
    customCursorEl = document.createElement('div');
    customCursorEl.id = 'transform-tool-cursor';
    customCursorEl.style.cssText = `
        position: fixed;
        pointer-events: none;
        z-index: 99999;
        width: 32px;
        height: 32px;
        margin-left: -16px;
        margin-top: -16px;
        display: none;
    `;
    document.body.appendChild(customCursorEl);
}

/**
 * SVG cursor icons
 */
const cursorIcons = {
    // Crosshair with precision dot - for selection mode
    crosshair: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Outer circle -->
            <circle cx="16" cy="16" r="10" fill="none" stroke="#4a9eff" stroke-width="1.5" filter="url(#shadow)"/>
            <!-- Cross lines -->
            <line x1="16" y1="4" x2="16" y2="12" stroke="#4a9eff" stroke-width="1.5" filter="url(#shadow)"/>
            <line x1="16" y1="20" x2="16" y2="28" stroke="#4a9eff" stroke-width="1.5" filter="url(#shadow)"/>
            <line x1="4" y1="16" x2="12" y2="16" stroke="#4a9eff" stroke-width="1.5" filter="url(#shadow)"/>
            <line x1="20" y1="16" x2="28" y2="16" stroke="#4a9eff" stroke-width="1.5" filter="url(#shadow)"/>
            <!-- Center dot -->
            <circle cx="16" cy="16" r="1.5" fill="#4a9eff" filter="url(#shadow)"/>
        </svg>
    `,

    // Move - four-way arrows
    move: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Up arrow -->
            <polygon points="16,4 12,10 14,10 14,12 18,12 18,10 20,10" fill="#4a9eff" filter="url(#shadow)"/>
            <!-- Down arrow -->
            <polygon points="16,28 12,22 14,22 14,20 18,20 18,22 20,22" fill="#4a9eff" filter="url(#shadow)"/>
            <!-- Left arrow -->
            <polygon points="4,16 10,12 10,14 12,14 12,18 10,18 10,20" fill="#4a9eff" filter="url(#shadow)"/>
            <!-- Right arrow -->
            <polygon points="28,16 22,12 22,14 20,14 20,18 22,18 22,20" fill="#4a9eff" filter="url(#shadow)"/>
            <!-- Center dot -->
            <circle cx="16" cy="16" r="2" fill="#4a9eff" filter="url(#shadow)"/>
        </svg>
    `,

    // NW-SE resize (top-left to bottom-right)
    nwseResize: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Background circle -->
            <circle cx="16" cy="16" r="12" fill="rgba(255, 180, 60, 0.2)" stroke="#ffb43c" stroke-width="1.5" filter="url(#shadow)"/>
            <!-- Diagonal arrow (top-left to bottom-right) -->
            <line x1="8" y1="8" x2="24" y2="24" stroke="#ffb43c" stroke-width="2.5" stroke-linecap="round" filter="url(#shadow)"/>
            <polygon points="6,14 6,6 14,6" fill="#ffb43c" filter="url(#shadow)"/>
            <polygon points="26,18 26,26 18,26" fill="#ffb43c" filter="url(#shadow)"/>
        </svg>
    `,

    // NE-SW resize (top-right to bottom-left)
    neswResize: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Background circle -->
            <circle cx="16" cy="16" r="12" fill="rgba(255, 180, 60, 0.2)" stroke="#ffb43c" stroke-width="1.5" filter="url(#shadow)"/>
            <!-- Diagonal arrow (top-right to bottom-left) -->
            <line x1="24" y1="8" x2="8" y2="24" stroke="#ffb43c" stroke-width="2.5" stroke-linecap="round" filter="url(#shadow)"/>
            <polygon points="26,14 26,6 18,6" fill="#ffb43c" filter="url(#shadow)"/>
            <polygon points="6,18 6,26 14,26" fill="#ffb43c" filter="url(#shadow)"/>
        </svg>
    `,

    // NS resize (vertical)
    nsResize: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Background circle -->
            <circle cx="16" cy="16" r="12" fill="rgba(100, 220, 120, 0.2)" stroke="#64dc78" stroke-width="1.5" filter="url(#shadow)"/>
            <!-- Vertical arrows -->
            <line x1="16" y1="6" x2="16" y2="26" stroke="#64dc78" stroke-width="2.5" stroke-linecap="round" filter="url(#shadow)"/>
            <polygon points="16,4 11,10 21,10" fill="#64dc78" filter="url(#shadow)"/>
            <polygon points="16,28 11,22 21,22" fill="#64dc78" filter="url(#shadow)"/>
        </svg>
    `,

    // EW resize (horizontal)
    ewResize: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Background circle -->
            <circle cx="16" cy="16" r="12" fill="rgba(100, 220, 120, 0.2)" stroke="#64dc78" stroke-width="1.5" filter="url(#shadow)"/>
            <!-- Horizontal arrows -->
            <line x1="6" y1="16" x2="26" y2="16" stroke="#64dc78" stroke-width="2.5" stroke-linecap="round" filter="url(#shadow)"/>
            <polygon points="4,16 10,11 10,21" fill="#64dc78" filter="url(#shadow)"/>
            <polygon points="28,16 22,11 22,21" fill="#64dc78" filter="url(#shadow)"/>
        </svg>
    `,

    // Rotate (grab)
    rotate: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Background circle -->
            <circle cx="16" cy="16" r="12" fill="rgba(255, 100, 180, 0.2)" stroke="#ff64b4" stroke-width="1.5" filter="url(#shadow)"/>
            <!-- Rotation arrow arc -->
            <path d="M 10,16 A 6,6 0 1,1 22,16" fill="none" stroke="#ff64b4" stroke-width="2.5" stroke-linecap="round" filter="url(#shadow)"/>
            <!-- Arrow head -->
            <polygon points="22,12 26,18 18,18" fill="#ff64b4" filter="url(#shadow)"/>
            <!-- Center dot -->
            <circle cx="16" cy="16" r="2" fill="#ff64b4" filter="url(#shadow)"/>
        </svg>
    `,

    // Rotating (grabbing) - filled version
    rotating: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <!-- Filled background circle -->
            <circle cx="16" cy="16" r="12" fill="#ff64b4" stroke="#ff64b4" stroke-width="1.5" filter="url(#shadow)"/>
            <!-- White arc -->
            <path d="M 10,16 A 6,6 0 1,1 22,16" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
            <!-- White arrow head -->
            <polygon points="22,12 26,18 18,18" fill="white"/>
            <!-- Center dot -->
            <circle cx="16" cy="16" r="2" fill="white"/>
        </svg>
    `,

    // Default - simple pointer dot
    default: `
        <svg width="32" height="32" viewBox="0 0 32 32">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="1" flood-color="black" flood-opacity="0.8"/>
                </filter>
            </defs>
            <circle cx="16" cy="16" r="4" fill="#4a9eff" filter="url(#shadow)"/>
        </svg>
    `
};

/**
 * 销毁自定义光标元素
 */
function destroyCustomCursor() {
    if (customCursorEl) {
        customCursorEl.remove();
        customCursorEl = null;
    }
}

/**
 * 更新自定义光标位置和样式
 */
function updateCustomCursor(clientX, clientY, cursorType) {
    if (!customCursorEl) return;

    customCursorEl.style.left = clientX + 'px';
    customCursorEl.style.top = clientY + 'px';

    // 根据 cursorType 更新光标 SVG
    let svg = cursorIcons.default;
    switch (cursorType) {
        case 'crosshair':
            svg = cursorIcons.crosshair;
            break;
        case 'move':
            svg = cursorIcons.move;
            break;
        case 'nwse-resize':
            svg = cursorIcons.nwseResize;
            break;
        case 'nesw-resize':
            svg = cursorIcons.neswResize;
            break;
        case 'ns-resize':
            svg = cursorIcons.nsResize;
            break;
        case 'ew-resize':
            svg = cursorIcons.ewResize;
            break;
        case 'grab':
            svg = cursorIcons.rotate;
            break;
        case 'grabbing':
            svg = cursorIcons.rotating;
            break;
        default:
            svg = cursorIcons.default;
    }
    customCursorEl.innerHTML = svg;
}

/**
 * 显示/隐藏自定义光标
 */
function showCustomCursor(show) {
    if (customCursorEl) {
        customCursorEl.style.display = show ? 'block' : 'none';
    }
}

// === Transform Tool State ===
const transformToolState = {
    // 当前阶段
    stage: 'idle', // 'idle' | 'selecting' | 'transforming'

    // 选区数据（仅在 selecting/transforming 时存在）
    selection: null,

    // 变换状态（仅在 transforming 时存在）
    transform: null,

    // 拖动状态
    drag: {
        active: false,
        type: null, // 'move' | 'corner' | 'edge' | 'rotate'
        targetIndex: -1,
        startMouse: { x: 0, y: 0 },
        startTransform: null
    },

    // 可复用的临时画布
    tempCanvas: null,

    // 事件绑定状态
    eventsBound: false,

    // 最后的鼠标位置
    lastMousePos: null,

    // 保存的旧选区状态（用于新选区无效时恢复）
    previousSelection: null,
    previousTransform: null
};

// === Math Utilities ===

/**
 * 检测点是否在四边形内部（射线法）
 */
function isPointInQuad(point, corners) {
    let inside = false;
    for (let i = 0, j = corners.length - 1; i < corners.length; j = i++) {
        if (((corners[i].y > point.y) !== (corners[j].y > point.y)) &&
            (point.x < (corners[j].x - corners[i].x) * (point.y - corners[i].y) / (corners[j].y - corners[i].y) + corners[i].x)) {
            inside = !inside;
        }
    }
    return inside;
}

/**
 * 计算点到线段的距离
 */
function pointToSegmentDistance(point, a, b) {
    const A = point.x - a.x;
    const B = point.y - a.y;
    const C = b.x - a.x;
    const D = b.y - a.y;

    const dot = A * C + B * D;
    const lenSq = C * C + D * D;
    let param = -1;
    if (lenSq !== 0) {
        param = dot / lenSq;
    }

    let xx, yy;
    if (param < 0) {
        xx = a.x;
        yy = a.y;
    } else if (param > 1) {
        xx = b.x;
        yy = b.y;
    } else {
        xx = a.x + param * C;
        yy = a.y + param * D;
    }

    const dx = point.x - xx;
    const dy = point.y - yy;
    return Math.sqrt(dx * dx + dy * dy);
}

/**
 * 检测句柄（优先级：四角 > 旋转手柄 > 四边 > 选区内部）
 */
function detectHandle(pos, transform) {
    const corners = transform.corners;

    // 1. 检测四角（最高优先级）- 15px 阈值（屏幕坐标）
    const cornerThreshold = screenToCanvasLength(15);
    for (let i = 0; i < 4; i++) {
        const dist = Math.hypot(pos.x - corners[i].x, pos.y - corners[i].y);
        if (dist <= cornerThreshold) {
            return { type: 'corner', index: i };
        }
    }

    // 2. 检测旋转手柄 - 10px 阈值（屏幕坐标）
    const handleLength = screenToCanvasLength(30);  // 30px 屏幕长度
    const topCenter = {
        x: (corners[0].x + corners[1].x) / 2,
        y: (corners[0].y + corners[1].y) / 2
    };
    const rotateHandle = {
        x: topCenter.x,
        y: topCenter.y - handleLength
    };
    const rotateDist = Math.hypot(pos.x - rotateHandle.x, pos.y - rotateHandle.y);
    if (rotateDist <= screenToCanvasLength(10)) {
        return { type: 'rotate', index: -1 };
    }

    // 3. 检测四边 - 10px 阈值（屏幕坐标）
    const edgeThreshold = screenToCanvasLength(10);
    for (let i = 0; i < 4; i++) {
        const next = (i + 1) % 4;
        const dist = pointToSegmentDistance(pos, corners[i], corners[next]);
        if (dist <= edgeThreshold) {
            return { type: 'edge', index: i };
        }
    }

    return null;
}

/**
 * 计算四边形的边界框，并确保在画布范围内
 */
function getBounds(corners) {
    let minX = corners[0].x, maxX = corners[0].x;
    let minY = corners[0].y, maxY = corners[0].y;

    for (let i = 1; i < 4; i++) {
        minX = Math.min(minX, corners[i].x);
        maxX = Math.max(maxX, corners[i].x);
        minY = Math.min(minY, corners[i].y);
        maxY = Math.max(maxY, corners[i].y);
    }

    // 获取画布尺寸进行边界检查
    const paintCanvas = getSharedPaintCanvas();
    const canvasWidth = paintCanvas ? paintCanvas.width : 1024;
    const canvasHeight = paintCanvas ? paintCanvas.height : 1024;

    // 确保边界在画布范围内
    const x = Math.max(0, Math.floor(minX));
    const y = Math.max(0, Math.floor(minY));
    const maxRight = Math.min(canvasWidth, Math.ceil(maxX));
    const maxBottom = Math.min(canvasHeight, Math.ceil(maxY));

    return {
        x: x,
        y: y,
        width: Math.max(0, maxRight - x),
        height: Math.max(0, maxBottom - y)
    };
}

/**
 * 检测图像数据是否为空
 */
function isImageDataEmpty(imageData, minOpaquePixels) {
    let opaqueCount = 0;
    for (let i = 3; i < imageData.data.length; i += 4) {
        if (imageData.data[i] > 0) {
            opaqueCount++;
            if (opaqueCount >= minOpaquePixels) {
                return false;
            }
        }
    }
    return true;
}

/**
 * 创建源图像画布
 * 把 imageData 转成 canvas, 作为后续变换的唯一源，避免被清空或重复变换污染
 */
function createSourceCanvas(imageData) {
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext('2d');
    ctx.putImageData(imageData, 0, 0);
    return canvas;
}

/**
 * 清除 paint 层指定区域
 */
function clearPaintRect(rect) {
    const paintCanvas = getSharedPaintCanvas();
    if (!paintCanvas) return;
    const ctx = paintCanvas.getContext('2d');
    ctx.clearRect(rect.x, rect.y, rect.width, rect.height);
}

/**
 * 从 base 层复制到 paint 层
 */
function copyToPaintLayer(rect, imageData) {
    const paintCanvas = getSharedPaintCanvas();
    if (!paintCanvas) return;
    const ctx = paintCanvas.getContext('2d');
    ctx.putImageData(imageData, rect.x, rect.y);
    getMaskEditorStore()?.canvasHistory?.saveState?.();
    // console.log('Saved state copyToPaintLayer');
}

/**
 * 采样 paint 层
 */
function samplePaintLayer(rect) {
    const paintCanvas = getSharedPaintCanvas();
    if (!paintCanvas) return null;
    const ctx = paintCanvas.getContext('2d', { willReadFrequently: true });
    return ctx.getImageData(rect.x, rect.y, rect.width, rect.height);
}

/**
 * 采样 base 层
 */
function sampleBaseLayer(rect) {
    const baseCanvas = getSharedBaseCanvas();
    if (!baseCanvas) return null;
    const ctx = baseCanvas.getContext('2d', { willReadFrequently: true });
    return ctx.getImageData(rect.x, rect.y, rect.width, rect.height);
}

/**
 * 设置 base 层变暗
 */
function setBaseLayerDimmed(dimmed) {
    const baseCanvas = getSharedBaseCanvas();
    if (!baseCanvas) return;
    if (dimmed) {
        baseCanvas.style.opacity = '0.4';
    } else {
        baseCanvas.style.opacity = '1';
    }
}

// === Perspective Transform Utilities ===

/**
 * 双线性插值
 */
function bilinearInterpolate(p0, p1, p2, p3, u, v) {
    const p01 = {
        x: p0.x + (p1.x - p0.x) * u,
        y: p0.y + (p1.y - p0.y) * u
    };
    const p32 = {
        x: p3.x + (p2.x - p3.x) * u,
        y: p3.y + (p2.y - p3.y) * u
    };
    return {
        x: p01.x + (p32.x - p01.x) * v,
        y: p01.y + (p32.y - p01.y) * v
    };
}

/**
 * 计算三角形仿射变换矩阵并应用到上下文
 */
function applyTriangleTransform(ctx, s0, s1, s2, d0, d1, d2) {
    // 计算源三角形的变换矩阵
    const denom = (s0.x * (s1.y - s2.y) - s1.x * (s0.y - s2.y) + s2.x * (s0.y - s1.y));
    if (Math.abs(denom) < 1e-10) return false;

    // 计算目标三角形的变换
    const m = new DOMMatrix();

    // 使用标准仿射变换: 3个点确定一个仿射变换
    const srcX0 = s0.x, srcY0 = s0.y;
    const srcX1 = s1.x, srcY1 = s1.y;
    const srcX2 = s2.x, srcY2 = s2.y;
    const dstX0 = d0.x, dstY0 = d0.y;
    const dstX1 = d1.x, dstY1 = d1.y;
    const dstX2 = d2.x, dstY2 = d2.y;

    // 计算从源到目标的变换矩阵
    const srcDet = (srcX0 - srcX2) * (srcY1 - srcY2) - (srcX1 - srcX2) * (srcY0 - srcY2);
    if (Math.abs(srcDet) < 1e-10) return false;

    const a11 = ((dstX0 - dstX2) * (srcY1 - srcY2) - (dstX1 - dstX2) * (srcY0 - srcY2)) / srcDet;
    const a12 = ((dstX1 - dstX2) * (srcX0 - srcX2) - (dstX0 - dstX2) * (srcX1 - srcX2)) / srcDet;
    const a21 = ((dstY0 - dstY2) * (srcY1 - srcY2) - (dstY1 - dstY2) * (srcY0 - srcY2)) / srcDet;
    const a22 = ((dstY1 - dstY2) * (srcX0 - srcX2) - (dstY0 - dstY2) * (srcX1 - srcX2)) / srcDet;
    const tx = dstX2 - a11 * srcX2 - a12 * srcY2;
    const ty = dstY2 - a21 * srcX2 - a22 * srcY2;

    m.a = a11; m.b = a21; m.c = a12; m.d = a22; m.e = tx; m.f = ty;
    ctx.setTransform(m);
    return true;
}

/**
 * 绘制透视四边形（网格细分法）
 * @param {CanvasRenderingContext2D} ctx - 目标画布上下文
 * @param {HTMLCanvasElement} srcCanvas - 源图像画布
 * @param {Array} srcQuad - 源四边形坐标 [{x,y}, {x,y}, {x,y}, {x,y}] (左上,右上,右下,左下)
 * @param {Array} dstQuad - 目标四边形坐标 [{x,y}, {x,y}, {x,y}, {x,y}] (左上,右上,右下,左下)
 * @param {number} gridSize - 网格细分数量（默认20）
 */
function drawPerspectiveQuad(ctx, srcCanvas, srcQuad, dstQuad, gridSize = 20) {
    if (!srcCanvas || !srcQuad || !dstQuad || srcQuad.length !== 4 || dstQuad.length !== 4) return;

    const srcW = srcCanvas.width;
    const srcH = srcCanvas.height;

    // 保存当前变换
    ctx.save();

    // 网格细分绘制
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const u0 = i / gridSize;
            const u1 = (i + 1) / gridSize;
            const v0 = j / gridSize;
            const v1 = (j + 1) / gridSize;

            // 计算源网格的四个角（归一化坐标）
            const s00 = bilinearInterpolate(srcQuad[0], srcQuad[1], srcQuad[2], srcQuad[3], u0, v0);
            const s10 = bilinearInterpolate(srcQuad[0], srcQuad[1], srcQuad[2], srcQuad[3], u1, v0);
            const s11 = bilinearInterpolate(srcQuad[0], srcQuad[1], srcQuad[2], srcQuad[3], u1, v1);
            const s01 = bilinearInterpolate(srcQuad[0], srcQuad[1], srcQuad[2], srcQuad[3], u0, v1);

            // 计算目标网格的四个角
            const d00 = bilinearInterpolate(dstQuad[0], dstQuad[1], dstQuad[2], dstQuad[3], u0, v0);
            const d10 = bilinearInterpolate(dstQuad[0], dstQuad[1], dstQuad[2], dstQuad[3], u1, v0);
            const d11 = bilinearInterpolate(dstQuad[0], dstQuad[1], dstQuad[2], dstQuad[3], u1, v1);
            const d01 = bilinearInterpolate(dstQuad[0], dstQuad[1], dstQuad[2], dstQuad[3], u0, v1);

            // 将四边形分成两个三角形绘制
            // 三角形1: s00 -> s10 -> s11
            ctx.save();
            if (applyTriangleTransform(ctx, s00, s10, s11, d00, d10, d11)) {
                // 计算三角形在源图像中的包围盒
                const minX = Math.floor(Math.max(0, Math.min(s00.x, s10.x, s11.x)));
                const minY = Math.floor(Math.max(0, Math.min(s00.y, s10.y, s11.y)));
                const maxX = Math.ceil(Math.min(srcW, Math.max(s00.x, s10.x, s11.x) + 1));
                const maxY = Math.ceil(Math.min(srcH, Math.max(s00.y, s10.y, s11.y) + 1));
                const w = maxX - minX;
                const h = maxY - minY;
                if (w > 0 && h > 0) {
                    ctx.drawImage(srcCanvas, minX, minY, w, h, minX, minY, w, h);
                }
            }
            ctx.restore();

            // 三角形2: s00 -> s11 -> s01
            ctx.save();
            if (applyTriangleTransform(ctx, s00, s11, s01, d00, d11, d01)) {
                const minX = Math.floor(Math.max(0, Math.min(s00.x, s11.x, s01.x)));
                const minY = Math.floor(Math.max(0, Math.min(s00.y, s11.y, s01.y)));
                const maxX = Math.ceil(Math.min(srcW, Math.max(s00.x, s11.x, s01.x) + 1));
                const maxY = Math.ceil(Math.min(srcH, Math.max(s00.y, s11.y, s01.y) + 1));
                const w = maxX - minX;
                const h = maxY - minY;
                if (w > 0 && h > 0) {
                    ctx.drawImage(srcCanvas, minX, minY, w, h, minX, minY, w, h);
                }
            }
            ctx.restore();
        }
    }

    // 恢复变换
    ctx.restore();
}

/**
 * 获取源四边形坐标（从选区矩形）
 */
function getSourceCorners(rect) {
    return [
        { x: 0, y: 0 },
        { x: rect.width, y: 0 },
        { x: rect.width, y: rect.height },
        { x: 0, y: rect.height }
    ];
}

/**
 * 获取光标样式
 */
function getCursorForHandle(handle) {
    if (!handle) return 'default';
    switch (handle.type) {
        case 'corner':
            // 根据角点索引返回对应光标
            const cornerCursors = ['nwse-resize', 'nesw-resize', 'nwse-resize', 'nesw-resize'];
            return cornerCursors[handle.index] || 'nwse-resize';
        case 'rotate':
            return 'grab';
        case 'edge':
            // 根据边的方向返回光标 (0=上,1=右,2=下,3=左)
            const edgeCursors = ['ns-resize', 'ew-resize', 'ns-resize', 'ew-resize'];
            return edgeCursors[handle.index] || 'move';
        case 'move':
            return 'move';
        default:
            return 'default';
    }
}

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
    const overlay = sharedOverlay;
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
    console.log("[Transform Tool] Events bound to document (capture)");
}

/**
 * 处理指针按下事件
 */
function onTransformPointerDown(e) {
    if (!isTransformActive()) return;

    // 获取共享资源
    const overlay = sharedOverlay;
    if (!overlay) return;

    // 检查是否在画布区域内
    const rect = overlay.getBoundingClientRect();
    if (e.clientX < rect.left || e.clientX > rect.right ||
        e.clientY < rect.top || e.clientY > rect.bottom) return;

    e.stopImmediatePropagation();
    e.preventDefault();

    const pos = displayToCanvas(overlay, e.clientX, e.clientY);

    if (transformToolState.stage === 'transforming') {
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
        transformToolState.previousSelection = transformToolState.selection;
        transformToolState.previousTransform = transformToolState.transform;
        clearSelection(false);  // false: 不还原像素
        transformToolState.stage = 'selecting';
        startSelection(pos);
    } else if (transformToolState.stage === 'selecting') {
        // 如果正在框选中（已按下未松开），继续；否则开始新框选
        if (!selectionStart) {
            startSelection(pos);
        }
    } else if (transformToolState.stage === 'idle') {
        // 从 idle 进入 selecting
        transformToolState.stage = 'selecting';
        startSelection(pos);
    }
}

/**
 * 处理指针移动事件
 */
function onTransformPointerMove(e) {
    if (!isTransformActive()) return;

    const overlay = sharedOverlay;
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

    const overlay = sharedOverlay;
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
 * 标准化矩形（确保宽高为正）
 */
function normalizeRect(start, end) {
    return {
        x: Math.min(start.x, end.x),
        y: Math.min(start.y, end.y),
        width: Math.abs(end.x - start.x),
        height: Math.abs(end.y - start.y)
    };
}

/**
 * 完成框选
 */
function finalizeSelection() {
    if (!selectionStart) return;

    const currentPos = transformToolState.lastMousePos || selectionStart;
    const rect = normalizeRect(selectionStart, currentPos);

    // 过滤过小的选区
    if (rect.width < 5 || rect.height < 5) {
        restorePreviousSelection();
        return;
    }

    // 检测选区内容
    const paintData = samplePaintLayer(rect);
    const minPixels = Math.max(16, Math.floor(rect.width * rect.height * 0.005));
    const isEmpty = isImageDataEmpty(paintData, minPixels);

    let sourceData = paintData;
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

        // 从 base 层复制到 paint 层
        copyToPaintLayer(rect, baseData);
        sourceData = baseData;

        showToast('Empty selection, auto-copied from base layer (Ctrl+Z to undo the copy)', {
            duration: 3000
        });
    }

    transformToolState.selection = {
        rect: rect,
        sourceCanvas: createSourceCanvas(sourceData),
        lastAppliedBounds: null,
        paintSnapshot: null,
        hasTransformed: false,
        stateSaved: false
    };

    // 剪切：从 paint 层清除选区像素
    clearPaintRect(rect);

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

    // 清空保存的旧选区（新选区已创建成功，不需要再恢复旧选区）
    transformToolState.previousSelection = null;
    transformToolState.previousTransform = null;
}

/**
 * 取消框选
 */
function cancelSelection() {
    selectionStart = null;
    setBaseLayerDimmed(false);
    transformToolState.stage = 'idle';
    const overlay = sharedOverlay;
    if (overlay) {
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }
    resetCursor();
}

/**
 * 恢复之前的选区（新选区无效时调用）
 */
function restorePreviousSelection() {
    selectionStart = null;
    setBaseLayerDimmed(false);

    // 清空 overlay
    const overlay = sharedOverlay;
    if (overlay) {
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    // 恢复旧选区
    if (transformToolState.previousSelection && transformToolState.previousTransform) {
        transformToolState.selection = transformToolState.previousSelection;
        transformToolState.transform = transformToolState.previousTransform;
        transformToolState.stage = 'transforming';

        // 将旧选区像素还原到 paint 层（因为创建选区时清除了）
        const paintCanvas = getSharedPaintCanvas();
        if (paintCanvas && transformToolState.selection.sourceCanvas) {
            const paintCtx = paintCanvas.getContext('2d');
            paintCtx.drawImage(
                transformToolState.selection.sourceCanvas,
                transformToolState.selection.rect.x,
                transformToolState.selection.rect.y
            );
        }

        // 重绘
        renderTransforming();

        showToast('New selection invalid, restored previous selection', { duration: 2000 });
    } else {
        transformToolState.stage = 'idle';
        resetCursor();
    }

    // 清空保存的状态
    transformToolState.previousSelection = null;
    transformToolState.previousTransform = null;
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
}

/**
 * 结束拖动
 */
function endDrag() {
    if (transformToolState.drag.active) {
        applyTransform();
        transformToolState.drag.active = false;
        renderTransforming();

        // 保存历史（仅在未保存过时）
        const { selection } = transformToolState;
        if (selection && !selection.stateSaved) {
            getMaskEditorStore()?.canvasHistory?.saveState?.();
            selection.stateSaved = true;
            // console.log('Saved state endDrag');
        }
    }
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

    // 1. 清除上一次变换写入的区域
    if (selection.lastAppliedBounds && selection.paintSnapshot) {
        const paintCtx = paintCanvas.getContext('2d');
        paintCtx.putImageData(selection.paintSnapshot,
            selection.lastAppliedBounds.x, selection.lastAppliedBounds.y);
    }

    // 2. 计算本次变换的 bounding box
    const bounds = getBounds(corners);

    // 如果变换后的区域无效，跳过
    if (bounds.width <= 0 || bounds.height <= 0) return;

    // 3. 保存 paint 层在新 bounds 区域的快照
    const paintCtx = paintCanvas.getContext('2d', { willReadFrequently: true });
    selection.paintSnapshot = paintCtx.getImageData(
        bounds.x, bounds.y, bounds.width, bounds.height);

    // 4. 渲染到临时画布
    if (!transformToolState.tempCanvas) {
        transformToolState.tempCanvas = document.createElement('canvas');
    }
    transformToolState.tempCanvas.width = bounds.width;
    transformToolState.tempCanvas.height = bounds.height;
    const tempCtx = transformToolState.tempCanvas.getContext('2d');
    tempCtx.clearRect(0, 0, bounds.width, bounds.height);

    // 计算目标四边形（相对于临时画布的本地坐标）
    const localCorners = corners.map(c => ({
        x: c.x - bounds.x,
        y: c.y - bounds.y
    }));

    // 源四边形（sourceCanvas 的完整区域）
    const srcQuad = getSourceCorners(selection.rect);

    // 使用透视变换渲染
    drawPerspectiveQuad(tempCtx, selection.sourceCanvas, srcQuad, localCorners, 20);

    // 5. 合成到 paint 层
    paintCtx.drawImage(transformToolState.tempCanvas, bounds.x, bounds.y);

    // 6. 记录本次写入区域
    selection.lastAppliedBounds = bounds;
    selection.hasTransformed = true;
}

// === Rendering ===

/**
 * 渲染 Transform 模式
 */
function renderTransforming() {
    const overlay = sharedOverlay;
    if (!overlay || transformToolState.stage !== 'transforming') return;

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const corners = transformToolState.transform.corners;

    // 绘制变换后的图像轮廓
    if (!transformToolState.drag.active || transformToolState.drag.type === 'move') {
        // 移动操作可以实时渲染
        drawTransformedImage(ctx, corners);
    } else {
        // 其他操作只绘制轮廓
        drawTransformOutline(ctx, corners);
    }

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

    // 计算边界框用于裁剪
    const bounds = getBounds(corners);

    // 源四边形（sourceCanvas 的完整区域）
    const srcQuad = getSourceCorners(selection.rect);

    // 使用透视变换渲染到 overlay
    drawPerspectiveQuad(ctx, selection.sourceCanvas, srcQuad, corners, 20);

    // 绘制轮廓线
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
 * @param {boolean} restorePixels - 是否在未变换时还原像素到 paint 层（默认 true）
 */
function clearSelection(restorePixels = true) {
    const { selection } = transformToolState;

    if (selection) {
        if (restorePixels && !selection.hasTransformed) {
            // 用户框选后未做任何变换就放弃——还原像素到 paint 层原位
            const paintCanvas = getSharedPaintCanvas();
            if (paintCanvas) {
                const paintCtx = paintCanvas.getContext('2d');
                paintCtx.drawImage(selection.sourceCanvas,
                    selection.rect.x, selection.rect.y);
            }
        }
        // 保存历史（仅在确实发生过变换且未保存过时才有意义）
        if (selection.hasTransformed && !selection.stateSaved) {
            getMaskEditorStore()?.canvasHistory?.saveState?.();
            selection.stateSaved = true;
            // console.log('Saved state clearSelection');
        }
    }

    // 清空 overlay
    const overlay = sharedOverlay;
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

    // 注意：不清除 previousSelection/previousTransform
    // 它们用于在创建新选区失败时恢复旧选区
    // 在 restorePreviousSelection 或 finalizeSelection 成功后会清除
}


/**
 * 清理 Transform 工具
 */
function cleanupTransform() {
    // 清除选区（保留已应用的变换）
    clearSelection();

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
    transformToolState.previousSelection = null;
    transformToolState.previousTransform = null;
}

/**
 * 检查 Transform 工具是否激活
 */
function isTransformActive() {
    return transformToolState.eventsBound;
}


// === Exports ===

export {
    // 状态
    transformToolState,

    // 工具管理
    initTransformToolEvents,
    cleanupTransform,
    isTransformActive,
    setSharedOverlay,

    // 渲染
    renderTransforming
};
