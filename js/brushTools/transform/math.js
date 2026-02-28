import { getCanvasScale } from "../../utils.js";
import { getSharedOverlay, getSharedPaintCanvas } from "../common/sharedCanvasRefs.js";

/**
 * 将屏幕长度转换为 canvas 坐标长度
 * @param {number} screenLength - 屏幕上的长度（像素）
 * @returns {number} canvas 坐标系中的长度
 */
function screenToCanvasLength(screenLength) {
    const overlay = getSharedOverlay();
    if (!overlay) return screenLength;
    const scale = getCanvasScale(overlay);
    // 使用平均缩放比例，保持圆形不变形
    return screenLength * (scale.x + scale.y) / 2;
}

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
            return { type: "corner", index: i };
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
        return { type: "rotate", index: -1 };
    }

    // 3. 检测四边 - 10px 阈值（屏幕坐标）
    const edgeThreshold = screenToCanvasLength(10);
    for (let i = 0; i < 4; i++) {
        const next = (i + 1) % 4;
        const dist = pointToSegmentDistance(pos, corners[i], corners[next]);
        if (dist <= edgeThreshold) {
            return { type: "edge", index: i };
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
 * 判断两组角点是否有可感知变化
 */
function hasCornersChanged(beforeCorners, afterCorners, epsilon = 1e-3) {
    if (!beforeCorners || !afterCorners || beforeCorners.length !== afterCorners.length) return false;
    for (let i = 0; i < beforeCorners.length; i++) {
        if (Math.abs(beforeCorners[i].x - afterCorners[i].x) > epsilon ||
            Math.abs(beforeCorners[i].y - afterCorners[i].y) > epsilon) {
            return true;
        }
    }
    return false;
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
 * 将浮点矩形对齐到 canvas 像素网格，并裁剪到画布范围内
 */
function snapRectToCanvasPixels(rect, canvasWidth, canvasHeight) {
    const left = Math.max(0, Math.floor(rect.x));
    const top = Math.max(0, Math.floor(rect.y));
    const right = Math.min(canvasWidth, Math.ceil(rect.x + rect.width));
    const bottom = Math.min(canvasHeight, Math.ceil(rect.y + rect.height));
    return {
        x: left,
        y: top,
        width: Math.max(0, right - left),
        height: Math.max(0, bottom - top)
    };
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

export {
    screenToCanvasLength,
    isPointInQuad,
    pointToSegmentDistance,
    detectHandle,
    getBounds,
    hasCornersChanged,
    normalizeRect,
    snapRectToCanvasPixels,
    getSourceCorners,
};
