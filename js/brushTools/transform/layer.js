import { getSharedBaseCanvas, getSharedPaintCanvas } from "../common/sharedCanvasRefs.js";

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
    const canvas = document.createElement("canvas");
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext("2d");
    ctx.putImageData(imageData, 0, 0);
    return canvas;
}

/**
 * 释放 canvas 后备存储，帮助浏览器尽快回收大块像素内存
 */
function releaseCanvasBuffer(canvas) {
    if (!canvas) return;
    canvas.width = 0;
    canvas.height = 0;
}

/**
 * 释放选区中持有的临时资源
 */
function releaseSelectionResources(selection) {
    if (!selection) return;
    releaseCanvasBuffer(selection.sourceCanvas);
    selection.sourceCanvas = null;
}

/**
 * 将未提交的 paint 选区内容回写到 paint 层。
 * 调用方负责保证该选区属于“仅剪切未变换”场景。
 */
function restoreUntransformedPaintSelection(selection) {
    if (!selection || !selection.cutFromPaint || !selection.sourceCanvas) {
        return;
    }

    const paintCanvas = getSharedPaintCanvas();
    if (!paintCanvas) return;

    const paintCtx = paintCanvas.getContext("2d");
    // Eraser may leave destination-out on context; force normal compositing for restore.
    paintCtx.save();
    paintCtx.globalCompositeOperation = "source-over";
    paintCtx.globalAlpha = 1;
    paintCtx.drawImage(selection.sourceCanvas, selection.rect.x, selection.rect.y);
    paintCtx.restore();
}

/**
 * 清除 paint 层指定区域
 */
function clearPaintRect(rect) {
    const paintCanvas = getSharedPaintCanvas();
    if (!paintCanvas) return;
    const ctx = paintCanvas.getContext("2d");
    ctx.clearRect(rect.x, rect.y, rect.width, rect.height);
}

/**
 * 按掩码清除 paint 层像素（仅清除 maskImageData 中 alpha>0 的像素）
 * 用于点击自动选择区域，区域包含其他"对象"的内容时，仅擦除本"对象"的内容
 */
function clearPaintByMask(rect, maskImageData) {
    const paintCanvas = getSharedPaintCanvas();
    if (!paintCanvas || !maskImageData) return;

    const ctx = paintCanvas.getContext("2d", { willReadFrequently: true });
    const paintData = ctx.getImageData(rect.x, rect.y, rect.width, rect.height);

    for (let i = 0; i < paintData.data.length; i += 4) {
        if (maskImageData.data[i + 3] > 0) {
            paintData.data[i] = 0;
            paintData.data[i + 1] = 0;
            paintData.data[i + 2] = 0;
            paintData.data[i + 3] = 0;
        }
    }

    ctx.putImageData(paintData, rect.x, rect.y);
}

/**
 * 采样 paint 层
 */
function samplePaintLayer(rect) {
    const paintCanvas = getSharedPaintCanvas();
    if (!paintCanvas) return null;
    const ctx = paintCanvas.getContext("2d", { willReadFrequently: true });
    return ctx.getImageData(rect.x, rect.y, rect.width, rect.height);
}

/**
 * 采样 base 层
 */
function sampleBaseLayer(rect) {
    const baseCanvas = getSharedBaseCanvas();
    if (!baseCanvas) return null;
    const ctx = baseCanvas.getContext("2d", { willReadFrequently: true });
    return ctx.getImageData(rect.x, rect.y, rect.width, rect.height);
}

/**
 * 在微小选区附近查找 paint 层非透明像素作为种子点
 */
function findPaintSeedAroundRect(rect, padding = 2) {
    const paintCanvas = getSharedPaintCanvas();
    if (!paintCanvas) return null;

    const minX = Math.max(0, Math.floor(rect.x) - padding);
    const minY = Math.max(0, Math.floor(rect.y) - padding);
    const maxX = Math.min(paintCanvas.width - 1, Math.ceil(rect.x + rect.width) + padding);
    const maxY = Math.min(paintCanvas.height - 1, Math.ceil(rect.y + rect.height) + padding);

    const width = maxX - minX + 1;
    const height = maxY - minY + 1;
    if (width <= 0 || height <= 0) return null;

    const ctx = paintCanvas.getContext("2d", { willReadFrequently: true });
    const data = ctx.getImageData(minX, minY, width, height).data;

    // 优先检查选区中心点（点击场景）
    const centerX = Math.round(rect.x + rect.width / 2) - minX;
    const centerY = Math.round(rect.y + rect.height / 2) - minY;
    if (centerX >= 0 && centerX < width && centerY >= 0 && centerY < height) {
        const centerIdx = (centerY * width + centerX) * 4 + 3;
        if (data[centerIdx] > 0) {
            return { x: centerX + minX, y: centerY + minY };
        }
    }

    // 其次扫描微小框附近
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const alpha = data[(y * width + x) * 4 + 3];
            if (alpha > 0) {
                return { x: x + minX, y: y + minY };
            }
        }
    }

    return null;
}

/**
 * 提取 paint 层中与种子点连通的对象（8 连通）
 */
function extractConnectedPaintObject(seedX, seedY) {
    const paintCanvas = getSharedPaintCanvas();
    if (!paintCanvas) return null;

    const width = paintCanvas.width;
    const height = paintCanvas.height;
    if (seedX < 0 || seedY < 0 || seedX >= width || seedY >= height) return null;

    const ctx = paintCanvas.getContext("2d", { willReadFrequently: true });
    const fullImageData = ctx.getImageData(0, 0, width, height);
    const pixels = fullImageData.data;

    const seedIndex = seedY * width + seedX;
    if (pixels[(seedIndex << 2) + 3] === 0) return null;

    const total = width * height;
    const visited = new Uint8Array(total);
    const componentMask = new Uint8Array(total);
    const queue = new Int32Array(total);
    let head = 0;
    let tail = 0;

    queue[tail++] = seedIndex;
    visited[seedIndex] = 1;

    let minX = seedX;
    let maxX = seedX;
    let minY = seedY;
    let maxY = seedY;

    while (head < tail) {
        const idx = queue[head++];
        const x = idx % width;
        const y = Math.floor(idx / width);
        componentMask[idx] = 1;

        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);

        for (let ny = y - 1; ny <= y + 1; ny++) {
            if (ny < 0 || ny >= height) continue;
            for (let nx = x - 1; nx <= x + 1; nx++) {
                if (nx < 0 || nx >= width) continue;
                if (nx === x && ny === y) continue;

                const nIdx = ny * width + nx;
                if (visited[nIdx]) continue;
                visited[nIdx] = 1;

                if (pixels[(nIdx << 2) + 3] > 0) {
                    queue[tail++] = nIdx;
                }
            }
        }
    }

    const rect = {
        x: minX,
        y: minY,
        width: maxX - minX + 1,
        height: maxY - minY + 1
    };
    if (rect.width <= 0 || rect.height <= 0) return null;

    const objectData = ctx.getImageData(rect.x, rect.y, rect.width, rect.height);
    for (let y = 0; y < rect.height; y++) {
        for (let x = 0; x < rect.width; x++) {
            const globalIdx = (rect.y + y) * width + (rect.x + x);
            if (componentMask[globalIdx]) continue;

            const localBase = (y * rect.width + x) * 4;
            objectData.data[localBase] = 0;
            objectData.data[localBase + 1] = 0;
            objectData.data[localBase + 2] = 0;
            objectData.data[localBase + 3] = 0;
        }
    }

    return { rect, imageData: objectData };
}

/**
 * 微小框选时自动选择 paint 连通对象
 */
function autoSelectPaintObjectFromTinyRect(rect) {
    const seed = findPaintSeedAroundRect(rect, 2);
    if (!seed) return null;
    return extractConnectedPaintObject(seed.x, seed.y);
}

/**
 * 设置 base 层变暗
 */
function setBaseLayerDimmed(dimmed) {
    const baseCanvas = getSharedBaseCanvas();
    if (!baseCanvas) return;
    if (dimmed) {
        baseCanvas.style.opacity = "0.4";
    } else {
        baseCanvas.style.opacity = "1";
    }
}

export {
    isImageDataEmpty,
    createSourceCanvas,
    releaseCanvasBuffer,
    releaseSelectionResources,
    restoreUntransformedPaintSelection,
    clearPaintRect,
    clearPaintByMask,
    samplePaintLayer,
    sampleBaseLayer,
    findPaintSeedAroundRect,
    extractConnectedPaintObject,
    autoSelectPaintObjectFromTinyRect,
    setBaseLayerDimmed,
};
