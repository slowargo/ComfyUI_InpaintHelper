import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// === CSS Loading ===

export function loadCSS(metaUrl, filename) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL(filename, metaUrl).href;
    document.head.appendChild(link);
}

// === Path Utilities ===

export function parseFilePath(filepath) {
    if (!filepath?.trim()) return { filename: '', subfolder: '' }

    const normalizedPath = filepath
      .replace(/[\\/]+/g, '/') // Normalize path separators
      .replace(/^\//, '') // Remove leading slash
      .replace(/\/$/, '') // Remove trailing slash

    const lastSlashIndex = normalizedPath.lastIndexOf('/')

    if (lastSlashIndex === -1) {
        return {
            filename: normalizedPath,
            subfolder: ''
        }
    }

    return {
        filename: normalizedPath.slice(lastSlashIndex + 1),
        subfolder: normalizedPath.slice(0, lastSlashIndex)
    }
}

export function getComfyFilePathFromViewUrl(url) {
    try {
        const urlObj = new URL(url);
        const params = urlObj.searchParams;

        const type = params.get('type') || 'output';           // 默认 output
        const filename = params.get('filename');
        let subfolder = params.get('subfolder') || '';         // 可能为空

        if (!filename) {
            throw new Error("URL 中缺少 filename 参数");
        }

        // 拼接，subfolder 为空时自动处理双斜杠
        const path = [type, subfolder, filename]
            .filter(part => part !== undefined && part !== null && part !== '')
            .join('/')
            .replace(/\/+/g, '/');  // 防止多余斜杠

        return path;
    } catch (err) {
        console.error("[getComfyFilePath] 解析失败:", err);
        return null;
    }
}

// === Node Preview ===

export function updateNodePreview(node, imageName) {
    if (!imageName || !node) return;
    let { filename, subfolder } = parseFilePath(imageName)

    // 检查子文件夹是否为 "clipspace"
    if (subfolder === "clipspace") {
        // 检查文件名是否以 [output] 结尾
        if (filename.endsWith("[output]")) {
            filename = filename.replace("[output]", "[input]")
        }
    }

    console.log("[slowargo.js] updateNodePreview", imageName, subfolder, filename);
    imageName = filename;

    // Release old image if exists
    if (node.imgs?.[0]?.src) {
        node.imgs[0].src = '';
    }

    const img = new Image();
    img.onload = () => {
        node.imgs = [img];
        app.graph.setDirtyCanvas(true);
    };

    img.onerror = () => {
        console.warn(`Failed to load preview for ${imageName}`, node);
    };

    // Get the input directory path
    const inputPathWidget = node.widgets?.find(w => w.name === "input_path");
    const inputPath = inputPathWidget?.value || "";

    // Construct URL with proper path handling
    const params = `${app.getPreviewFormatParam?.() || ""}${app.getRandParam?.() || ""}`;

    // Use the thumbnail name directly since it's already in the input directory
    img.src = api.apiURL(`/view?filename=${encodeURIComponent(imageName)}${subfolder ? `&subfolder=${encodeURIComponent(subfolder)}` : ''}&type=output${params}`);
}

// === Async Utilities ===

export function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// === Pinia Store Access ===

export function getKeybindingStore() {
    try {
        const vueApp = document.querySelector('#vue-app')?.__vue_app__;
        if (!vueApp) return null;
        const pinia = vueApp.config.globalProperties?.$pinia;
        if (!pinia?._s) return null;
        return pinia._s.get('keybinding') || null;
    } catch (e) {
        return null;
    }
}

export function getMaskEditorStore() {
    try {
        const vueApp = document.querySelector('#vue-app')?.__vue_app__;
        if (!vueApp) return null;
        const pinia = vueApp.config.globalProperties?.$pinia;
        if (!pinia?._s) return null;
        return pinia._s.get('maskEditor') || null;
    } catch (e) {
        return null;
    }
}

export function getWorkflowStore() {
    try {
        const vueApp = document.querySelector('#vue-app')?.__vue_app__;
        if (!vueApp) return null;
        const pinia = vueApp.config.globalProperties?.$pinia;
        if (!pinia?._s) return null;
        return pinia._s.get('workflow') || null;
    } catch (e) {
        return null;
    }
}

export function getToastStore() {
    try {
        const vueApp = document.querySelector('#vue-app')?.__vue_app__;
        if (!vueApp) return null;
        const pinia = vueApp.config.globalProperties?.$pinia;
        if (!pinia?._s) return null;
        return pinia._s.get('toast') || null;
    } catch (e) {
        return null;
    }
}

// === Keybinding Utilities ===

export function eventMatchesCommand(event, commandId) {
    const store = getKeybindingStore();
    if (!store) return false;
    const bindings = store.getKeybindingsByCommandId(commandId);
    for (const binding of bindings) {
        const combo = binding.combo;
        if (combo.key.toUpperCase() === event.key.toUpperCase() &&
            combo.ctrl === (event.ctrlKey || event.metaKey) &&
            combo.alt === event.altKey &&
            combo.shift === event.shiftKey) {
            return true;
        }
    }
    return false;
}

// === Canvas Coordinate Utilities ===

/**
 * Map client (CSS) coordinates to canvas pixel coordinates.
 * Necessary because the canvas display size may differ from its intrinsic pixel size
 * (e.g. the canvas is rendered at 50% scale, so CSS pixels must be multiplied by 2).
 *
 * 坐标映射函数。需要进行坐标映射是因为 Canvas 的显示尺寸与实际像素尺寸可能不一致（画布可能以缩小/放大状态显示)
 * 例如缩小到 50%, rect.width 为 canvas.width 的一半，下面公式就相当于 offset * 2，放大回正确的像素位置
 *
 * @param {HTMLCanvasElement} canvas - The target canvas element
 * @param {number} clientX - Client X coordinate from the pointer event
 * @param {number} clientY - Client Y coordinate from the pointer event
 * @returns {{x: number, y: number}} Canvas pixel coordinates
 */
export function displayToCanvas(canvas, clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (clientX - rect.left) * canvas.width  / rect.width,
        y: (clientY - rect.top)  * canvas.height / rect.height,
    };
}

/**
 * 获取 canvas 缩放比例（canvas 像素尺寸 / CSS 显示尺寸）
 * @param {HTMLCanvasElement} canvas
 * @returns {{x: number, y: number}}
 */
export function getCanvasScale(canvas) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: canvas.width / rect.width,
        y: canvas.height / rect.height
    };
}

/**
 * 显示 Toast 消息
 * @param {string} message
 * @param {{duration?: number}} [options]
 */
export function showToast(message, options = {}) {
    console.log(`[Transform Tool] ${message}`);

    // 尝试使用现有的 toast 系统
    try {
        const toastStore = getToastStore();
        if (toastStore?.add) {
            toastStore.add({
                severity: 'info',
                summary: message,
                life: options.duration || 3000
            });
            return;
        }
    } catch (e) {
        // 忽略错误，继续尝试其他方法
    }

    // 备用方案：创建简单的 toast 元素
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 12px 16px;
        border-radius: 4px;
        z-index: 10000;
        font-size: 14px;
        max-width: 300px;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, options.duration || 3000);
}

// === Mask Canvas Utilities ===

export function isMaskNonEmpty() {
    const canvases = document.querySelectorAll("#maskEditorCanvasContainer canvas");
    if (canvases.length < 3) return false;

    const maskCanvas = canvases[2]; // z-30, the mask layer
    const ctx = maskCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return false;

    const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const data = imageData.data;

    // Sample every 16th pixel for performance
    const step = 4 * 16;
    for (let i = 3; i < data.length; i += step) {
        if (data[i] > 0) return true;
    }
    return false;
}
