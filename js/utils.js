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
