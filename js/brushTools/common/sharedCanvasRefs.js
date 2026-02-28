let sharedOverlay = null;
let sharedBaseCanvas = null;
let sharedPaintCanvas = null;

function setSharedCanvases(baseCanvas, paintCanvas) {
    sharedBaseCanvas = baseCanvas || null;
    sharedPaintCanvas = paintCanvas || null;
}

/**
 * 设置共享 overlay；为避免持有失效引用，overlay 变化时清空 base/paint 缓存。
 */
function setSharedOverlay(overlay) {
    sharedOverlay = overlay || null;
    sharedBaseCanvas = null;
    sharedPaintCanvas = null;
}

function getSharedOverlay() {
    return sharedOverlay;
}

function getSharedBaseCanvas() {
    if (!sharedBaseCanvas) {
        const canvases = document.querySelectorAll('#maskEditorCanvasContainer canvas');
        sharedBaseCanvas = canvases[0] || null;
    }
    return sharedBaseCanvas;
}

function getSharedPaintCanvas() {
    if (!sharedPaintCanvas) {
        const canvases = document.querySelectorAll('#maskEditorCanvasContainer canvas');
        sharedPaintCanvas = canvases[1] || null;
    }
    return sharedPaintCanvas;
}

export {
    setSharedCanvases,
    setSharedOverlay,
    getSharedOverlay,
    getSharedBaseCanvas,
    getSharedPaintCanvas,
};
