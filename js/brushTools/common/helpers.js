import { getMaskEditorStore } from "../../utils.js";

/**
 * 统一 history 保存入口；调用方不需要重复判空。
 */
function saveCanvasHistory() {
    getMaskEditorStore()?.canvasHistory?.saveState?.();
}

/**
 * 通过触发 Vue checkbox 的 change 事件来保证 paint 层可见。
 * checkbox 顺序： [mask, paint, baseImage]
 */
function ensurePaintLayerVisible() {
    const checkboxes = document.querySelectorAll('.maskEditor_sidePanelLayerCheckbox');
    const paintCheckbox = checkboxes?.[1];
    if (!paintCheckbox) {
        console.warn("[slowargo.js] Paint layer checkbox not found");
        return;
    }

    if (!paintCheckbox.checked) {
        paintCheckbox.checked = true;
        paintCheckbox.dispatchEvent(new Event('change', { bubbles: true }));
    }
}

function getMaskEditorCanvasContainer() {
    return document.querySelector('#maskEditorCanvasContainer');
}

export {
    saveCanvasHistory,
    ensurePaintLayerVisible,
    getMaskEditorCanvasContainer,
};
