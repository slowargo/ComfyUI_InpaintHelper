# loadClipspaceToEditor GPU Sync Fix

## Problem

`maskEditorTurbo.js` loads clipspace images to canvas via `drawImage`, but does not sync the GPU textures used by ComfyUI frontend's WebGPU rendering pipeline. This causes:

- GPU preview canvas shows stale/cleared data after reload
- Brush drawing operates on outdated GPU textures
- Loaded image state is not saved to undo history

## Root Cause Analysis

### Current flow (broken)

```
Click Clear button (DOM selector)
  → clearMask(): clearRect + saveState (saves cleared state to history)
  → triggerClear(): clearGPU (writes zeros to GPU textures)
  → Wait 200ms
  → drawImage to canvas (canvas updated, GPU still cleared)
  → No GPU sync, no history save
```

### GPU sync mechanism in ComfyUI frontend

The GPU sync is triggered by a Vue watch chain in `useBrushDrawing.ts:220-234`:

```typescript
watch(
    () => store.canvasHistory.currentStateIndex,
    async () => {
        if (isSavingHistory.value) return  // Only true during drawEnd()
        await updateGPUFromCanvas()        // Reads canvas → uploads to GPU
    }
)
```

`canvasHistory.saveState()` increments `currentStateIndex`, which triggers this watch. The `isSavingHistory` guard is only set to `true` inside `drawEnd()` (brush stroke end, `useBrushDrawing.ts:976-981`). External calls to `saveState()` bypass the guard, so `updateGPUFromCanvas()` runs.

### Access path to Pinia store

```
document.querySelector('#vue-app').__vue_app__     // Vue 3 app instance (main.ts:91)
  .config.globalProperties.$pinia                  // Pinia instance (main.ts:84)
  ._s.get('maskEditor')                            // Store by ID (maskEditorStore.ts:19)
  .canvasHistory.saveState()                        // Trigger GPU sync
```

### Full sync chain

```
store.canvasHistory.saveState()                    // useCanvasHistory.ts:58-104
  → getImageData from canvas, push to history
  → currentStateIndex++                            // useCanvasHistory.ts:93
  → watch fires                                    // useBrushDrawing.ts:220
  → isSavingHistory === false (not in drawEnd)     // useBrushDrawing.ts:224
  → updateGPUFromCanvas()                          // useBrushDrawing.ts:314-355
  → premultiplyData(maskImageData)                 // useBrushDrawing.ts:302-308
  → device.queue.writeTexture(maskTexture, ...)    // useBrushDrawing.ts:335-340
  → device.queue.writeTexture(rgbTexture, ...)     // useBrushDrawing.ts:349-354
```

## Changes

### 1. Add getMaskEditorStore helper

```javascript
function getMaskEditorStore() {
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
```

### 2. Replace Clear button click with clearRect

**Before:**
```javascript
const clearBtn = document.querySelector("#global-mask-editor > div.flex.items-center > div > button:nth-child(4)");
if (clearBtn) {
    clearBtn.click();
    await new Promise(resolve => setTimeout(resolve, 200));
}
```

**After:**
```javascript
maskCtx.clearRect(0, 0, canvases[2].width, canvases[2].height);
maskCtx.drawImage(maskImg, 0, 0, canvases[2].width, canvases[2].height);
```

Benefits:
- Removes fragile CSS selector `button:nth-child(4)`
- Removes 200ms hardcoded wait
- Does not add a "cleared" entry to undo history

### 3. Call saveState after drawing to trigger GPU sync

```javascript
const store = getMaskEditorStore();
if (store?.canvasHistory?.saveState) {
    store.canvasHistory.saveState();
}
```

This also saves the loaded image state to undo history, allowing the user to undo back to the pre-load state.

## Complete replacement for loadClipspaceToEditor

```javascript
// === Access Pinia Store for GPU Sync ===
function getMaskEditorStore() {
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

async function loadClipspaceToEditor(reloadMaskOnly = false) {
    try {
        const response = await api.fetchApi('/slowargo_api/refresh_previews_recent', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({watch_folders: 'clipspace [1][input]'})
        });

        const data = await response.json();
        if (!data.image_name?.[0]) {
            console.warn("[slowargo.js] No clipspace files found");
            return;
        }

        const timestamp = data.image_name[0].match(/clipspace-painted-masked-(\d+)\.png/)?.[1];
        if (!timestamp) {
            console.warn("[slowargo.js] Invalid clipspace filename");
            return;
        }

        const canvases = document.querySelectorAll("#maskEditorCanvasContainer canvas");
        if (canvases.length < 3) return;

        console.log("[slowargo.js] Loading clipspace, timestamp:", timestamp, "maskOnly:", reloadMaskOnly);

        const params = `${app.getPreviewFormatParam?.() || ""}${app.getRandParam?.() || ""}`;
        const loadImg = (url) => new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });

        // Load mask layer
        const maskImg = await loadImg(api.apiURL(
            `/view?filename=clipspace-mask-${timestamp}.png&subfolder=clipspace&type=input&channel=a${params}`
        ));
        const maskCtx = canvases[2].getContext('2d', {willReadFrequently: true});
        maskCtx.clearRect(0, 0, canvases[2].width, canvases[2].height);
        maskCtx.drawImage(maskImg, 0, 0, canvases[2].width, canvases[2].height);
        maskImg.src = '';

        // Invert alpha channel
        const maskData = maskCtx.getImageData(0, 0, canvases[2].width, canvases[2].height);
        for (let i = 3; i < maskData.data.length; i += 4) {
            maskData.data[i] = 255 - maskData.data[i];
        }
        maskCtx.putImageData(maskData, 0, 0);

        // Load base and paint layers
        if (!reloadMaskOnly) {
            const baseImg = await loadImg(api.apiURL(
                `/view?filename=clipspace-mask-${timestamp}.png&subfolder=clipspace&type=input&channel=rgb${params}`
            ));
            const baseCtx = canvases[0].getContext('2d', {willReadFrequently: true});
            baseCtx.clearRect(0, 0, canvases[0].width, canvases[0].height);
            baseCtx.drawImage(baseImg, 0, 0, canvases[0].width, canvases[0].height);
            baseImg.src = '';

            try {
                const paintImg = await loadImg(api.apiURL(
                    `/view?filename=clipspace-paint-${timestamp}.png&subfolder=clipspace&type=input${params}`
                ));
                const paintCtx = canvases[1].getContext('2d', {willReadFrequently: true});
                paintCtx.clearRect(0, 0, canvases[1].width, canvases[1].height);
                paintCtx.drawImage(paintImg, 0, 0, canvases[1].width, canvases[1].height);
                paintImg.src = '';
            } catch (e) {
                // Paint layer is optional
            }
        }

        // Sync GPU textures via Pinia store's canvasHistory
        const store = getMaskEditorStore();
        if (store?.canvasHistory?.saveState) {
            store.canvasHistory.saveState();
            console.log("[slowargo.js] GPU sync triggered via canvasHistory.saveState()");
        } else {
            console.warn("[slowargo.js] Pinia store not accessible, GPU textures may be stale");
        }

        console.log("[slowargo.js] Clipspace loaded successfully");
    } catch (error) {
        console.error("[slowargo.js] Failed to load clipspace:", error);
    }
}
```

## Why canvas size changes are not needed

The original analysis report suggested setting canvas dimensions before drawImage. This is unnecessary because:

1. Canvas sizes are set when the mask editor opens (`useCanvasManager.ts:54-59`), matching the original image
2. `clearRect()` does not change canvas dimensions
3. `drawImage(img, 0, 0, canvas.width, canvas.height)` scales the source image to fit the canvas — correct behavior for clipspace images derived from the same source
4. Changing canvas `width`/`height` clears all content and would break Vue ref bindings and GPU texture size assumptions

## Risks

| Risk | Assessment |
|------|------------|
| `pinia._s` is internal API | Stable across Pinia 2.x, used by Pinia devtools. May change in major version bumps |
| `__vue_app__` is Vue 3 internal | Stable across all Vue 3 versions, used by Vue devtools |
| `saveState()` called externally | No context dependency — accesses store refs via closure. Safe to call externally |
| Watch timing | Vue 3 watchers flush before next render. `writeTexture` synchronously queues GPU commands. GPU updated before next frame |

## Reference: key source locations

| File | Lines | Purpose |
|------|-------|---------|
| `useBrushDrawing.ts` | 77 | `isSavingHistory` ref definition |
| `useBrushDrawing.ts` | 212-217 | watch `clearTrigger` → `clearGPU()` |
| `useBrushDrawing.ts` | 220-234 | watch `currentStateIndex` → `updateGPUFromCanvas()` |
| `useBrushDrawing.ts` | 302-308 | `premultiplyData()` |
| `useBrushDrawing.ts` | 314-355 | `updateGPUFromCanvas()` |
| `useBrushDrawing.ts` | 976-981 | `isSavingHistory` set/unset in `drawEnd()` |
| `useBrushDrawing.ts` | 1447-1468 | `clearGPU()` |
| `useCanvasHistory.ts` | 58-104 | `saveState()` |
| `useCanvasHistory.ts` | 10 | `currentStateIndex` ref |
| `useCanvasTools.ts` | 440-453 | `clearMask()` |
| `maskEditorStore.ts` | 19 | Store ID `'maskEditor'` |
| `maskEditorStore.ts` | 175-177 | `triggerClear()` |
| `MaskEditorContent.vue` | 9-36 | Canvas layer z-ordering |
| `TopBarHeader.vue` | 97-100 | Clear button handler |
| `main.ts` | 44-91 | Vue app + Pinia creation and mount |
