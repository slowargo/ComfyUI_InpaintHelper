# Mask Editor 内存清理分析

## ✅ 已正确清理的资源

### 1. Canvas 对象
**位置**: `maskEditorStore.ts` - `resetState()`
```typescript
if (maskCanvas.value) {
  maskCanvas.value.width = 0  // ✅ 清空像素数据
  maskCanvas.value.height = 0
}
if (rgbCanvas.value) {
  rgbCanvas.value.width = 0   // ✅ 清空像素数据
  rgbCanvas.value.height = 0
}
if (imgCanvas.value) {
  imgCanvas.value.width = 0   // ✅ 清空像素数据
  imgCanvas.value.height = 0
}
```
**状态**: ✅ 正确清理

### 2. Image 对象
**位置**: `maskEditorStore.ts` - `resetState()`
```typescript
if (image.value) {
  image.value.src = ''  // ✅ 释放解码位图
  image.value = null
}
```

**位置**: `useImageLoader.ts` - `loadImages()`
```typescript
maskImage.src = ''           // ✅ 加载后立即释放
if (paintImage) paintImage.src = ''
```
**状态**: ✅ 正确清理

### 3. GPU Textures
**位置**: `useBrushDrawing.ts` - `onUnmounted()` 和 `destroy()`
```typescript
if (maskTexture) {
  maskTexture.destroy()      // ✅ 销毁纹理
  maskTexture = null
}
if (rgbTexture) {
  rgbTexture.destroy()       // ✅ 销毁纹理
  rgbTexture = null
}
```
**状态**: ✅ 正确清理

### 4. GPU Buffers
**位置**: `useBrushDrawing.ts` - `onUnmounted()` 和 `destroy()`
```typescript
readbackStorageMask?.destroy()   // ✅ 销毁缓冲
readbackStorageRgb?.destroy()
readbackStagingMask?.destroy()
readbackStagingRgb?.destroy()
```

**位置**: `GPUBrushRenderer.ts` - `destroy()`
```typescript
this.quadVertexBuffer.destroy()  // ✅ 销毁缓冲
this.indexBuffer.destroy()
this.instanceBuffer.destroy()
this.uniformBuffer.destroy()
if (this.currentStrokeTexture) this.currentStrokeTexture.destroy()
```
**状态**: ✅ 正确清理

### 5. GPU Renderer
**位置**: `useBrushDrawing.ts` - `onUnmounted()`
```typescript
if (renderer) {
  renderer.destroy()         // ✅ 销毁渲染器
  renderer = null
}
```
**状态**: ✅ 正确清理

### 6. GPU Preview Canvas
**位置**: `useBrushDrawing.ts` - `onUnmounted()`
```typescript
if (previewContext) {
  previewContext.unconfigure()  // ✅ 取消配置
  previewContext = null
}
if (previewCanvas) {
  previewCanvas.width = 0       // ✅ 清空像素数据
  previewCanvas.height = 0
  previewCanvas = null
}
```
**状态**: ✅ 正确清理

### 7. Canvas History
**位置**: `MaskEditorContent.vue` - `onBeforeUnmount()`
```typescript
store.canvasHistory.clearStates()  // ✅ 清理历史状态
```
**状态**: ✅ 正确清理

### 8. TypeGPU Root
**位置**: `useBrushDrawing.ts` - `destroy()`
```typescript
if (store.tgpuRoot) {
  store.tgpuRoot.destroy()    // ✅ 销毁 TGPU 根
  store.tgpuRoot = null
}
```
**状态**: ✅ 正确清理

### 9. ResizeObserver
**位置**: `MaskEditorContent.vue` - `onBeforeUnmount()`
```typescript
if (resizeObserver) {
  resizeObserver.disconnect()  // ✅ 断开观察器
  resizeObserver = null
}
```
**状态**: ✅ 正确清理

### 10. Keyboard Listeners
**位置**: `MaskEditorContent.vue` - `onBeforeUnmount()`
```typescript
keyboard?.removeListeners()    // ✅ 移除监听器
```
**状态**: ✅ 正确清理

## ❌ 未清理的资源（重新评估后）

经过详细分析，所有资源都已正确处理：

### 1. Brush Texture Cache ✅
**位置**: `useBrushDrawing.ts`
```typescript
const brushTextureCache = new QuickLRU<string, HTMLCanvasElement>({
  maxSize: 20
})
```

**分析**: 
- `brushTextureCache` 是 `useBrushDrawing` 的局部变量
- 当 `useToolManager` 实例销毁时，cache 引用也会断开
- QuickLRU 和其中的 canvas 都是普通 JS 对象，没有持有系统资源

**结论**: ✅ **不需要手动清理**
- 编辑器关闭后会自动被 GC 回收
- 不会造成内存泄漏

### 2. BaseImage 引用链 ✅
**位置**: 多个位置
```typescript
// useMaskEditorLoader.ts - 创建
const baseImage = new Image()

// useImageLoader.ts - 存储
store.image = baseImage

// maskEditorStore.ts - 清理
if (image.value) {
  image.value.src = ''  // ✅ 清空解码位图
  image.value = null
}
```

**清理路径**:
1. `useImageLoader.ts` 清理了 `maskImage.src` 和 `paintImage.src`
2. `baseImage` 存储在 `store.image`，在 `resetState()` 中清理
3. `dataStore.reset()` 设置 `inputData = null`

**清理顺序**（`MaskEditorContent.vue` - `onBeforeUnmount()`）:
```typescript
store.resetState()    // 先清理 store.image.src
dataStore.reset()     // 后清理 inputData
```

**结论**: ✅ **已正确清理**
- 清理顺序正确
- `baseImage.src` 会被清空，释放解码位图
- 所有引用断开后会被 GC 回收

### 3. OutputData 中的 Canvas 和 Blob ✅

**Canvas 清理**:
```typescript
// useMaskEditorSaver.ts - save() 函数中
function cleanupTemporaryCanvases(outputData: EditorOutputData): void {
  const canvases = [
    outputData.maskedImage.canvas,
    outputData.paintLayer.canvas,
    outputData.paintedImage.canvas,
    outputData.paintedMaskedImage.canvas
  ]
  for (const canvas of canvases) {
    if (canvas) {
      canvas.width = 0  // ✅ 已清理
      canvas.height = 0
    }
  }
}
```

**结论**: ✅ **Canvas 已正确清理**

**Blob 分析**:
```typescript
interface EditorOutputLayer {
  canvas: HTMLCanvasElement  // ✅ 已清理
  blob: Blob                 // Blob 是普通 JS 对象
  ref: ImageRef
}
```

**Blob 内存占用**:
- 对于 2048×2048 图像：8-40 MB（4 个 PNG blob）
- 在上传后不再需要，但保留到编辑器关闭
- Blob 是普通 JavaScript 对象，没有持有文件句柄等系统资源

**结论**: ✅ **Blob 不需要手动清理**
- `dataStore.reset()` 后引用断开，GC 会自动回收
- 不会造成内存泄漏

## 总结

### 清理完整性评分: 90/100

**优点**:
- 主要的大对象（GPU 资源、主 canvas、history、output canvas）都有正确清理
- 清理逻辑集中且有序
- 使用了 Vue 的生命周期钩子
- Output canvas 已在 cleanupTemporaryCanvases() 中主动清理

**需要改进**:
1. **低优先级**: 清理 brush texture cache（自动淘汰，优先级不高）
2. **低优先级**: 确保 image 清理顺序的健壮性

### 内存泄漏风险评估

- **GPU 资源**: ✅ 无风险
- **主 Canvas**: ✅ 无风险
- **History**: ✅ 无风险
- **Brush Cache**: ⚠️ 低风险（自动淘汰，但不主动清理）
- **Output Canvas**: ✅ 无风险（已在 cleanupTemporaryCanvases() 中清理）
- **Image 对象**: ✅ 无风险

### 建议的清理顺序

当前清理顺序（`MaskEditorContent.vue` - `onBeforeUnmount`）:
```
1. toolManager.brushDrawing.saveBrushSettings()
2. keyboard.removeListeners()
3. resizeObserver.disconnect()
4. store.canvasHistory.clearStates()
5. store.resetState()
6. dataStore.reset()
```

这个顺序是合理的，但应该在 `useBrushDrawing` 的清理中添加 cache 清理。
