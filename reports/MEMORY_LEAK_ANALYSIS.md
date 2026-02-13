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

## ❌ 未清理的资源

### 1. Brush Texture Cache
**位置**: `useBrushDrawing.ts`
```typescript
const brushTextureCache = new QuickLRU<string, HTMLCanvasElement>({
  maxSize: 20
})
```

**问题**:
- 缓存中存储了最多 20 个 canvas 元素
- 在 `onUnmounted()` 和 `destroy()` 中都没有清理
- 每个 canvas 可能占用几百 KB 到几 MB

**影响**:
- 如果多次打开/关闭 mask editor，缓存会累积
- QuickLRU 会自动淘汰旧条目，但不会主动清理 canvas 内存

**建议修复**:
```typescript
onUnmounted(() => {
  // ... 现有清理代码 ...
  
  // 清理笔刷纹理缓存
  brushTextureCache.clear()
})
```

### 2. DataStore 中的 Image 引用
**位置**: `maskEditorDataStore.ts` - `reset()`
```typescript
const reset = () => {
  inputData.value = null      // ⚠️ 只是设为 null
  outputData.value = null
  sourceNode.value = null
  // ...
}
```

**问题**:
- `inputData` 包含 `baseLayer.image`, `maskLayer.image`, `paintLayer?.image`
- 这些 Image 对象的 `src` 没有被清空
- 虽然引用被设为 null，但如果其他地方还持有引用，解码位图不会释放

**当前状态**:
- `useImageLoader.ts` 中已经清理了 `maskImage.src` 和 `paintImage.src`
- 但 `baseImage.src` 没有被清理（因为它被存储在 `store.image` 中）
- `store.image` 在 `resetState()` 中被清理了

**结论**: ⚠️ 基本正确，但依赖清理顺序

### 3. OutputData 中的 Canvas 和 Blob
**位置**: `maskEditorDataStore.ts`
```typescript
interface EditorOutputLayer {
  canvas: HTMLCanvasElement  // ⚠️ 未清理
  blob: Blob                 // ⚠️ 未清理
  ref: ImageRef
}
```

**问题**:
- `outputData` 包含 4 个 `EditorOutputLayer`
- 每个包含一个 canvas 和 blob
- `reset()` 只是设为 null，没有清空 canvas

**建议修复**:
```typescript
const reset = () => {
  // 清理 output canvas
  if (outputData.value) {
    const layers = [
      outputData.value.maskedImage,
      outputData.value.paintLayer,
      outputData.value.paintedImage,
      outputData.value.paintedMaskedImage
    ]
    layers.forEach(layer => {
      if (layer?.canvas) {
        layer.canvas.width = 0
        layer.canvas.height = 0
      }
    })
  }
  
  inputData.value = null
  outputData.value = null
  // ...
}
```

## 总结

### 清理完整性评分: 85/100

**优点**:
- 主要的大对象（GPU 资源、主 canvas、history）都有正确清理
- 清理逻辑集中且有序
- 使用了 Vue 的生命周期钩子

**需要改进**:
1. **高优先级**: 清理 brush texture cache
2. **中优先级**: 清理 outputData 中的 canvas
3. **低优先级**: 确保 image 清理顺序的健壮性

### 内存泄漏风险评估

- **GPU 资源**: ✅ 无风险
- **主 Canvas**: ✅ 无风险
- **History**: ✅ 无风险
- **Brush Cache**: ⚠️ 低风险（自动淘汰，但不主动清理）
- **Output Canvas**: ⚠️ 中风险（如果保存后未清理）
- **Image 对象**: ✅ 基本无风险

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
