# ComfyUI InpaintHelper 扩展

## 简介

#### 一组用于提高 ComfyUI 重绘流程的节点。我在 inpaint 的时候会经常进行以下操作，如果你也有类似的需求，此扩展能帮到你：
- 频繁切换 denoise 强度, 较弱强度用于修复细节和细微错误（例如眼睛、嘴唇），较高强度用于根据提示词重新生成画面（例如肢体错误、手的形状错误）
- 

## 功能特性

### 节点类型

#### 浮点数切换节点
- **FloatSwitch**: 一个多功能的浮点数值选择器，允许根据布尔开关在两个浮点值之间切换。
  - 提供两个带滑块控制的输入浮点值
  - 切换开关以选择两个值中的一个
  - 可选的覆盖输入，当值大于0时具有优先权
  - 使用例子: 可设置 0.32 0.55 两个值，并连接到 ksampler 的 denoise 参数。0.32 对应轻度重绘(修复细微错误)，0.55 对应深度重绘(根据 prompt 改变内容)。点击开关在两者之间切换。一般建议深度重绘后再进行一次轻度重绘让画面更自然。

#### 图像加载节点
- **Load Image (from Outputs) Plus V1**: 增强版输出目录图像加载功能，带有最近文件预览
  - 类似官方的 Load Image from Outputs，但也会把 input 下的 clipspace 也加入到列表 (最多10个最新 output 文件和 3 个 clipspace 文件)。clipspace 文件会在编辑 mask 后自动产生，有时 comfyui 会丢失编辑结果，此设计可以方便找回上次编辑的结果。

- **Load Recent Image**: 从可配置的监控文件夹加载图像，具有灵活的过滤功能
  - 允许指定多个要监控的文件夹 (目前仅支持 input 和 output 目录)
  - 每个文件夹可配置显示最近文件的数量
  - 例如 `[10][output]; [5][input]; clipspace [6][input]` 表示刷新时检查 output 目录、 input 目录和 input/clipspace 目录，显示最近 10 个 output 目录下图片文件，input 目录下最近 5 个 input 目录下图片文件，input/clipspace 目录下最近 6 个图片文件
  - watch_folders 变更后需点击 refresh 按钮刷新列表
  - 当需要对画面不同位置进行重绘，可先 mask edit 一个位置，多次抽卡。只要不 refresh，mask 将持续有效。出现满意的结果后点 refresh 再 mask edit 下一个位置。
  - 此节点还导出了 rgthree action (Refresh action) 可结合 rgthree-comfy 的 Fast Actions 节点使用。

- **Load Image (from Any Path)**: 从任意路径加载图像
  - 从 ComfyUI 以外的路径加载图像
  - 可用于批量处理图片。例如使用 listdir 节点获取文件列表，传入此节点逐个处理。

#### 图像保存节点
- **Save Image to Specified File Name**: 带有自定义文件名的增强保存功能
  - 自定义输出文件名
  - 支持多种图像格式（PNG、JPEG、WEBP）根据文件名自动判断，或强行指定
  - 元数据保存 (输入来自 Load Image (from Any Path) 节点)
  - 子文件夹支持
  - 允许保存后在浏览器自动打开图片，方便仔细检视输出结果

#### 实用工具节点
- **Extract Sub Folder**: 从给定文件路径中提取子目录
  - 可配置提取级别（文件夹层数）
  - 有助于组织输出文件
- **Remember Recent Strings**: 带有记忆功能多行字符串输入框
  - 将输入字符串存储到 JSON 文件中，作为常用文本（例如 Prompt）的历史记录
  - 如果字符串已存在，则将其移至列表顶部；如果不存在，则添加
  - 支持“置顶”（pin）功能，置顶的条目会保持在列表前端
  - 可通过前端界面管理（查看、置顶、删除）
- **Run Button**: 运行按钮
  - 点击按钮会触发整个 ComfyUI 工作流的执行
  - 可以在 workflow 多个任意位置放置此节点，方便就近触发（当然也可以直接用系统快捷键触发）
  - 此节点还导出了 rgthree action (Run action) 可结合 rgthree-comfy 的 Fast Actions 节点使用。

#### 其他增强
- 快捷键
  - **Open Image**: 节点有 image widget 时，此快捷键会在浏览器打开图片。相当于右键菜单的 open image
  - **Switch to Mask**: 在 Mask Editor 中切换到 mask 工具
  - **Switch to Eye Dropper**: 在 Mask Editor 中切换到画笔工具并激活 eye dropper 拾色
  - **Save Mask**: 在 Mask Editor 中保存 mask 并退出
  - **Load Clipspace (Ctrl+L)**: 在 Mask Editor 中按 Ctrl+L (Mac 上为 Cmd+L) 可将最近的 clipspace 内容（底图、mask 和 paint 图层）加载到当前编辑画布。方便复用之前的 mask 编辑结果。
- 选中节点的 toolbox 按键
  - **Open Image**: 增加一个打开图片的按键，跟快捷键和右键菜单效果一样
- 其他完善
  - ComfyUI 在 Mask Editor 中用 Ctrl+Z 撤销改动时，可能会意外撤销外面的节点编辑或者文字编辑的内容。此扩展尝试对此问题进行修复。

## 安装方法

1. 将此存储库克隆或复制到您的 ComfyUI 自定义节点目录
2. 重启 ComfyUI
3. 节点将在节点菜单的 "Slowargo" 类别中出现

## 使用方法

这些节点在 ComfyUI 的节点菜单中归类于 "Slowargo" 类别下。只需根据需要将它们拖放到工作流中即可。

## 分类

此扩展中的节点在 ComfyUI 节点菜单中归类于 "Slowargo" 类别下。