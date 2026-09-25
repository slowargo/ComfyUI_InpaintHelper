"""Slowargo：ComfyUI 的 inpaint 辅助节点集。

本文件只做两件事：把各模块的节点类汇总进 NODE_CLASS_MAPPINGS，以及声明
WEB_DIRECTORY。节点实现按职责分在下面这几个模块里：

    nodes_color.py     Lab 色彩校正（MaskedColorMatch / InpaintRegionColorFix）
    nodes_image_io.py  图像读取、最近文件枚举，以及刷新预览的路由
    nodes_save.py      图像保存、服务端文件搬运，以及搬运路由
    nodes_strings.py   字符串记忆，以及它的历史记录路由
    nodes_preset.py    Widget Preset 的节点声明（功能全在前端）
    nodes_util.py      浮点开关/选择器、触发器、历史清理、SSIM 比较
    nodes_clip_cache.py  带磁盘缓存的 CLIPTextEncode
    nodes_sampling.py  采样循环内的 inpaint 偏色抑制（逐步修正 x0 预测）

注意：API 路由是在模块 import 时注册的副作用。下面每一行导入看起来「只用到
了几个节点类」，但删掉任何一行都会让该模块的路由一起消失。
"""

# 色彩校正节点拆到同名模块；映射表仍在本文件末尾统一维护
from .nodes_color import InpaintRegionColorFix, MaskedColorMatch
from .nodes_util import (
    ClearHistoryNode,
    FloatSelector,
    FloatSwitch,
    ImageSimilaritySSIM,
    RefreshTriggerV1,
    RunButtonNode,
)
from .nodes_save import ExtractSubFolder, SaveImageToFileName, ServerFileTransfer
from .nodes_image_io import (
    ImageWidgetSourcePath,
    LoadImageFromAnyPath,
    LoadImageFromOutputPlusV1,
    LoadRecentImagePlusV1,
)
from .nodes_strings import RememberStrings
from .nodes_preset import WidgetPreset
from .nodes_clip_cache import SimpleCachedCLIPTextEncode
from .nodes_sampling import InpaintX0DriftGuard


##############################################

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./js"


# V3 Extension declaration

# class SlowargoExtensions(ComfyExtension):
#     @override
#     async def get_node_list(self) -> list[type[io.ComfyNode]]:
#         return [
#             FloatSwitch,
#             LoadImageFromOutputsPlus,
#         ]
#
#
# async def comfy_entrypoint() -> ComfyExtension:  # ComfyUI calls this to load your extension and its nodes.
#     return SlowargoExtensions()

# V1 Extension declaration
NODE_CLASS_MAPPINGS = {
    "FloatSwitch": FloatSwitch,
    "FloatSelector": FloatSelector,
    "LoadImageFromOutputPlusV1": LoadImageFromOutputPlusV1,
    "LoadImageFromAnyPath": LoadImageFromAnyPath,
    "ImageWidgetSourcePath": ImageWidgetSourcePath,
    "LoadRecentImagePlusV1": LoadRecentImagePlusV1,
    "SaveImageToFileName": SaveImageToFileName,
    "ImageSimilaritySSIM": ImageSimilaritySSIM,
    "ExtractSubFolder": ExtractSubFolder,
    "RememberStrings": RememberStrings,
    "RunButtonNode": RunButtonNode,
    "ServerFileTransfer": ServerFileTransfer,
    "RefreshTriggerV1": RefreshTriggerV1,
    "ClearHistoryNode": ClearHistoryNode,
    "MaskedColorMatch": MaskedColorMatch,
    "InpaintRegionColorFix": InpaintRegionColorFix,
    "WidgetPreset": WidgetPreset,
    "SimpleCachedCLIPTextEncode": SimpleCachedCLIPTextEncode,
    "InpaintX0DriftGuard": InpaintX0DriftGuard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatSwitch": "Float Switch",
    "FloatSelector": "Float Selector",
    "LoadImageFromOutputPlusV1": "Load Image (from Outputs) Plus V1 (deprecated)",
    "LoadImageFromAnyPath": "Load Image (from Any Path)",
    "ImageWidgetSourcePath": "Image Widget Source Path",
    "LoadRecentImagePlusV1": "Load Recent Image",
    "SaveImageToFileName": "Save Image to Specified File Name",
    "ImageSimilaritySSIM": "Image Similarity (SSIM)",
    "ExtractSubFolder": "Extract Sub Folder",
    "RememberStrings": "Remember Recent Strings",
    "RunButtonNode": "Run Button",
    "ServerFileTransfer": "Server File Transfer",
    "RefreshTriggerV1": "Refresh Trigger",
    "ClearHistoryNode": "Clear History",
    "MaskedColorMatch": "Masked Color Match (Inpaint Drift Fix)",
    "InpaintRegionColorFix": "Inpaint Region Color Fix",
    "WidgetPreset": "Widget Preset",
    "SimpleCachedCLIPTextEncode": "CLIP Text Encode (Disk Cache)",
    "InpaintX0DriftGuard": "Inpaint X0 Drift Guard",
}
