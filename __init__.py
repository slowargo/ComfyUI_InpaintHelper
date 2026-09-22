import heapq
import json
import logging
import os
import shutil
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
from aiohttp import web
from comfy.cli_args import args
from comfy_api.latest import io
import folder_paths
import node_helpers
import nodes
import numpy as np
from server import PromptServer
import torch
import torch.nn.functional as F
import re
import hashlib

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

logger = logging.getLogger(__name__)

# 工具函数：处理图像并转换为 PyTorch 张量


# 单目录扫描结果缓存：key=(绝对目录, sub_folder, label, max_count) -> (目录 mtime_ns, 存入时刻, [(mtime, 显示名)])
# 失效判断用两道条件，任一不满足就重扫：
#   1) 目录自身的 mtime_ns 未变——新增/删除/改名会更新它；
#   2) 缓存年龄未超过 _RECENT_DIR_TTL。
# 光靠条件 1 不够：原地覆盖同名文件（SaveImageToFileName 就是这么干的）只会改
# 文件的 mtime，不会改目录的，而缓存里存的 mtime 同时决定了跨目录排序和
# top-N 的入选，漏判会让刚存的图一直不出现在列表里。TTL 把这种陈旧限死在 1 秒内。
# 单次 os.stat 很便宜，但重扫一个几百文件的目录要贵上三个数量级；真正要挡的是
# INPUT_TYPES 在同一次 prompt 校验里被连续调用好几次。


#######################################################################################################################
# V3 style nodes


#######################################################################################################################
# V1 style nodes


##############################################

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./js"

# Add custom API routes, using router


# 获取历史记录接口

# Toggle Pin 接口 (修改返回值为最新列表)

# 3. 删除记录接口

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
}
