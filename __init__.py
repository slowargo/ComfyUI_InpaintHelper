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


logger = logging.getLogger(__name__)

# 工具函数：处理图像并转换为 PyTorch 张量
def process_image_to_tensor(image_path):
    """
    通用的图像处理函数，将图像文件转换为PyTorch张量
    这个函数封装了原来在多个 load_image 方法中重复的图像处理逻辑
    """
    img = node_helpers.pillow(Image.open, image_path)

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']

    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]

        if image.size[0] != w or image.size[1] != h:
            continue

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        elif i.mode == 'P' and 'transparency' in i.info:
            mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    meta_data = ""
    if img.format == 'PNG':
        meta_data = img.info

    return output_image, output_mask, meta_data

_FINGERPRINT_CHUNK = 64 * 1024


def file_content_fingerprint(path) -> str:
    """给 IS_CHANGED 用的文件指纹：(size, mtime_ns, 首尾各 64KB 的 sha256)。

    不做整文件哈希：大图全读比只读首尾两块贵一到两个数量级。
    也不用纯 (mtime_ns, size)：NTFS 时间戳的实际更新粒度远粗于 100ns，
    同一路径背靠背写入两份不同内容会拿到完全相同的 (mtime_ns, size)，
    而 SaveImageToFileName 正是往固定文件名原地覆盖、再喂给本节点——
    这是本插件明确支持的工作流，纯 stat 指纹会让节点漏掉这次改动。
    注意 NOT_IDEMPOTENT 并不强制重新执行（comfy_execution/caching.py 只是把
    node_id 加进签名），IS_CHANGED 仍是签名里唯一对内容敏感的部分。

    文件不存在/不可读时照旧向上抛，交给 execution.py 处理。
    """
    st = os.stat(path)
    m = hashlib.sha256()
    m.update(f"{st.st_size}:{st.st_mtime_ns}:".encode())
    with open(path, 'rb') as f:
        if st.st_size <= _FINGERPRINT_CHUNK * 2:
            m.update(f.read())
        else:
            m.update(f.read(_FINGERPRINT_CHUNK))
            f.seek(-_FINGERPRINT_CHUNK, os.SEEK_END)
            m.update(f.read())
    return m.hexdigest()


# 单目录扫描结果缓存：key=(绝对目录, sub_folder, label, max_count) -> (目录 mtime_ns, 存入时刻, [(mtime, 显示名)])
# 失效判断用两道条件，任一不满足就重扫：
#   1) 目录自身的 mtime_ns 未变——新增/删除/改名会更新它；
#   2) 缓存年龄未超过 _RECENT_DIR_TTL。
# 光靠条件 1 不够：原地覆盖同名文件（SaveImageToFileName 就是这么干的）只会改
# 文件的 mtime，不会改目录的，而缓存里存的 mtime 同时决定了跨目录排序和
# top-N 的入选，漏判会让刚存的图一直不出现在列表里。TTL 把这种陈旧限死在 1 秒内。
# 单次 os.stat 很便宜，但重扫一个几百文件的目录要贵上三个数量级；真正要挡的是
# INPUT_TYPES 在同一次 prompt 校验里被连续调用好几次。
_RECENT_DIR_TTL = 1.0
_recent_dir_cache: dict = {}


def _list_recent_in_dir(
    full_dir: Path,
    sub_folder: str,
    label: str,
    max_count: int,
    valid_exts: set,
    use_cache: bool = True,
) -> List[Tuple[float, str]]:
    """扫描单个目录，返回按 mtime 降序的前 max_count 项 (mtime, 显示名称)。

    use_cache=False 只跳过“读”缓存，扫完照样把新结果写回去——否则手动刷新
    只能修好这一次的 HTTP 响应，下一次 INPUT_TYPES 又会读到那条没被覆盖的旧记录。
    """
    cache_key = (str(full_dir), sub_folder, label, max_count)
    # os.stat 必须严格早于下面的 os.scandir：并发下最坏只会把偏旧的 mtime 和
    # 偏新的内容存在一起，导致多扫一次，而不会把陈旧结果当成新的发出去。
    try:
        dir_mtime_ns = os.stat(full_dir).st_mtime_ns
    except OSError:
        dir_mtime_ns = None
    now = time.monotonic()
    if use_cache and dir_mtime_ns is not None:
        cached = _recent_dir_cache.get(cache_key)
        if cached is not None and cached[0] == dir_mtime_ns and now - cached[1] < _RECENT_DIR_TTL:
            return cached[2]

    # 热路径里不构造 Path 对象：原实现对每个文件都要建两个 Path（一次判后缀、一次存路径），
    # 几百个文件下这部分开销比 scandir 本身还大。DirEntry 已缓存 stat，
    # entry.stat() 在 Windows 上不会再发 syscall。
    entries: List[Tuple[float, str]] = []
    with os.scandir(full_dir) as it:
        for entry in it:
            name = entry.name
            if name.startswith('.'):
                continue
            dot = name.rfind('.')
            if dot < 0 or name[dot:].lower() not in valid_exts:
                continue
            if not entry.is_file():
                continue
            entries.append((entry.stat().st_mtime, name))

    # 只要前 max_count 个，用 nlargest 避免对整个目录做全排序。
    # 必须带 key：不带 key 会拿整个 (mtime, name) 元组比较，mtime 相同的文件
    # 变成按文件名倒序，连入选的那批都会变（解压/robocopy 拷进来的图常常时间戳全同）。
    # 带 key 的 nlargest 内部用递减序号打破平局，等价于原来 reverse=True 的稳定排序。
    result: List[Tuple[float, str]] = []
    for mtime, name in heapq.nlargest(max_count, entries, key=lambda t: t[0]):
        # 非递归 scandir，相对 full_dir 的路径就是文件名本身
        display_base = f"{sub_folder}/{name}" if sub_folder else name
        result.append((mtime, f"{display_base} [{label}]" if label else display_base))

    if dir_mtime_ns is not None:
        # watch_folders 是自由文本，用户每敲一个新的子目录/数量就多一个永久 key，
        # 而 max_count=9999 的那一路每个 key 能钉住上万条元组。条目数封顶后整体丢弃——
        # 缓存本来就只为挡住一秒内的重复调用，重建代价就是一次重扫。
        if len(_recent_dir_cache) >= 64:
            _recent_dir_cache.clear()
        _recent_dir_cache[cache_key] = (dir_mtime_ns, now, result)
    return result


def get_recent_image_files(
    directories: List[Tuple[str, str, int]],   # (sub_folder相对路径, label标记, 最大数量)
    base_root_getter=None,
    valid_exts: set = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"},
    use_cache: bool = True,
) -> List[str]:
    """
    从多个 (子目录 + 标记) 配置中，获取最新的图片文件显示名称列表
    所有文件按修改时间全局降序排序
    
    directories 中的每一项：
    - sub_folder: 相对于 base 的子路径（如 ""、"my/sub"）
    - label:      标记类型（如 "output"、"input" 或 ""）
                 - 如果 label 非空，会添加 [label] 到显示名
                 - 如果 label 为空，则不加标签
    """
    if base_root_getter is None:
        def base_root_getter(label: str) -> Path:
            if label == "input":
                return Path(folder_paths.get_input_directory())
            else:
                # 默认当作 output 处理
                return Path(folder_paths.get_output_directory())

    file_items: List[Tuple[float, str]] = []  # (mtime, 显示名称)

    for sub_folder, label, max_count in directories:
        # 确定实际根目录
        root_dir = base_root_getter(label)
        full_dir = root_dir / sub_folder if sub_folder else root_dir

        if not full_dir.is_dir():
            logger.warning(f"Invalid dir: {full_dir}")
            continue

        file_items.extend(
            _list_recent_in_dir(full_dir, sub_folder, label, max_count, valid_exts, use_cache)
        )

    # 全局按修改时间重新排序（复用上面已取到的 mtime）
    file_items.sort(key=lambda x: x[0], reverse=True)

    return [display_name for _, display_name in file_items]
#######################################################################################################################
# V3 style nodes

class FloatSwitchV3(io.ComfyNode):
    """
    A float switch node that outputs one of two float values based on a toggle switch.
    
    When the toggle is on, it outputs the first float value.
    When the toggle is off, it outputs the second float value.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        Define the schema for the float switch node.
        """
        return io.Schema(
            node_id="FloatSwitch",
            display_name="Float Switch",
            category="Slowargo",
            inputs=[
                io.Float.Input(
                    "float_a",
                    default=0.32,
                    min=0.1,
                    max=0.9,
                    step=0.02,
                    round=0.01,
                    display_mode=io.NumberDisplay.slider,
                    display_name="Float A (Toggle ON)"
                ),
                io.Float.Input(
                    "float_b",
                    default=0.55,
                    min=0.1,
                    max=0.9,
                    step=0.02,
                    round=0.01,
                    display_mode=io.NumberDisplay.slider,
                    display_name="Float B (Toggle OFF)"
                ),
                io.Boolean.Input(
                    "toggle",
                    default=False,
                    display_name="Toggle Switch"
                ),
                io.Float.Input(
                    "float_ovr",
                    default=0.0,
                    min=0.0,
                    max=0.9,
                    step=0.02,
                    round=0.01,
                    display_mode=io.NumberDisplay.number,
                    display_name="Float C (Effective If > 0)"
                ),
            ],
            outputs=[
                io.Float.Output(id="selected_float"),
            ],
            hidden=[
                io.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(cls, float_a, float_b, float_ovr, toggle) -> io.NodeOutput:
        """
        Execute the float selection logic.
        """
        if toggle == True:
            selected_value = float_a
        else:
            selected_value = float_b
        if float_ovr > 0:
            selected_value = float_ovr
        # logger.info(f"""[FloatSwitch]: selected_value:{selected_value} cls:{cls.GET_NODE_INFO_V3()} """)
        PromptServer.instance.send_sync("slowargo.js.extension.FloatSwitch", {"selected_value": selected_value})
        return io.NodeOutput(selected_value)

class LoadImageFromOutputsPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        file_names = cls.get_file_names()

        return io.Schema(
            node_id="LoadImageFromOutputsPlus",
            display_name="Load Image (from Outputs) Plus",
            category="Slowargo",
            description="Load an image from the output directory",
            inputs=[
                # io.Combo.Input(
                #     "image",
                #     upload=io.UploadType.image,
                #     image_folder=io.FolderType.output,
                #     remote=io.RemoteOptions(
                #         route= "/internal/files/output",
                #         refresh_button= True,
                #         # control_after_refresh= "first",
                #     )
                # )
                io.Combo.Input(
                    "image",
                    options=file_names,
                    # default=files[0],
                    upload=io.UploadType.image,
                ),
                io.String.Input(
                    "image_folder",
                    default="",
                    display_name="Subfolder in output folder"
                )
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.Mask.Output("MASK"),
                io.String.Output("file_name","File Name"),
                io.String.Output("meta_data","Meta Data"),
            ]
        )

    @staticmethod
    def get_file_names(sub_folder="", use_cache=True) -> List[str]:
        return get_recent_image_files([
            (sub_folder, "", 9999),  # 無標籤，數量幾乎不限
        ], use_cache=use_cache)

    @staticmethod
    def get_image_metadata(image_path):
        try:
            with Image.open(image_path) as img:
                if img.format == 'JPEG':
                    # exif_data_raw = img.info.get('exif')
                    # if exif_data_raw:
                    #     exif_dict = piexif.load(exif_data_raw)
                    #     exif_data = {}
                    #     for ifd_name in exif_dict:
                    #         if ifd_name != "thumbnail":  # Skip the thumbnail data
                    #             for tag, value in exif_dict[ifd_name].items():
                    #                 decoded_tag = TAGS.get(tag, tag)
                    #                 exif_data[decoded_tag] = value
                    #     return exif_data
                    return None
                elif img.format == 'PNG':
                    text_data = img.info
                    # logger.info(f"text_data: {text_data}")
                    return text_data
                    #return {key: text_data[key] for key in text_data if key not in ['exif', 'dpi']}
        except Exception as e:
            logger.error(f"Error: {e}")
            return None

    @classmethod
    def execute(cls, image: str, image_folder: str ) -> io.NodeOutput:
        try:
            def_dir = folder_paths.get_output_directory()
            def_dir = os.path.join(def_dir, image_folder)

            # image_folder as default folder
            image_path = folder_paths.get_annotated_filepath(image, def_dir)
            # logger.info(f"[LoadImageFromOutputsPlus] image:{image} image_folder:{image_folder} def_dir: {def_dir} -> {image_path}")

            output_image, output_mask, _ = process_image_to_tensor(image_path)
        
            return (output_image, output_mask)

        except Exception as e:
            logger.error(f"Error: {e}")
            return None

#######################################################################################################################
# V1 style nodes
class FloatSwitch:
    """
    浮点数切换器
    开关打开(toggle=True) → 输出 float_a
    开关关闭(toggle=False) → 输出 float_b
    当 float_ovr > 0 时，强制使用 float_ovr 的值（优先级最高）
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_a": ("FLOAT", {
                    "default": 0.32,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.02,
                    "round": 0.01,
                    "display": "slider",
                    # "tooltip": "Float A (Toggle ON)"
                }),
                "float_b": ("FLOAT", {
                    "default": 0.55,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.02,
                    "round": 0.01,
                    "display": "slider",
                    # "tooltip": "Float B (Toggle OFF)"
                }),
                "toggle": ("BOOLEAN", {
                    "default": False,
                    "label_on": "ON",
                    "label_off": "OFF",
                    "tooltip": "Outputs Float A when on, Float B when off. Overrides with float_ovr if > 0."
                }),
            },
            "optional": {
                "float_ovr": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.9,
                    "step": 0.02,
                    "round": 0.01,
                    "display": "number",
                    # "tooltip": "Float C - 如果 > 0 则强制使用此值"
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("selected_float",)

    FUNCTION = "do_switch"
    CATEGORY = "Slowargo"

    def do_switch(self, float_a, float_b, toggle, float_ovr=0.0, node_id=None):
        if toggle:
            selected = float_a
        else:
            selected = float_b

        # override 优先级最高
        if float_ovr > 0:
            selected = float_ovr

        # logger.info(f"[FloatSwitch] node_id:{node_id} selected:{selected}")
        # PromptServer.instance.send_sync("slowargo.js.extension.FloatSwitch", {"node_id": node_id, "selected_value": selected})

        return (selected,)


class FloatSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values_string": ("STRING", {
                    "default": "0.18;0.32;0.55",
                    "multiline": False,
                }),
                "selected_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                }),
                "min_value": ("FLOAT", {
                    "default": 0.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                }),
                "max_value": ("FLOAT", {
                    "default": 0.9,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                }),
                "step_value": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.001,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                }),
                "slot_count": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("selected_float",)

    FUNCTION = "select_float"
    CATEGORY = "Slowargo"

    @staticmethod
    def _normalize_values(values_string, slot_count):
        normalized_slot_count = max(int(slot_count or 0), 0)
        parts = str(values_string or "").split(";") if values_string is not None else []
        values = []

        for part in parts[:normalized_slot_count]:
            try:
                values.append(float(part.strip()))
            except (TypeError, ValueError):
                values.append(0.0)

        while len(values) < normalized_slot_count:
            values.append(0.0)

        return values

    @staticmethod
    def _get_step_precision(step_value):
        try:
            normalized = Decimal(str(step_value)).normalize()
        except (InvalidOperation, ValueError, TypeError):
            return 0

        exponent = normalized.as_tuple().exponent
        return max(0, -exponent)

    @staticmethod
    def _clamp(value, min_value, max_value):
        if value < min_value:
            return min_value
        if value > max_value:
            return max_value
        return value

    @classmethod
    def _quantize_value(cls, value, min_value, max_value, step_value):
        lower = float(min(min_value, max_value))
        upper = float(max(min_value, max_value))
        numeric_value = cls._clamp(float(value), lower, upper)

        numeric_step = float(step_value) if step_value is not None else 0.0
        if numeric_step > 0:
            snapped = round((numeric_value - lower) / numeric_step)
            numeric_value = lower + snapped * numeric_step
            numeric_value = cls._clamp(numeric_value, lower, upper)

        precision = cls._get_step_precision(numeric_step)
        if precision > 0:
            numeric_value = round(numeric_value, precision)

        return numeric_value

    def select_float(self, values_string, selected_index, slot_count, min_value=0.0, max_value=0.9, step_value=0.02):
        # Keep frontend-config widgets (`min`/`max`/`step`) in the signature
        # so ComfyUI can pass required inputs without runtime argument errors.
        values = self._normalize_values(values_string, slot_count)

        if not values:
            return (0.0,)

        clamped_index = min(max(int(selected_index or 0), 0), len(values) - 1)
        selected_value = values[clamped_index]
        quantized_value = self._quantize_value(selected_value, min_value, max_value, step_value)
        return (quantized_value,)

class LoadImageFromOutputPlusV1(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        # 不在这里扫目录：下面声明了 remote，列表由前端按需 GET
        # /slowargo_api/refresh_previews 拉取，内嵌 options 是重复劳动。
        # 官方 nodes.py 的 LoadImageOutput 就是这么写的——有 remote、不带 options。
        # 而 INPUT_TYPES 在每次 prompt 校验时都会被调用，这次扫描的结果压根没人读：
        # 继承自 LoadImage 的 VALIDATE_INPUTS(s, image) 让 execution.py:1033 跳过了
        # 整段 combo 成员检查，校验只用到输入的名字和类型，不碰 options。
        return {
            "required": {
                "image": ("COMBO", {
                    "image_upload": True,
                    "image_folder": "output",
                    "remote": {
                        "route": "/slowargo_api/refresh_previews",
                        "refresh_button": True,
                        "control_after_refresh": "first",
                    },
                }),
            },
            # "optional": {
            #     "sub_folder": ("STRING", {"default": ""}),
            # }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "File Name", "Meta Data")
    DESCRIPTION = "Load an image from the output folder. When the refresh button is clicked, the node will update the image list (takes 10 from output folder and 3 from clipspace folder) and automatically select the first image, allowing for easy iteration."
    EXPERIMENTAL = True
    FUNCTION = "load_image"
    CATEGORY = "Slowargo"

    def load_image(self, image, sub_folder=""):
        # 获取输出目录
        output_dir = Path(folder_paths.get_output_directory()) / sub_folder
        # 构造图像路径
        image_path = folder_paths.get_annotated_filepath(image, output_dir)

        # logger.info(f"[LoadImageFromOutputPlusV1] image:{image} sub_folder:{sub_folder} -> {image_path}")

        # 使用优化后的工具函数处理图像
        output_image, output_mask, meta_data = process_image_to_tensor(image_path)
        
        file_name = os.path.basename(image_path)

        return (output_image, output_mask, file_name, meta_data)

    @staticmethod
    def get_file_names(sub_folder="", use_cache=True) -> List[str]:
        return get_recent_image_files([
            (sub_folder, "", 10),       # output 目录 + sub_folder 前缀，不强制加 [output]
            ("clipspace","input", 3),   # clipspace 固定子目录，加 [input]
        ], use_cache=use_cache)

class LoadRecentImagePlusV1(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        default_watch_folders = "[10][output]; [5][input]; clipspace [6][input]"
        file_names = cls.get_file_names(default_watch_folders)
        return {
            "required": {
                "image": ("COMBO", {
                    "options":file_names,
                    "image_upload": True,
                    # "image_folder": "output",
                    # "remote": {
                    #     "route": "/slowargo_api/refresh_previews_recent",
                    #     "refresh_button": True,
                    #     "control_after_refresh": "first",
                    # },
                }),
            },
            "optional": {
                # 从指定目录获取最近文件。格式：sub folder + [最多结果数] + [目录类型]
                "watch_folders": ("STRING", {"default": default_watch_folders}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "File Name", "Meta Data")
    DESCRIPTION = "Load an image from the output folder. When the refresh button is clicked, the node will update the image list and automatically select the first image, allowing for easy iteration."
    EXPERIMENTAL = True
    FUNCTION = "load_image"
    CATEGORY = "Slowargo"
    NOT_IDEMPOTENT = True

    def load_image(self, image, watch_folders=""):
        # 构造图像路径
        image_path = folder_paths.get_annotated_filepath(image)

        # logger.info(f"[LoadImageFromOutputPlusV1] image:{image} -> {image_path}")

        # 使用优化后的工具函数处理图像
        output_image, output_mask, meta_data = process_image_to_tensor(image_path)
       
        file_name = os.path.basename(image_path)

        return (output_image, output_mask, file_name, meta_data)

    @staticmethod
    def get_file_names(watch_folders="", use_cache=True) -> List[str]:
        default_watch = "[5][input]"
        watch_folders = watch_folders.strip() or default_watch

        directories = []

        for item in watch_folders.split(";"):
            item = item.strip()
            if not item:
                continue

            match = re.match(r"^(.*?)\s*\[(\d+)\]\[(.*?)\]$", item)
            if not match:
                logger.warning(f"[LoadRecentImagePlusV1] Invalid watch folder item: {item}")
                continue

            sub_folder, count_str, folder_type = match.groups()
            sub_folder = sub_folder.strip()
            count = int(count_str)

            folder_type = folder_type.lower()

            directories.append((sub_folder, folder_type, count))

        return get_recent_image_files(directories, use_cache=use_cache)

    @classmethod
    def IS_CHANGED(s, image,watch_folders=""):
        image_path = folder_paths.get_annotated_filepath(image)
        return file_content_fingerprint(image_path)

class RefreshTriggerV1:
    """A remote refresh trigger for LoadRecentImagePlusV1.
    Connect the trigger input to any output of LoadRecentImagePlusV1,
    then use the refresh button to trigger a refresh with custom watch_folders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        default_watch_folders = "[10][output]; [5][input]; clipspace [6][input]"
        return {
            "required": {
                "trigger": ("*", {"tooltip": "Connect to any output of Load Recent Image"}),
            },
            "optional": {
                "watch_folders": ("STRING", {"default": default_watch_folders}),
            }
        }

    RETURN_TYPES = ()
    DESCRIPTION = "Remote refresh trigger for Load Recent Image node. Use the refresh button to trigger the connected Load Recent Image node's refresh with this node's watch_folders configuration."
    FUNCTION = "execute"
    CATEGORY = "Slowargo"
    OUTPUT_NODE = True

    def execute(self, trigger, watch_folders=""):
        return {}

    @staticmethod
    def IS_CHANGED(trigger, watch_folders=""):
        return watch_folders


def resolve_image_widget_source_path(prompt, node_id, input_name="trigger"):
    node_key = str(node_id)
    node_prompt = prompt.get(node_key)
    if node_prompt is None:
        raise ValueError(f"Node {node_key} was not found in prompt")

    trigger_input = node_prompt.get("inputs", {}).get(input_name)
    if not isinstance(trigger_input, (list, tuple)) or len(trigger_input) < 1:
        raise ValueError(f"Connect {input_name} to a node output")

    origin_key = str(trigger_input[0])
    origin_prompt = prompt.get(origin_key)
    if origin_prompt is None:
        raise ValueError(f"Connected source node {origin_key} was not found in prompt")

    image_value = origin_prompt.get("inputs", {}).get("image")
    if isinstance(image_value, (list, tuple)):
        raise ValueError("Connected source node image input is linked, not a widget value")
    if image_value is None or not str(image_value).strip():
        raise ValueError("Connected source node has no image widget value")

    image_text = str(image_value).strip()
    image_path = Path(image_text).expanduser()
    if image_path.is_absolute():
        return str(image_path)

    output_dir = Path(folder_paths.get_output_directory())
    return str(Path(folder_paths.get_annotated_filepath(image_text, output_dir)))


class ImageWidgetSourcePath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*", {"tooltip": "Connect to any output of a node that has an image widget."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("source_path",)
    DESCRIPTION = "Get the server-side source path from the connected source node's image widget."
    FUNCTION = "get_source_path"
    CATEGORY = "Slowargo"

    def get_source_path(self, trigger, prompt=None, node_id=None):
        return (resolve_image_widget_source_path(prompt or {}, node_id),)

    @classmethod
    def IS_CHANGED(cls, trigger, prompt=None, node_id=None):
        try:
            return resolve_image_widget_source_path(prompt or {}, node_id)
        except Exception:
            return ""

class LoadImageFromAnyPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "File Name", "Meta Data")
    DESCRIPTION = "Load an image from any path."
    FUNCTION = "load_image"
    CATEGORY = "Slowargo"
    NOT_IDEMPOTENT = True

    def load_image(self, image_path):
        # logger.info(f"[LoadImageFromAnyPath] image:{image_path}")

        output_image, output_mask, meta_data = process_image_to_tensor(image_path)

        return (output_image, output_mask, image_path, meta_data)

    @classmethod
    def IS_CHANGED(s, image_path):
        if image_path is None or not os.path.exists(image_path):
            # logger.info(f"[LoadImageFromAnyPath] IS_CHANGED image_path:{image_path}")
            return ""
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

class SaveImageToFileName(nodes.SaveImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"default": "ComfyUI_output", "tooltip": "Output file name. If no suffix, default to .png. Parent paths will be ignored."}),
                "sub_folder": ("STRING", {"default": ""}),
                "meta_data": ("STRING", {"default": ""}),
                "force_format": ("COMBO", {"options": ["PNG", "JPEG", "WEBP","auto"], "default": "auto"}),
                "auto_open": ("BOOLEAN", {"default": False, "label_on": "Open After Save", "label_off": "Don't Open After Save"})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    FUNCTION = "save_image"
    CATEGORY = "Slowargo"

    def save_image(self, image, filename="ComfyUI_output", sub_folder="", meta_data=None, force_format="auto", auto_open=False, prompt=None, extra_pnginfo=None):
        """
        Save / Overwrite image with filename.
        保存圖像，支持：
        - 根據 filename 後綴自動選擇格式（.png / .jpg / .jpeg / .webp 等）
        - 優先使用傳入的 meta_data（字符串 JSON 格式，包含原始 PNG 元數據）
        - 其次使用 ComfyUI 標準的 prompt / extra_pnginfo
        - PNG 格式完整保留元數據（文本 + ICC Profile）
        - 其他格式（如 JPG/WEBP）不寫元數據（因為不支援或不推薦）
        """
        results = []
        img = None
        # 只處理第一張圖像
        for (batch_number, image) in enumerate(image):
            # 轉換為 Pillow Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            break

        # 準備保存目錄  
        full_output_folder = Path(folder_paths.get_output_directory()) / sub_folder
        full_output_folder.mkdir(parents=True, exist_ok=True)

        filename = os.path.basename(filename)
        full_file_path = full_output_folder / filename

        # 解析文件名與格式
        file_path = Path(filename)
        suffix = file_path.suffix.lower()  # 如 ".png", ".jpg"
        stem = file_path.stem              # 主文件名

        file_format = "PNG"
        if force_format != "auto":
            # 強制指定格式, 但保存時還是使用原文件名
            file_format = force_format.upper()
        elif suffix:
            # Auto detect file type
            if suffix in (".png", ".jpg", ".jpeg", ".webp"):
                file_format = suffix[1:].upper()  # 如 "PNG", "JPEG"
            else:
                logger.warning(f"[SaveImageToFileName] unknown suffix: {suffix}, default to PNG")

        if not suffix:
            suffix = f".{file_format.lower()}"
            full_file_path = full_output_folder / f"{stem}{suffix}"

        # 默認 PNG（如果無後綴）
        # if not suffix:
        #     if force_format != "auto":
        #         suffix = f".{force_format.lower()}"
        #     else:
        #         # default to PNG
        #         suffix = ".png"
        #     file_name = f"{stem}{suffix}"
        # else:
        #     file_name = filename

        # 準備保存參數
        save_kwargs = {
            "compress_level": 5,
            # "optimize": True
        }

        # if force_format != "auto":
        #     save_kwargs["format"] = force_format.upper()

        pnginfo = None
        icc_profile = None

        # 優先級 1：用戶傳入的 meta_data（字符串 JSON）
        if meta_data:
            try:
                if file_format == "PNG":
                    pnginfo = PngInfo()

                    if isinstance(meta_data, str):
                        meta_data = json.loads(meta_data)
                    metadata = meta_data

                    # 恢復文本元數據
                    # for key, value in metadata.get("text", {}).items():
                    for key, value in metadata.items():
                        # logger.info(f"[SaveImageToFileName] meta_data key:{key} ")
                        if isinstance(value, str):
                            pnginfo.add_text(key, value)
                        else:
                            enc = json.dumps(value, ensure_ascii=True)
                            # logger.info(f"[SaveImageToFileName] meta_data value:{enc} ")
                            pnginfo.add_text(key, str(enc))

                    # 恢復 ICC Profile
                    if metadata.get("icc_profile_base64"):
                        import base64
                        icc_bytes = base64.b64decode(metadata["icc_profile_base64"])
                        icc_profile = icc_bytes
            except Exception as e:
                logger.warning(f"[SaveImageToFileName] failed to parse custom meta_data : {e}")

        # 優先級 2：ComfyUI 標準元數據（僅在 PNG 且未被覆蓋時添加）
        if not args.disable_metadata and file_format == "PNG" and pnginfo is None:
            pnginfo = PngInfo()

            # 添加 prompt
            if prompt is not None:
                pnginfo.add_text("prompt", json.dumps(prompt))

            # 添加 extra_pnginfo（如 workflow）
            if extra_pnginfo is not None:
                for key in extra_pnginfo:
                    pnginfo.add_text(key, json.dumps(extra_pnginfo[key]))

        # 只有 PNG 才傳 pnginfo 和 icc_profile
        if file_format == "PNG":
            if pnginfo:
                save_kwargs["pnginfo"] = pnginfo
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            save_kwargs["format"] = "PNG"
        elif file_format == "JPEG":
            save_kwargs["format"] = "JPEG"
            save_kwargs["quality"] = 95      # 高品質 JPEG
            save_kwargs["optimize"] = True
        elif file_format == "WEBP":
            save_kwargs["format"] = "WEBP"
            save_kwargs["quality"] = 95
            save_kwargs["method"] = 6         # 最高壓縮質量
        # should not reach here!
        # else:
        #     # 不支援的格式，強制轉 PNG
        #     logger.warning(f"[SaveImageToFileName] Unknown format {file_format}，強制保存為 PNG")
        #     save_kwargs["format"] = "PNG"
        #     if pnginfo:
        #         save_kwargs["pnginfo"] = pnginfo
        #     if icc_profile:
        #         save_kwargs["icc_profile"] = icc_profile

        # 執行保存
        img.save(full_file_path, **save_kwargs)

        # 返回 ComfyUI 標準格式
        results.append({
            "filename": full_file_path.name,
            "subfolder": sub_folder,
            "type": "output"  # 或 self.type，根據你的節點類型調整
        })

        # logger.info(f"[SaveImageToFileName] full_output_folder:{full_output_folder} full_file_path:{full_file_path} args:{save_kwargs}")
        if auto_open:
            PromptServer.instance.send_sync("slowargo.js.extension.SaveImageToFileName", {"results": results})

        return {"ui": {"images": results}}

class ExtractSubFolder:
    """
    从传入的路径解析出子目录
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "tooltip": "Input path, e.g., /sub_folder/image.png"
                }),
                "max_level": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Maximum extraction level"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sub_folder",)

    FUNCTION = "extract_sub_folder"
    CATEGORY = "Slowargo"

    def extract_sub_folder(self, path, max_level=1):
        """
        从文件路径中提取最后 max_level 层子目录（以 / 结尾）
        示例：
        - /foo/bar/baz/image.png, max_level=1 → 'baz/'
        - 同上, max_level=2 → 'bar/baz/'
        - 同上, max_level=3 → 'foo/bar/baz/'
        """
        # 转换为 Path 对象，更方便处理
        p = Path(path)
        
        # 获取所有父目录部分（不含文件名）
        parts = list(p.parent.parts)  # 例如 ['', 'foo', 'bar', 'baz']
        # logger.info(f"[ExtractSubFolder] parts:{parts}")
        
        # 如果层级不够，返回能取到的全部（或根据需求返回空）
        if len(parts) <= 1:  # 只有根目录或空
            return ''
        
        # 从后面取 max_level 层（去掉空字符串的部分）
        start_idx = max(1, len(parts) - max_level)  # 至少保留一层
        selected_parts = parts[start_idx:]
        
        # 拼接回路径，并保证以 / 结尾
        sub_folder = '/'.join(selected_parts)
        # if sub_folder:
        #     sub_folder += '/'
        # logger.info(f"[ExtractSubFolder] sub_folder: {sub_folder}")
        
        return (sub_folder,)

class RememberStrings:
    """
    Store input string to json. If the string has been remembered, move it to the top.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Input string"
                }),
                "store_file": ("STRING", {
                    "default": "remember_strings.json[output]",
                    "tooltip": "Store the string in this file in json format."
                }),
                "max_entries": ("INT", {
                    "default": 10,
                    "tooltip": "Maximum number of entries to store."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "remember_strings"
    CATEGORY = "Slowargo"

    def remember_strings(self, string, store_file, max_entries=10):
        # trim string, trim store_file，解析store_file格式确定文件路径，文件如果存在，读入已记忆的string列表。
        # 判断是否已记忆过该string，如果已记忆则将其移动到列表的顶部，否则添加到列表顶部。记录条数不超过 max_entries
        # 之后保存回文件（确保string内容不会破坏json格式）。
        
        # Trim the inputs
        string = string.strip()

        if not string: return ("",)

        file_path, stored_entries = RememberStrings.read_stored_strings(store_file)

        # 1. 查找当前字符串是否已存在
        existing_entry = next((item for item in stored_entries if item["content"] == string), None)

        if existing_entry:
            # 如果存在，先移除旧的（为了重新排序到顶端）
            is_pinned = existing_entry.get("pinned", False)
            stored_entries.remove(existing_entry)
        else:
            is_pinned = False

        # 2. 插入到最前面
        new_entry = {"content": string, "pinned": is_pinned}
        stored_entries.insert(0, new_entry)

        # 3. 淘汰逻辑
        # 我们需要保留所有 pinned=True 的，以及排在前面的非 pinned 条目，总数不超过 max_entries
        pinned_items = [item for item in stored_entries if item.get("pinned")]
        unpinned_items = [item for item in stored_entries if not item.get("pinned")]

        # 计算还能容纳多少个非置顶条目
        # 即使 pinned 很多，我们也至少保证总数逻辑或优先保证 pinned
        allowed_unpinned_count = max(0, max_entries - len(pinned_items))
        final_entries = pinned_items + unpinned_items[:allowed_unpinned_count]

        # 如果你希望最新的操作始终排在最前（无论是否 pin），可以用下面的简单逻辑：
        # 但通常逻辑是：Pinned 永远在顶端，新的在 Pinned 下方，或者干脆只按时间排，淘汰时跳过 Pinned。

        # 重新排序：Pinned 在上，其余按新旧排
        final_entries = sorted(final_entries, key=lambda x: x.get("pinned", False), reverse=True)

        # 保存文件
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(final_entries, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"[RememberStrings] Save error: {e}")

        # PromptServer.instance.send_sync("slowargo.js.extension.RememberStrings", {"entries": final_entries})
        return (string,)

    @staticmethod
    def read_stored_strings(store_file):
        store_file = store_file.strip()
        # 使用正则表达式解析 store_file 格式，支持类似 "filename.json[output]" 的格式
        # 忽略文件名和 [ 之间的空格
        match = re.match(r"^(.+?)\s*\[([^\]]+)\]$", store_file)
        if match:
            file_name = match.group(1).strip()  # 提取文件名部分并去除空格
            folder_type = match.group(2).lower()  # 提取 [] 中的内容并转为小写

            base_dir = folder_paths.get_input_directory() if folder_type == "input" else folder_paths.get_output_directory()

            # logger.info(f"[RememberStrings] base_dir:{base_dir} file_name:{file_name}")

            file_path = os.path.join(base_dir, file_name)
        else:
            # 如果没有格式，默认放 output
            file_path = os.path.join(folder_paths.get_output_directory(), store_file)

        # 读取已存在的列表，如果文件不存在则创建空列表
        stored_entries = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # 兼容老格式：如果读取到的是纯字符串，自动转为对象
                        stored_entries = [
                            item if isinstance(item, dict) else {"content": item, "pinned": False}
                            for item in data
                        ]
            except Exception as e:
                logger.warning(f"[RememberStrings] Read error: {e}")

        # logger.info(f"[RememberStrings] file_path:{file_path}")
        return file_path, stored_entries

class ImageSimilaritySSIM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN")
    RETURN_NAMES = ("similarity", "is_similar")
    FUNCTION = "compute_similarity"
    CATEGORY = "Slowargo"

    @staticmethod
    def _to_grayscale_bchw(image: torch.Tensor) -> torch.Tensor:
        # IMAGE input is typically [B, H, W, C] with range [0, 1]
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor with 4 dims [B,H,W,C], got shape: {tuple(image.shape)}")

        image = image.float()
        image = image.permute(0, 3, 1, 2)
        channels = image.shape[1]

        if channels == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
            return (image * weights).sum(dim=1, keepdim=True)
        if channels == 1:
            return image
        return image.mean(dim=1, keepdim=True)

    @staticmethod
    def _ssim_batch(gray_a: torch.Tensor, gray_b: torch.Tensor) -> torch.Tensor:
        # Global SSIM per image in batch.
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        eps = 1e-8

        mu_a = gray_a.mean(dim=(-2, -1), keepdim=True)
        mu_b = gray_b.mean(dim=(-2, -1), keepdim=True)

        var_a = ((gray_a - mu_a) ** 2).mean(dim=(-2, -1), keepdim=True)
        var_b = ((gray_b - mu_b) ** 2).mean(dim=(-2, -1), keepdim=True)
        cov_ab = ((gray_a - mu_a) * (gray_b - mu_b)).mean(dim=(-2, -1), keepdim=True)

        numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
        denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2) + eps
        ssim = numerator / denominator
        return ssim.squeeze(-1).squeeze(-1).squeeze(-1).clamp(0.0, 1.0)

    def compute_similarity(self, image_a, image_b, threshold=0.9):
        if image_a.shape[0] != image_b.shape[0]:
            raise ValueError(
                f"Batch size mismatch: image_a batch={image_a.shape[0]}, image_b batch={image_b.shape[0]}"
            )

        gray_a = self._to_grayscale_bchw(image_a)
        gray_b = self._to_grayscale_bchw(image_b)

        target_h = min(gray_a.shape[-2], gray_b.shape[-2])
        target_w = min(gray_a.shape[-1], gray_b.shape[-1])

        if gray_a.shape[-2:] != (target_h, target_w):
            gray_a = F.interpolate(gray_a, size=(target_h, target_w), mode="bilinear", align_corners=False)
        if gray_b.shape[-2:] != (target_h, target_w):
            gray_b = F.interpolate(gray_b, size=(target_h, target_w), mode="bilinear", align_corners=False)

        ssim_values = self._ssim_batch(gray_a, gray_b)
        similarity = float(ssim_values.mean().item())
        is_similar = similarity >= float(threshold)

        return (similarity, is_similar)

def _resize_bchw_to(tensor_bchw: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """把 [B,C,H,W] 缩放到目标尺寸；尺寸已一致时原样返回，不产生拷贝。"""
    if tensor_bchw.shape[-2:] == size:
        return tensor_bchw
    return F.interpolate(tensor_bchw, size=size, mode="bilinear", align_corners=False)


def _broadcast_batch(tensor: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    """batch 为 1 时零拷贝展开到目标 batch；既不为 1 也不匹配则报错。"""
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(f"Batch size mismatch: image batch={batch}, {name} batch={tensor.shape[0]}")


def _normalize_mask(mask: torch.Tensor, height: int, width: int, device) -> torch.Tensor:
    """把常见形状的 MASK 归一成 [B,H,W]、值域 [0,1]、对齐到目标尺寸。

    不做 batch 广播：调用方通常要先做形态学再广播，否则会把展开后的副本实体化。
    """
    msk = mask.float().to(device)
    if msk.ndim == 2:
        msk = msk.unsqueeze(0)
    elif msk.ndim == 4:
        # [B,1,H,W] 或 [B,H,W,1] 都归一到 [B,H,W]
        if msk.shape[1] == 1:
            msk = msk.squeeze(1)
        elif msk.shape[-1] == 1:
            msk = msk.squeeze(-1)
    if msk.ndim != 3:
        raise ValueError(f"Expected MASK tensor with 3 dims [B,H,W], got shape: {tuple(mask.shape)}")
    if msk.shape[1:3] != (height, width):
        msk = _resize_bchw_to(msk.unsqueeze(1), (height, width)).squeeze(1)
    return msk.clamp(0.0, 1.0)


class MaskedColorMatch:
    """基于遮罩外区域的线性色彩回归，校正 inpaint 的 VAE 重建偏差。

    VAE encode/decode 往返带有系统性偏差，与内容无关、全图一致，迭代式修补下会逐轮累积。

    本节点只用遮罩外（内容理论上未被改动）的像素拟合逐通道的 reference = k * image + c，
    再把该变换施加到整张图上。

    作用范围仅限上述 VAE 分量。重绘区内采样器自身还会叠加一层偏移，那部分在遮罩外没有
    任何可观测样本，本节点无法、也不试图消除它——强行把重绘区均值拉回原图均值会破坏
    "重绘本来就该改变颜色"的正常情况。见 InpaintRegionColorFix。
    """

    # Guard rails for the per-channel least-squares fit.
    # 样本区纹理太弱时 gain 会被 regression dilution 系统性压低（x 自带 VAE 重建噪声，
    # k → var_true/(var_true+var_noise)），把这种被低估的 k 施加到全图等于压对比度，
    # 危害远大于它要修的那点偏移。std 低于阈值就降级为纯 offset。
    MIN_SAMPLE_STD = 0.12          # ≈ 30/255
    MIN_SAMPLE_RATIO = 0.002       # 样本数下限取 max(MIN_SAMPLES, 像素总数 * 该比例)
    MIN_SAMPLES = 1024
    GAIN_LIMITS = (0.9, 1.1)
    EPS = 1e-12

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The VAE-decoded inpaint result to correct."}),
                "reference": ("IMAGE", {"tooltip": "The original image before VAE encode, same framing as image."}),
                "mode": (["offset_only", "gain_offset"], {
                    "default": "offset_only",
                    "tooltip": "offset_only fits a per-channel constant shift (recommended, matches the measured VAE bias). gain_offset also fits a slope, which needs a well-textured sample region to be reliable."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blends the correction: 0 leaves the image untouched, 1 applies the full fit."
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Only pixels whose mask value is at or below this are used to fit the correction. Keep it low so that repainted pixels never enter the fit."
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "The inpaint mask. Without it the fit uses the whole image, which lets repainted content distort the result."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "match_color"
    CATEGORY = "Slowargo"
    DESCRIPTION = "Remove the VAE roundtrip color drift from an inpaint result by fitting a per-channel linear correction on the pixels outside the mask, then applying it to the whole image."

    def _sample_region(self, mask, batch, height, width, mask_threshold, device):
        """遮罩外像素的选取。返回 [B,H,W] 的 bool。"""
        if mask is None:
            logger.warning(
                "[MaskedColorMatch] no mask connected, fitting on the whole image - "
                "repainted content will distort the correction"
            )
            return torch.ones((batch, height, width), dtype=torch.bool, device=device)

        msk = _normalize_mask(mask, height, width, device)
        # mask 缩小时 bilinear 是点采样，细羽化带会被跳过而让重绘像素混进样本池，
        # 用 3x3 max 保守膨胀挡掉这种漏采。注意它挡不住 VAE 解码感受野造成的
        # 边界渗透（尺度是 8px 量级），那部分只占样本池极小比例，对均值影响可忽略。
        msk = F.max_pool2d(msk.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        return _broadcast_batch(msk, batch, "mask") <= mask_threshold

    def match_color(self, image, reference, mode="offset_only", strength=1.0, mask_threshold=0.05, mask=None):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor with 4 dims [B,H,W,C], got shape: {tuple(image.shape)}")

        img = image.float()
        batch, height, width, channels = img.shape

        ref = reference.float().to(img.device)
        if ref.ndim != 4:
            raise ValueError(f"Expected reference IMAGE with 4 dims [B,H,W,C], got shape: {tuple(ref.shape)}")
        if ref.shape[-1] != channels:
            raise ValueError(f"Channel mismatch: image has {channels}, reference has {ref.shape[-1]}")
        if ref.shape[1:3] != (height, width):
            logger.warning(
                f"[MaskedColorMatch] reference size {tuple(ref.shape[1:3])} != image size {(height, width)}, "
                f"resizing reference - the interpolation blur it adds will bias the fit"
            )
            ref = _resize_bchw_to(ref.permute(0, 3, 1, 2), (height, width)).permute(0, 2, 3, 1)
        ref = _broadcast_batch(ref, batch, "reference")

        sample_region = self._sample_region(mask, batch, height, width, mask_threshold, img.device)
        min_samples = max(self.MIN_SAMPLES, int(height * width * self.MIN_SAMPLE_RATIO))
        gain_lo, gain_hi = self.GAIN_LIMITS

        corrected = torch.empty_like(img)
        report_lines = [f"mode={mode} strength={strength:.2f} mask_threshold={mask_threshold:.2f}"]

        for b in range(batch):
            weight = sample_region[b].to(img.dtype)
            sample_count = int(weight.sum().item())
            if sample_count < min_samples:
                logger.warning(
                    f"[MaskedColorMatch] batch {b}: only {sample_count} usable pixels "
                    f"(need {min_samples}, mask_threshold={mask_threshold}), skipping correction"
                )
                corrected[b] = img[b]
                report_lines.append(f"[{b}] SKIPPED n={sample_count} < {min_samples}")
                continue

            # 逐通道加权统计，一次算完所有通道，避免 per-channel 的布尔 gather
            wc = weight.unsqueeze(-1)
            x, y = img[b], ref[b]
            x_mean = (x * wc).sum(dim=(0, 1)) / sample_count
            y_mean = (y * wc).sum(dim=(0, 1)) / sample_count
            x_dev, y_dev = x - x_mean, y - y_mean
            x_var = ((x_dev * x_dev) * wc).sum(dim=(0, 1)) / sample_count
            y_var = ((y_dev * y_dev) * wc).sum(dim=(0, 1)) / sample_count
            covariance = ((x_dev * y_dev) * wc).sum(dim=(0, 1)) / sample_count

            # 上游任何一个坏像素都会让均值/方差变成 NaN，再逐像素相加就是整图报废
            if not bool(torch.isfinite(torch.stack([x_mean, y_mean, x_var, y_var, covariance])).all()):
                logger.warning(
                    f"[MaskedColorMatch] batch {b}: non-finite statistics in the sample region "
                    f"(NaN/Inf in image or reference), skipping correction"
                )
                corrected[b] = img[b]
                report_lines.append(f"[{b}] SKIPPED non-finite statistics")
                continue

            x_std = x_var.sqrt()
            unit_gain = torch.ones_like(x_std)
            if mode == "gain_offset":
                fit_ok = x_std >= self.MIN_SAMPLE_STD
                raw_gain = covariance / x_var.clamp_min(self.EPS)
                bounded = raw_gain.clamp(gain_lo, gain_hi)
                gain = torch.where(fit_ok, bounded, unit_gain)
                was_clamped = fit_ok & (bounded != raw_gain)
                if not bool(fit_ok.all()):
                    logger.warning(
                        f"[MaskedColorMatch] batch {b}: sample region too flat "
                        f"(std={[round(v * 255.0, 2) for v in x_std.tolist()]}/255 < "
                        f"{self.MIN_SAMPLE_STD * 255.0:.0f}/255), falling back to offset_only on those channels"
                    )
                if bool(was_clamped.any()):
                    logger.warning(
                        f"[MaskedColorMatch] batch {b}: gain clamped to {self.GAIN_LIMITS} "
                        f"(raw={[round(v, 4) for v in raw_gain.tolist()]}), the fit is not trustworthy"
                    )
            else:
                gain = unit_gain
                was_clamped = torch.zeros_like(unit_gain, dtype=torch.bool)

            offset = y_mean - gain * x_mean
            # strength 插值：out = (1-s)*img + s*(k*img+c)
            gain_eff = 1.0 + (gain - 1.0) * strength
            offset_eff = offset * strength
            # clamp 会削掉本该提亮的高光，8bit 管线下无解，多轮迭代后高光会略微压平
            corrected[b] = (x * gain_eff + offset_eff).clamp_(0.0, 1.0)

            # 诊断量：均值处的修正是拟合的恒等式（直线必过样本均值点），说明不了任何问题，
            # 所以额外报 ±2σ 两端的修正量——gain 一旦失真，这两个数会立刻劈叉。
            lo = (x_mean - 2.0 * x_std).clamp(0.0, 1.0)
            hi = (x_mean + 2.0 * x_std).clamp(0.0, 1.0)
            stats = torch.stack([
                gain,
                offset * 255.0,
                x_std * 255.0,
                (covariance * covariance) / (x_var * y_var).clamp_min(self.EPS),
                (gain_eff * lo + offset_eff - lo) * 255.0,
                (gain_eff * hi + offset_eff - hi) * 255.0,
                was_clamped.to(x_std.dtype),
            ]).t().tolist()
            channel_reports = [
                f"k={k:.5f} c={c:+.3f} std={s:.1f} R2={r2:.4f} @-2s={s_lo:+.2f} @+2s={s_hi:+.2f}"
                + (" CLAMPED" if clamped else "")
                for k, c, s, r2, s_lo, s_hi, clamped in stats
            ]
            report_lines.append(f"[{b}] n={sample_count} " + " | ".join(channel_reports))

        report = "\n".join(report_lines)
        logger.debug(f"[MaskedColorMatch]\n{report}")
        return (corrected, report)


class InpaintRegionColorFix:
    """把重绘区的色彩基准拉回原图，按遮罩值在「遮罩外」和「遮罩内」两个基准之间过渡。

    与 MaskedColorMatch 的本质区别：遮罩外有 ground truth（内容未改动，原图就是答案），
    拟合出来的是测量值；遮罩内没有 ground truth，任何校正都是在编码一条假设——
    「重绘不应该改变该区域的平均亮度（components 选 luminance_and_chroma 时还包括平均色度）」。

    默认只校正亮度：亮度损失跨内容、跨采样器都表现为同向的系统性偏移，而色度偏移的方向
    随内容变化，与真实重绘意图分不开，一并拉回就会撤销重绘本该带来的颜色变化。

    核心区基准默认取均值：只做细化时它更准，而它「跟随内容变化」的风险已经由 max_excess
    封顶。区域可能被大面积重绘时改用中位数，它不会被那部分带跑。

    高 denoise 的内容替换场景上述假设不成立（重绘本来就该改变颜色）。那种情况下把
    inside_source 切到 manual：改用手工标定的常量偏移，完全不看重绘区内容，于是假设从
    「内容属性」变成「管道属性」——代价是换一组 模型/采样器/步数/denoise 就要重新标定。

    校正量按遮罩值插值，与偏移量随遮罩强度递增的实际趋势同向，羽化带不会被过校正，
    也不会引入新的接缝。
    """

    # 遮罩外总体是整张图的大头，下限跟 MaskedColorMatch 对齐。
    MIN_OUTSIDE_SAMPLES = 1024
    MIN_OUTSIDE_RATIO = 0.002
    # 核心区下限只做数值保护，不做统计准入。退回遮罩外基准不是中性的「更安全」，
    # 它是一个已知偏了整整一个 drift 量的选择；而局部估计的样本外误差远小于此，
    # 即使在很小的区域上也是如此。区域越小，局部测量相对全局常数的优势反而越大，
    # 所以这里不该有一个会把小重绘挡在外面的门槛。
    MIN_INSIDE_SAMPLES = 64
    # estimator=clip 迭代时，某通道截断后剩余像素少于此值就冻结该通道。与
    # MIN_INSIDE_SAMPLES 同值纯属巧合：那个管核心区总体准入，这个管「还剩多少
    # 样本才敢再走一步」。
    MIN_CLIP_INLIERS = 64

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The VAE-decoded inpaint result to correct."}),
                "reference": ("IMAGE", {"tooltip": "The original image before VAE encode, same framing as image."}),
                "mask": ("MASK", {"tooltip": "The inpaint mask. Its feather is used as the blend ramp between the outside and inside baselines."}),
                "components": (["luminance", "luminance_and_chroma"], {
                    "default": "luminance",
                    "tooltip": "luminance shifts L* only, so hue and saturation are left alone except where the shift pushes a pixel out of gamut - the safe default, since L* drift is the only part that measures as systematic. luminance_and_chroma also pulls a*/b* back, which fixes colour casts but undoes intentional colour changes in the repainted area."
                }),
                "inside_source": (["fit", "manual"], {
                    "default": "fit",
                    "tooltip": "fit measures the repainted core's baseline from the image, which assumes repainting should not change the region's mean. manual ignores the core entirely and uses inside_offset_l as the baseline shift, so it never fights an intentional colour change - but it has to be calibrated per model/sampler/steps/denoise."
                }),
                "inside_offset_l": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 0.1,
                    "tooltip": "Baseline shift for the repainted core, in Lab L* units; positive brightens. Added on top of the measured shift in fit mode, used as the whole shift in manual mode."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blends the correction: 0 returns the input untouched, 1 applies the full baseline shift."
                }),
                "outside_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask values at or below this define the outside population, whose baseline shift is a real measurement. Must stay below inside_threshold."
                }),
                "inside_threshold": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask values at or above this define the repainted core, whose baseline shift is an assumption. A mask whose peak never reaches this value gets no inside correction - which is correct, since such a mask barely repaints anything."
                }),
                "max_excess": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Cap, in Lab units, on how far the repainted core's baseline may sit from the outside one; 0 disables the cap. Drift alone stays well inside it, so the cap only binds when the repaint really changed the content - it then limits the damage instead of letting the correction undo that change. For heavy content replacement, inside_source=manual is still better."
                }),
                "estimator": (["mean", "median", "clip"], {
                    "default": "mean",
                    "tooltip": "How the repainted core's baseline is summarised. mean is marginally more accurate when the repaint only refined detail, but it tracks any real content change one-for-one and is unbounded. median gives that up for a small, fixed cost and stays close to the drift even when part of the region was genuinely repainted. clip assumes the repaint only altered part of the region: it locks onto the dominant offset and discards pixels that moved far from it, controlled by clip_k. Pick it when a minority of the core was really repainted and the rest only drifted. If instead the whole region was reworked there is no untouched majority to lock onto, and it latches onto whichever mode dominates and over-corrects past mean - so this is a scenario switch, not a safer default. Only affects the core - the outside baseline always uses the mean, where there is no content change to be robust against."
                }),
                "clip_k": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "For estimator=clip: how far a core pixel may sit from the current offset estimate before it stops counting towards it, in units of the outside population's robust dispersion. That dispersion is small, so this is the width of a narrow mode-seeking window rather than an outlier threshold - the default already leaves most of the core out, and raising k widens the window without converging on the plain mean. Smaller rejects harder. Taking the unit from the outside keeps it meaningful across images of similar detail, but a core whose texture differs a lot from its surroundings will still need a different k."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "fix_region"
    CATEGORY = "Slowargo"
    DESCRIPTION = "Pull the repainted region's colour baseline back to the original in Lab space, ramping the correction by mask value so the feathered edge stays seamless. Corrects the sampler-induced drift that MaskedColorMatch cannot see."

    @staticmethod
    def _masked_mean(lab_hwc: torch.Tensor, selection: torch.Tensor, count: int) -> torch.Tensor:
        return (lab_hwc * selection.unsqueeze(-1)).sum(dim=(0, 1)) / count

    @staticmethod
    def _masked_median(diff_hwc: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
        # 必须是「逐像素差值的中位数」，不能是「两个中位数之差」——后者不再是配对
        # 估计量，内容方差不会抵消。
        return diff_hwc[selection].median(dim=0).values

    @classmethod
    def _masked_clip(cls, diff_hwc: torch.Tensor, selection: torch.Tensor,
                     out_diff: torch.Tensor, k: float, iters: int = 200):
        """中心自由的迭代截断：找核心区的主导偏移，把离它太远的像素当成重绘内容剔除。

        中心必须是自由参数。用「接近遮罩外基准」来筛像素会把待估的量当成筛选条件，
        估计量被结构性地拉回遮罩外值，无论真实偏移是多少都得到零——看着很稳，实则退化。

        尺度取自遮罩外的离散度。那里内容没有改动，所以离散度只反映重建误差而非重绘，
        用它当单位比凭空给一个 Lab 绝对值合理；但它随局部细节密度变化，核心区的纹理
        统计与遮罩外不同时，同一个 k 的实际松紧程度也会跟着变。

        迭代是单调慢漂而非震荡，轮数上限必须给够：截断过早等于让「走了几步」成为
        估计量定义的一部分，同一张图换个上限就是另一个答案。

        返回 (center, info)。info 里带每通道保留像素数、实际轮数、以及哪些通道在第一
        轮就被冻结（即该通道实际退化成了中位数），供 report 暴露出来——否则用户无法
        知道自己选的 clip 到底跑成了什么。
        """
        din = diff_hwc[selection]
        center = din.median(dim=0).values
        # 稳健 sigma：MAD 换算到正态等效标准差，不被遮罩外的少量坏像素带跑
        mad = (out_diff - out_diff.median(dim=0).values).abs().median(dim=0).values
        scale = (1.4826 * mad * k).clamp(min=1e-6)

        active = torch.ones_like(center, dtype=torch.bool)
        n_keep = torch.full_like(center, float(din.shape[0]))
        froze_at_once = torch.zeros_like(active)
        used = 0
        for it in range(iters):
            keep = (din - center).abs() <= scale
            cnt = keep.sum(dim=0)
            # 逐通道独立冻结，不是任一通道不行就整体放弃：默认 components=luminance
            # 下 a*/b* 随后会被 keep 向量归零，让一个注定被丢弃的通道有权取消 L* 的
            # 精修是错的。而 clip 的目标场景（少数区域被真重绘）往往正是色相替换，
            # a*/b* 分布宽、L* 才是要估的量。
            newly_frozen = active & (cnt < cls.MIN_CLIP_INLIERS)
            if it == 0:
                froze_at_once = newly_frozen.clone()
            # 在冻结判定之后、break 之前记录：此时 active 仍含刚冻结的通道，report
            # 里才看得到是「剩太少」才停的。全通道同时冻结时也不会漏记。
            n_keep = torch.where(active, cnt.to(n_keep.dtype), n_keep)
            active = active & ~newly_frozen
            if not bool(active.any()):
                break
            nxt = torch.where(active, (din * keep).sum(dim=0) / cnt.clamp(min=1), center)
            moved = (nxt - center).abs() >= 1e-4
            center = nxt
            used = it + 1
            if not bool((active & moved).any()):
                break
        return center, {"n_keep": n_keep, "iters": used, "froze": froze_at_once}

    @staticmethod
    def _fmt_delta(delta: torch.Tensor) -> str:
        return "L*={:+.3f} a*={:+.3f} b*={:+.3f}".format(*delta.tolist())

    def fix_region(self, image, reference, mask, components="luminance", inside_source="fit",
                   inside_offset_l=0.0, strength=1.0, outside_threshold=0.05, inside_threshold=0.95,
                   max_excess=8.0, estimator="mean", clip_k=3.0):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor with 4 dims [B,H,W,C], got shape: {tuple(image.shape)}")
        if image.shape[-1] != 3:
            raise ValueError(f"Expected a 3-channel IMAGE for Lab conversion, got {image.shape[-1]} channels")
        if reference.ndim != 4 or reference.shape[-1] != 3:
            raise ValueError(f"Expected reference IMAGE with shape [B,H,W,3], got: {tuple(reference.shape)}")
        if outside_threshold >= inside_threshold:
            raise ValueError(
                f"outside_threshold ({outside_threshold}) must stay below inside_threshold "
                f"({inside_threshold}); otherwise the two populations overlap and the outside "
                f"baseline gets contaminated by repainted content"
            )

        # 整条链路要过一次 RGB->Lab->RGB 往返，本身就不是无损的；strength 为 0 时
        # 在校验之后短路，保证原样返回，同时不让非法输入蒙混过关。
        if strength == 0:
            return (image.float(), "strength=0, unchanged")

        # kornia 是 comfy_extras/nodes_post_processing.py 的模块级依赖，必然可用；
        # 这里延迟导入只是不想让本扩展的加载路径多挂一个硬依赖，不是可选降级。
        import kornia

        # kornia 的 rgb_to_lab 假定输入在 [0,1]，越界不会产 NaN 但会让 Lab 值跑飞、
        # 统计量随之失真。上游并非所有 VAE 分支都 clamp，这里自己兜住。
        img = image.float().clamp(0.0, 1.0)
        batch, height, width, _ = img.shape

        ref = reference.float().to(img.device).clamp(0.0, 1.0)
        if ref.shape[1:3] != (height, width):
            logger.warning(
                f"[InpaintRegionColorFix] reference size {tuple(ref.shape[1:3])} != image size {(height, width)}, resizing reference"
            )
            ref = _resize_bchw_to(ref.permute(0, 3, 1, 2), (height, width)).permute(0, 2, 3, 1)
        ref = _broadcast_batch(ref, batch, "reference")

        msk = _normalize_mask(mask, height, width, img.device)
        # 两个统计总体都往保守方向收一格：外部用 max 膨胀遮罩后再取阈，核心区用 min
        # 腐蚀后再取阈，避免过渡带同时混进两边。ramp 用的是未经形态学的原始遮罩，
        # 所以核心边缘会欠校正约一格、外部边缘会渗入约一格，量级可忽略。
        grown = F.max_pool2d(msk.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        outside = _broadcast_batch(grown <= outside_threshold, batch, "mask")
        manual_inside = inside_source == "manual"
        if manual_inside:
            # 不看重绘区内容，核心区总体连算都不用算
            inside = None
        else:
            shrunk = -F.max_pool2d(-msk.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
            inside = _broadcast_batch(shrunk >= inside_threshold, batch, "mask")
        msk = _broadcast_batch(msk, batch, "mask")

        lab_img = kornia.color.rgb_to_lab(img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        lab_ref = kornia.color.rgb_to_lab(ref.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        chroma = components == "luminance_and_chroma"
        keep = torch.tensor([1.0, 1.0, 1.0] if chroma else [1.0, 0.0, 0.0],
                            dtype=lab_img.dtype, device=lab_img.device)
        offset = torch.tensor([inside_offset_l, 0.0, 0.0], dtype=lab_img.dtype, device=lab_img.device)
        min_outside = max(self.MIN_OUTSIDE_SAMPLES, int(height * width * self.MIN_OUTSIDE_RATIO))

        lab_out = torch.empty_like(lab_img)
        est_desc = f"{estimator}(k={clip_k:.1f})" if estimator == "clip" else estimator
        report_lines = [f"components={components} inside_source={inside_source} estimator={est_desc} "
                        f"inside_offset_l={inside_offset_l:+.2f} strength={strength:.2f} "
                        f"outside<={outside_threshold:.2f} inside>={inside_threshold:.2f} "
                        f"max_excess={max_excess:.1f}"]

        for b in range(batch):
            n_out = int(outside[b].sum().item())
            flags = []
            clip_note = ""

            outside_ok = n_out >= min_outside
            if outside_ok:
                delta_out = (self._masked_mean(lab_ref[b], outside[b], n_out)
                             - self._masked_mean(lab_img[b], outside[b], n_out))
            else:
                logger.warning(
                    f"[InpaintRegionColorFix] batch {b}: outside population too small "
                    f"({n_out} < {min_outside}), treating the outside baseline as unshifted"
                )
                delta_out = torch.zeros_like(keep)
                flags.append("OUTSIDE_ZEROED")

            if manual_inside:
                # 基准完全由 inside_offset_l 给出，它在下面统一加上，这里先置零
                n_in = "manual"
                delta_in = torch.zeros_like(keep)
            else:
                n_in = int(inside[b].sum().item())
                if n_in < self.MIN_INSIDE_SAMPLES:
                    logger.warning(
                        f"[InpaintRegionColorFix] batch {b}: repainted core too small "
                        f"({n_in} < {self.MIN_INSIDE_SAMPLES}), falling back to the outside baseline"
                    )
                    delta_in = delta_out
                    flags.append("INSIDE_FALLBACK")
                else:
                    if estimator == "median":
                        delta_in = self._masked_median(lab_ref[b] - lab_img[b], inside[b])
                    elif estimator == "clip":
                        diff_b = lab_ref[b] - lab_img[b]
                        if outside_ok:
                            delta_in, info = self._masked_clip(
                                diff_b, inside[b], diff_b[outside[b]], clip_k)
                            clip_note = " clip[keep={} it={}]".format(
                                "/".join(f"{int(v)}" for v in info["n_keep"].tolist()),
                                info["iters"])
                            if bool(info["froze"].any()):
                                flags.append("CLIP_AT_MEDIAN")
                        else:
                            # 尺度取自遮罩外的离散度；那里的样本连均值都测不出来，
                            # 更给不出可信的离散度。退回中位数而不是崩掉。
                            delta_in = self._masked_median(diff_b, inside[b])
                            flags.append("CLIP_NO_SCALE")
                    else:
                        delta_in = (self._masked_mean(lab_ref[b], inside[b], n_in)
                                    - self._masked_mean(lab_img[b], inside[b], n_in))
                    # 判据取 delta_in 超出 delta_out 的部分：delta_in 自身含全局 VAE
                    # 偏置，上游没接 MaskedColorMatch 时直接拿它比会误报。
                    # 也不受 components 过滤——luminance 模式下纯色相替换的 ΔL* 很小，
                    # 只有 a*/b* 会暴露「假设不成立」，过滤后就永远看不到了。
                    excess = delta_in - delta_out
                    if max_excess > 0 and bool((excess.abs() > max_excess).any()):
                        logger.warning(
                            f"[InpaintRegionColorFix] batch {b}: inside baseline exceeds the outside one by "
                            f"{[round(v, 2) for v in excess.tolist()]}, over max_excess={max_excess} - "
                            f"this is more likely a real content change than drift, capping the correction. "
                            f"Use inside_source=manual with a calibrated offset for heavy content replacement."
                        )
                        # 限幅而不是放弃：退回 delta_out 等于主动接受一整个 drift 量的
                        # 偏差，而封顶只是不让校正跟着内容变化跑远，方向仍然是对的。
                        delta_in = delta_out + excess.clamp(-max_excess, max_excess)
                        flags.append("CLAMPED")

            # 上游任何一个坏像素都会让均值变成 NaN，再逐像素相加就是整图报废
            if not bool(torch.isfinite(torch.stack([delta_out, delta_in])).all()):
                logger.warning(
                    f"[InpaintRegionColorFix] batch {b}: non-finite baseline shift "
                    f"(NaN/Inf in image or reference), skipping correction"
                )
                lab_out[b] = lab_img[b]
                report_lines.append(f"[{b}] SKIPPED non-finite statistics")
                continue

            # offset 不受 components 过滤：暴露出来的就只有 L*，用户设了就该生效
            delta_out = delta_out * keep
            delta_in = delta_in * keep + offset
            # 按遮罩值在两个基准之间线性过渡。下游 ImageCompositeMasked 是
            # mask * source + (1 - mask) * destination，所以相对「未校正合成图」的净
            # 修正是 m * delta_out + m^2 * (delta_in - delta_out)：一次项对应全局 VAE
            # 分量，二次项对应内外基准差。m→0 时连续趋零，不产生新接缝。
            ramp = msk[b].unsqueeze(-1)
            lab_out[b] = lab_img[b] + (delta_out + (delta_in - delta_out) * ramp) * strength

            suffix = (" " + " ".join(flags)) if flags else ""
            report_lines.append(
                f"[{b}] n_out={n_out} n_in={n_in} | outside {self._fmt_delta(delta_out)} "
                f"| inside {self._fmt_delta(delta_in)}{clip_note}{suffix}"
            )

        # L* 平移会把高光推出色域：R/G 截到 1.0 而 B 仍在上升，净效果是色相/饱和度
        # 位移而非单纯提亮，迭代多轮后高光会逐步压平。8bit 管线下无解。
        # 截断实际发生在 kornia 内部（lab_to_rgb 的 clip 默认为 True），外面这层
        # clamp 只是防它哪天改默认值。
        corrected = kornia.color.lab_to_rgb(lab_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).clamp_(0.0, 1.0)
        report = "\n".join(report_lines)
        logger.debug(f"[InpaintRegionColorFix]\n{report}")
        return (corrected, report)


class RunButtonNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 这里的参数名要与前端对应
                "trigger_count": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("trigger_count",)

    FUNCTION = "do_run"
    CATEGORY = "Slowargo"

    def do_run(self, trigger_count):
        # print(f"按钮被点击了！当前触发次数: {trigger_count}")
        return trigger_count

def resolve_server_file_source_path(source_path):
    source_text = str(source_path or "").strip()
    if not source_text:
        raise ValueError("source_path is required")

    source = Path(source_text).expanduser()
    if source.exists() or source.is_absolute():
        return source

    output_dir = Path(folder_paths.get_output_directory())
    return Path(folder_paths.get_annotated_filepath(source_text, output_dir)).expanduser()


def transfer_server_file(source_path, target_dir, target_filename, move_file=False):
    target_dir_text = str(target_dir or "").strip()
    target_filename_text = str(target_filename or "").strip()

    if not target_dir_text:
        raise ValueError("target_dir is required")

    source = resolve_server_file_source_path(source_path)
    target_directory = Path(target_dir_text).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Source path is not a file: {source}")

    filename = os.path.basename(target_filename_text) if target_filename_text else source.name
    if not filename:
        raise ValueError("target_filename is required when source_path has no file name")

    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / filename

    try:
        if source.resolve() == target.resolve():
            raise ValueError("Source and target paths are the same")
    except FileNotFoundError:
        # The target may not exist yet on some Python/platform combinations.
        pass

    if target.exists() and target.is_dir():
        raise ValueError(f"Target path is a directory: {target}")

    if move_file:
        shutil.move(str(source), str(target))
        action = "moved"
    else:
        shutil.copy2(str(source), str(target))
        action = "copied"

    return {
        "success": True,
        "source_path": str(source),
        "target_path": str(target),
        "message": f"File {action} to {target}",
    }


class ServerFileTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": "", "tooltip": "Source file path on the server. Empty can use image_source."}),
                "target_dir": ("STRING", {"default": "", "tooltip": "Target directory on the server."}),
                "target_filename": ("STRING", {"default": "", "tooltip": "Target file name. Empty uses the source file name. Parent paths are ignored."}),
                "move_file": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Move",
                    "label_off": "Copy",
                    "tooltip": "Move the source file instead of copying it."
                }),
                "auto_execute": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto",
                    "label_off": "Manual",
                    "tooltip": "Auto runs during prompt execution. Manual runs only from the node button."
                }),
            },
            "optional": {
                "image_source": ("*", {"tooltip": "Connect to any output of a node that has an image widget."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("source_path", "target_path", "success", "message")
    DESCRIPTION = "Copy or move a server-side file to a target directory and file name."
    FUNCTION = "execute"
    CATEGORY = "Slowargo"
    OUTPUT_NODE = True
    NOT_IDEMPOTENT = True

    def execute(self, source_path="", target_dir="", target_filename="", move_file=False, auto_execute=True, image_source=None, prompt=None, node_id=None):
        if not auto_execute:
            return (source_path, "", False, "Skipped: manual mode")

        try:
            source_path = resolve_image_widget_source_path(prompt or {}, node_id, "image_source")
        except Exception:
            if not str(source_path or "").strip():
                raise

        result = transfer_server_file(source_path, target_dir, target_filename, move_file)
        return (
            result["source_path"],
            result["target_path"],
            result["success"],
            result["message"],
        )

class ClearHistoryNode(io.ComfyNode):
    """Clear canvas editing history"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ClearHistoryNode",
            display_name="Clear History",
            category="Slowargo",
            inputs=[
                io.Boolean.Input(
                    "auto_clear",
                    default=True,
                    display_name="Auto Clear on Execute"
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, auto_clear: bool = True) -> io.NodeOutput:
        return io.NodeOutput()

    @classmethod
    def IS_CHANGED(cls, auto_clear: bool = True):
        return auto_clear

##############################################

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./js"

# Add custom API routes, using router

@PromptServer.instance.routes.post("/slowargo_api/refresh_previews")
async def refresh_previews_api(request):
    try:
        data = await request.json()
        # if not data.get("input_path"):
        #     raise ValueError("No input path provided")

        # Extract parameters
        # load_limit = int(data.get("load_limit", "1000"))
        # start_index = int(data.get("start_index", 0))
        # stop_index = int(data.get("stop_index", 10))
        # include_subfolders = data.get("include_subfolders", False)
        # filter_type = data.get("filter_type", "none")
        # sort_method = data.get("sort_method", "date_modified")
        success = True

        # use get_file_names to get the lates file names
        # 用户手动点刷新时绕过目录缓存，强制重扫
        file_names = LoadImageFromOutputsPlus.get_file_names(data["input_path"], use_cache=False)

        return web.json_response({
            "success": success,
            # "message": message,
            "image_name": file_names,
            # "thumbnails": thumbnails,
            # "total_images": total_available,
            # "visible_images": len(thumbnails),
            # "start_index": start_index,
            # "stop_index": stop_index,
            # "image_order": image_order
        })

    except Exception as e:
        logger.error(f"Error in refresh_previews route: {str(e)}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)

@PromptServer.instance.routes.get("/slowargo_api/refresh_previews")
async def refresh_previews_v1_api(request):
    try:
        # use get_file_names to get the lates file names
        # 用户手动点刷新时绕过目录缓存，强制重扫
        file_names = LoadImageFromOutputPlusV1.get_file_names(use_cache=False)

        return web.json_response(file_names)

    except Exception as e:
        logger.error(f"Error in refresh_previews route: {str(e)}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)

@PromptServer.instance.routes.post("/slowargo_api/refresh_previews_recent")
async def refresh_previews_recent_api(request):
    try:
        # logger.info(f"[refresh_previews_recent] request: {request}")
        data = await request.json()
        # logger.info(f"[refresh_previews_recent] data: {data}")

        # use get_file_names to get the lates file names
        # 用户手动点刷新时绕过目录缓存，强制重扫
        file_names = LoadRecentImagePlusV1.get_file_names(data["watch_folders"], use_cache=False)

        return web.json_response({
            "success": True,
            "image_name": file_names,
        })

    except Exception as e:
        logger.error(f"Error in refresh_previews_recent route: {str(e)}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)

@PromptServer.instance.routes.post("/slowargo_api/file_transfer")
async def file_transfer_api(request):
    try:
        data = await request.json()
        source_path = data.get("image_source") or data.get("source_path", "")
        result = transfer_server_file(
            source_path,
            data.get("target_dir", ""),
            data.get("target_filename", ""),
            data.get("move_file", False),
        )
        return web.json_response(result)

    except Exception as e:
        logger.error(f"Error in file_transfer route: {str(e)}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)

# 获取历史记录接口
@PromptServer.instance.routes.get("/slowargo_api/get_string_history")
async def get_string_history_api(request):
    store_file = request.query.get("store_file", "")
    _, stored_entries = RememberStrings.read_stored_strings(store_file)
    return web.json_response({"entries": stored_entries})

# Toggle Pin 接口 (修改返回值为最新列表)
@PromptServer.instance.routes.post("/slowargo_api/toggle_string_history_pin")
async def toggle_string_history_pin_api(request):
    json_data = await request.json()
    content = json_data.get("content")
    store_file = json_data.get("store_file")

    file_path, stored_entries = RememberStrings.read_stored_strings(store_file)

    for entry in stored_entries:
        if entry["content"] == content:
            entry["pinned"] = not entry.get("pinned", False)
            break

    # 排序：Pin 优先，其余按位置
    stored_entries.sort(key=lambda x: x.get("pinned", False), reverse=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(stored_entries, f, ensure_ascii=False, indent=2)

    return web.json_response({"entries": stored_entries})

# 3. 删除记录接口
@PromptServer.instance.routes.post("/slowargo_api/delete_string_history")
async def delete_entry_api(request):
    json_data = await request.json()
    content = json_data.get("content")
    store_file = json_data.get("store_file")

    file_path, stored_entries = RememberStrings.read_stored_strings(store_file)

    # 过滤掉匹配的内容
    initial_count = len(stored_entries)
    stored_entries = [entry for entry in stored_entries if entry["content"] != content]

    # 只有在确实删除了内容时才写入文件
    if len(stored_entries) < initial_count:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stored_entries, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"[RememberStrings] Delete save error: {e}")

    return web.json_response({"entries": stored_entries})

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
