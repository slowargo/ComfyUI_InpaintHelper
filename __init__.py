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
from .nodes_image_io import (
    ImageWidgetSourcePath,
    LoadImageFromAnyPath,
    LoadImageFromOutputPlusV1,
    LoadRecentImagePlusV1,
    resolve_image_widget_source_path,
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
