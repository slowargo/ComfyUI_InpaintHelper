import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from typing_extensions import override

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
from aiohttp import web
from comfy.cli_args import args
from comfy_api.latest import ComfyExtension, io
import folder_paths
import node_helpers
import nodes
import numpy as np
from server import PromptServer
import torch
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

def get_recent_image_files(
    directories: List[Tuple[str, str, int]],   # (sub_folder相对路径, label标记, 最大数量)
    base_root_getter=None,
    valid_exts: set = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
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

    file_items: List[Tuple[Path, str]] = []  # (绝对路径, 显示名称)

    def is_valid_image(entry: os.DirEntry) -> bool:
        return (
            entry.is_file()
            and not entry.name.startswith('.')
            and Path(entry.name).suffix.lower() in valid_exts
        )

    for sub_folder, label, max_count in directories:
        # 确定实际根目录
        root_dir = base_root_getter(label)
        full_dir = root_dir / sub_folder if sub_folder else root_dir

        if not full_dir.is_dir():
            logger.warning(f"Invalid dir: {full_dir}")
            continue

        recent_files = []
        for entry in os.scandir(full_dir):
            if is_valid_image(entry):
                recent_files.append(Path(entry))

        recent_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        recent_files = recent_files[:max_count]

        for path in recent_files:
            # 相对于 full_dir 的相对路径
            rel_path = path.relative_to(full_dir).as_posix()

            # 构建显示名称
            if sub_folder:
                display_base = f"{sub_folder}/{rel_path}"
            else:
                display_base = rel_path

            if label:
                display_name = f"{display_base} [{label}]"
            else:
                display_name = display_base

            file_items.append((path, display_name))

    # 全局按修改时间重新排序
    file_items.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

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
    def get_file_names(sub_folder="") -> List[str]:
        return get_recent_image_files([
            (sub_folder, "", 9999),  # 無標籤，數量幾乎不限
        ])

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

class LoadImageFromOutputPlusV1(nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(cls):
        file_names = cls.get_file_names()
        return {
            "required": {
                "image": ("COMBO", {
                    "options":file_names,
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
    def get_file_names(sub_folder="") -> List[str]:
        return get_recent_image_files([
            (sub_folder, "", 10),       # output 目录 + sub_folder 前缀，不强制加 [output]
            ("clipspace","input", 3),   # clipspace 固定子目录，加 [input]
        ])

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
    def get_file_names(watch_folders="") -> List[str]:
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

        return get_recent_image_files(directories)

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
                        logger.info(f"[SaveImageToFileName] meta_data key:{key} ")
                        if isinstance(value, str):
                            pnginfo.add_text(key, value)
                        else:
                            enc = json.dumps(value, ensure_ascii=True)
                            # logger.info(f"[SaveImageToFileName] meta_data value:{enc} ")
                            pnginfo.add_text(key, str(enc))

                    # 恢復 ICC Profile
                    if metadata.get("icc_profile_base64"):
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

# class FloatSwitch:
#     """
#     浮点数切换器
#     开关打开(toggle=True) → 输出 float_a
#     开关关闭(toggle=False) → 输出 float_b
#     当 float_ovr > 0 时，强制使用 float_ovr 的值（优先级最高）
#     """
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "float_a": ("FLOAT", {
#                     "default": 0.32,
#                     "min": 0.1,
#                     "max": 0.9,
#                     "step": 0.02,
#                     "round": 0.01,
#                     "display": "slider",
#                     "tooltip": "Float A (Toggle ON)"
#                 }),
#                 "float_b": ("FLOAT", {
#                     "default": 0.55,
#                     "min": 0.1,
#                     "max": 0.9,
#                     "step": 0.02,
#                     "round": 0.01,
#                     "display": "slider",
#                     "tooltip": "Float B (Toggle OFF)"
#                 }),
#                 "toggle": ("BOOLEAN", {
#                     "default": False,
#                     "label_on": "ON",
#                     "label_off": "OFF",
#                 }),
#             },
#             "optional": {
#                 "float_ovr": ("FLOAT", {
#                     "default": 0.0,
#                     "min": 0.0,
#                     "max": 0.9,
#                     "step": 0.02,
#                     "round": 0.01,
#                     "display": "number",
#                     "tooltip": "Float C - 如果 > 0 则强制使用此值"
#                 }),
#             },
#             "hidden": {
#                 "node_id": "UNIQUE_ID"
#             }
#         }
#
#     RETURN_TYPES = ("FLOAT",)
#     RETURN_NAMES = ("selected_float",)
#
#     FUNCTION = "do_switch"
#     CATEGORY = "Slowargo"
#
#     def do_switch(self, float_a, float_b, toggle, float_ovr=0.0, node_id=None):
#         if toggle:
#             selected = float_a
#         else:
#             selected = float_b
#
#         # override 优先级最高
#         if float_ovr > 0:
#             selected = float_ovr
#
#         # logger.info(f"[FloatSwitch] node_id:{node_id} selected:{selected}")
#         # PromptServer.instance.send_sync("slowargo.js.extension.FloatSwitch", {"node_id": node_id, "selected_value": selected})
#
#         return (selected,)

##############################################

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./js"

# Add custom API routes, using router

@PromptServer.instance.routes.post("/slowargo_api/refresh_previews")
async def refresh_previews(request):
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
        file_names = LoadImageFromOutputsPlus.get_file_names(data["input_path"])

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
async def refresh_previews_v1(request):
    try:
        # use get_file_names to get the lates file names
        file_names = LoadImageFromOutputPlusV1.get_file_names()

        return web.json_response(file_names)

    except Exception as e:
        logger.error(f"Error in refresh_previews route: {str(e)}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)

@PromptServer.instance.routes.post("/slowargo_api/refresh_previews_recent")
async def refresh_previews_recent(request):
    try:
        # logger.info(f"[refresh_previews_recent] request: {request}")
        data = await request.json()
        # logger.info(f"[refresh_previews_recent] data: {data}")

        # use get_file_names to get the lates file names
        file_names = LoadRecentImagePlusV1.get_file_names(data["watch_folders"])

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
    "LoadImageFromOutputPlusV1": LoadImageFromOutputPlusV1,
    "LoadImageFromAnyPath": LoadImageFromAnyPath,
    "LoadRecentImagePlusV1": LoadRecentImagePlusV1,
    "SaveImageToFileName": SaveImageToFileName,
    "ExtractSubFolder": ExtractSubFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatSwitch": "Float Switch",
    "LoadImageFromOutputPlusV1": "Load Image (from Outputs) Plus V1 (deprecated)",
    "LoadImageFromAnyPath": "Load Image (from Any Path)",
    "LoadRecentImagePlusV1": "Load Recent Image",
    "SaveImageToFileName": "Save Image to Specified File Name",
    "ExtractSubFolder": "Extract Sub Folder",
}