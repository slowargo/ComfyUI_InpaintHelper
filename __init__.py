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

logger = logging.getLogger(__name__)

class Example(io.ComfyNode):
    """
    An example node

    Class methods
    -------------
    define_schema (io.Schema):
        Tell the main program the metadata, input, output parameters of nodes.
    fingerprint_inputs:
        optional method to control when the node is re executed.
    check_lazy_status:
        optional method to control list of input names that need to be evaluated.

    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
            Return a schema which contains all information about the node.
            Some types: "Model", "Vae", "Clip", "Conditioning", "Latent", "Image", "Int", "String", "Float", "Combo".
            For outputs the "io.Model.Output" should be used, for inputs the "io.Model.Input" can be used.
            The type can be a "Combo" - this will be a list for selection.
        """
        return io.Schema(
            node_id="Example",
            display_name="Example Node",
            category="Slowargo",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input(
                    "int_field",
                    min=0,
                    max=4096,
                    step=64, # Slider's step
                    display_mode=io.NumberDisplay.number,  # Cosmetic only: display as "number" or "slider"
                    lazy=True,  # Will only be evaluated if check_lazy_status requires it
                ),
                io.Float.Input(
                    "float_field",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    round=0.001, #The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    display_mode=io.NumberDisplay.number,
                    lazy=True,
                ),
                io.Combo.Input("print_to_screen", options=["enable", "disable"]),
                io.String.Input(
                    "string_field",
                    multiline=False,  # True if you want the field to look like the one on the ClipTextEncode node
                    default="Hello world!",
                    lazy=True,
                ),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def check_lazy_status(cls, image, string_field, int_field, float_field, print_to_screen):
        """
            Return a list of input names that need to be evaluated.

            This function will be called if there are any lazy inputs which have not yet been
            evaluated. As long as you return at least one field which has not yet been evaluated
            (and more exist), this function will be called again once the value of the requested
            field is available.

            Any evaluated inputs will be passed as arguments to this function. Any unevaluated
            inputs will have the value None.
        """
        if print_to_screen == "enable":
            return ["int_field", "float_field", "string_field"]
        else:
            return []

    @classmethod
    def execute(cls, image, string_field, int_field, float_field, print_to_screen) -> io.NodeOutput:
        if print_to_screen == "enable":
            logger.info(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """)
        #do some processing on the image, in this example I just invert it
        image = 1.0 - image
        return io.NodeOutput(image)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def fingerprint_inputs(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


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
        directory = folder_paths.get_output_directory()
        VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

        # 安全拼接 sub_folder
        directory = os.path.join(directory, sub_folder)

        def is_valid_image(filename: str) -> bool:
            # logger.info(f"[is_valid_image]: files: {filename}")
            return any(filename.lower().endswith(ext) for ext in VALID_EXTS)

        def is_visible_file(entry: os.DirEntry) -> bool:
            """Filter out hidden files (e.g., .DS_Store on macOS)."""
            return entry.is_file() and not entry.name.startswith('.') and is_valid_image(entry.name)

        files = [entry for entry in os.scandir(directory) if is_visible_file(entry)]
        sorted_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
        file_names = [entry.name for entry in sorted_files]
        # logger.info(f"[get_file_names]: files:{directory} -> {file_names}")
        return file_names

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
        # directory = folder_paths.get_output_directory()
        # directory = os.path.join(directory, image_folder)
        # image_path = os.path.join(directory, image)
        # try:
        #     with Image.open(image_path) as img:
        #         w, h = img.size
        #         c = len(img.getbands())
        #         normalized = np.array(img).astype(np.float32) / 255.0
        #         tensor = torch.from_numpy(normalized).reshape(1, h, w, c)
        #         match c:
        #             case 1:
        #                 oimage = tensor.expand(1, h, w, 3)
        #                 mask = tensor.reshape(1, h, w)
        #             case 3:
        #                 oimage = tensor
        #                 mask = tensor[..., 0]
        #             case 4:
        #                 oimage = tensor[..., :3]
        #                 mask = tensor[..., 3]
        #         # read meta data
        #         meta_data = cls.get_image_metadata(image_path)
        #         logger.info(f"meta_data: {meta_data}")
        #         return io.NodeOutput(oimage, mask,image, meta_data)
        # except Exception as e:
        #     logger.error(f"Error: {e}")
        #     return None

        try:
            def_dir = folder_paths.get_output_directory()
            def_dir = os.path.join(def_dir, image_folder)

            # image_folder as default folder
            image_path = folder_paths.get_annotated_filepath(image, def_dir)
            logger.info(f"[LoadImageFromOutputsPlus] image:{image} image_folder:{image_folder} def_dir: {def_dir} -> {image_path}")

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

            return (output_image, output_mask)
        except Exception as e:
            logger.error(f"Error: {e}")
            return None

#######################################################################################################################
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
                    "tooltip": "Float A (Toggle ON)"
                }),
                "float_b": ("FLOAT", {
                    "default": 0.55,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.02,
                    "round": 0.01,
                    "display": "slider",
                    "tooltip": "Float B (Toggle OFF)"
                }),
                "toggle": ("BOOLEAN", {
                    "default": False,
                    "label_on": "ON",
                    "label_off": "OFF",
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
                    "tooltip": "Float C - 如果 > 0 则强制使用此值"
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
    DESCRIPTION = "Load an image from the output folder. When the refresh button is clicked, the node will update the image list and automatically select the first image, allowing for easy iteration."
    EXPERIMENTAL = True
    FUNCTION = "load_image"
    CATEGORY = "Slowargo"

    def load_image(self, image, sub_folder=""):
        # 获取输出目录
        output_dir = Path(folder_paths.get_output_directory()) / sub_folder
        # 构造图像路径
        image_path = folder_paths.get_annotated_filepath(image, output_dir)

        file_name = image_path
        meta_data = ""

        # if image.endswith("[output]"):
        #     # try sub_folder

        logger.info(f"[LoadImageFromOutputPlusV1] image:{image} sub_folder:{sub_folder} -> {image_path}")

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

        # logger.info(f"[LoadImageFromOutputPlusV1] img:{img}")
        if img.format == 'PNG':
            meta_data = img.info

        return (output_image, output_mask, os.path.basename(file_name), meta_data)

    @staticmethod
    def get_file_names(sub_folder="") -> List[str]:
        VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        """
        获取 output 目录 + clipspace 目录中最新的圖片文件名（按修改時間排序）
        - output 目录下最多取 10 張
        - clipspace 目录下最多取 3 張，並標記為 [input]
        """
        output_dir = Path(folder_paths.get_output_directory()) / sub_folder
        clipspace_dir = Path(folder_paths.get_input_directory()) / "clipspace"

        def is_valid_image_file(entry: os.DirEntry) -> bool:
            """判斷是否為可見的圖片文件"""
            return (
                    entry.is_file()
                    and not entry.name.startswith(".")
                    and Path(entry.name).suffix.lower() in VALID_IMAGE_EXTS
            )

        def get_recent_files(directory: Path, max_count: int) -> List[Path]:
            """獲取目錄下最新的 max_count 個圖片文件"""
            if not directory.is_dir():
                return []

            files = [
                entry.path
                for entry in os.scandir(directory)
                if is_valid_image_file(entry)
            ]
            # 按修改時間降序排序
            sorted_paths = sorted(
                (Path(p) for p in files),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            return sorted_paths[:max_count]

        # 獲取兩組文件
        output_files = get_recent_files(output_dir, 10)
        clipspace_files = get_recent_files(clipspace_dir, 3)

        # 建立 (路徑, 顯示名稱) 的對應關係
        file_items = []

        # output 目錄的文件
        for path in output_files:
            rel_path = path.relative_to(output_dir).as_posix()
            file_items.append((path, rel_path))

        # clipspace 目錄的文件 + 標記
        # 前端会自动识别处理 annotation (see createAnnotatedPath)
        for path in clipspace_files:
            rel_path = path.relative_to(clipspace_dir).as_posix()
            display_name = f"clipspace/{rel_path} [input]"
            file_items.append((path, display_name))

        # 按修改時間重新整體排序（最重要的步驟）
        file_items.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

        # 只取最終顯示名稱
        final_names = [display_name for _, display_name in file_items]

        # logger.info(f"[get_file_names] {output_dir} + clipspace → {final_names}")
        return final_names

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

    def load_image(self, image, watch_folders=""):
        # 构造图像路径
        image_path = folder_paths.get_annotated_filepath(image)

        file_name = image_path
        meta_data = ""

        # if image.endswith("[output]"):
        #     # try sub_folder

        # logger.info(f"[LoadImageFromOutputPlusV1] image:{image} -> {image_path}")

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

        # logger.info(f"[LoadImageFromOutputPlusV1] img:{img}")
        if img.format == 'PNG':
            meta_data = img.info

        return (output_image, output_mask, os.path.basename(file_name), meta_data)

    @staticmethod
    def get_file_names(watch_folders="") -> List[str]:
        VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        """
        获取 output 目录 + clipspace 目录中最新的圖片文件名（按修改時間排序）
        - output 目录下最多取 10 張
        - clipspace 目录下最多取 3 張，並標記為 [input]
        """
        # 初始化文件列表
        file_items = []
        
        def is_valid_image_file(entry: os.DirEntry) -> bool:
            """判斷是否為可見的圖片文件"""
            return (
                entry.is_file()
                and not entry.name.startswith(".")
                and Path(entry.name).suffix.lower() in VALID_IMAGE_EXTS
            )

        def get_recent_files(directory: Path, max_count: int) -> List[Path]:
            """獲取目錄下最新的 max_count 個圖片文件"""
            if not directory.is_dir():
                return []
                
            files = [
                entry.path
                for entry in os.scandir(directory)
                if is_valid_image_file(entry)
            ]
            # 按修改時間降序排序
            sorted_paths = sorted(
                (Path(p) for p in files),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            return sorted_paths[:max_count]

        # parse watch_folders
        watch_folders = watch_folders.strip()
        if not watch_folders:
            watch_folders = "[5][input]"
            logger.info(f"[LoadRecentImagePlusV1] use default watch_folders: {watch_folders}")

        folder_items = watch_folders.split(";")
        # 解析每个文件夹项
        for item in folder_items:
            item = item.strip()
            if not item:
                continue
                
            try:
                # 正则提取 sub_folder, 数量和类型 - 支持空sub_folder
                match = re.match(r"^(.*?)\s*\[(\d+)\]\[(.*?)\]$", item)
                if not match:
                    logger.warning(f"[LoadRecentImagePlusV1] 无效的文件夹配置项: {item}")
                    continue
                
                sub_folder, count_str, folder_type = match.groups()
                sub_folder = sub_folder.strip()
                count = int(count_str)
                
                # 根据类型设置目录
                if folder_type == "output":
                    directory = Path(folder_paths.get_output_directory()) / sub_folder
                elif folder_type == "input":
                    directory = Path(folder_paths.get_input_directory()) / sub_folder
                else:
                    logger.warning(f"[LoadRecentImagePlusV1] 未知目录类型: {folder_type}")
                    continue
                    
                # 获取最新文件
                recent_files = get_recent_files(directory, count)
                for path in recent_files:
                    rel_path = path.relative_to(directory).as_posix()
                    display_name = f"{sub_folder}/{rel_path} [{folder_type}]" if sub_folder else f"{rel_path} [{folder_type}]"
                    file_items.append((path, display_name))
                    
            except (ValueError, AttributeError) as e:
                logger.warning(f"[LoadRecentImagePlusV1] 解析文件夹配置项失败: {item}, 错误: {str(e)}")
                continue

        # 按修改時間重新整體排序（最重要的步驟）
        file_items.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

        # 只取最終顯示名稱
        final_names = [display_name for _, display_name in file_items]

        return final_names

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

    def load_image(self, image_path):
        logger.info(f"[LoadImageFromAnyPath] -----> image:{image_path}")
        # image_path = image
        meta_data = ""

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

        # logger.info(f"[LoadImageFromOutputPlusV1] img:{img}")
        if img.format == 'PNG':
            meta_data = img.info

        return (output_image, output_mask, image_path, meta_data)

    @classmethod
    def IS_CHANGED(s, image_path):
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
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    FUNCTION = "save_image"
    CATEGORY = "Slowargo"

    def save_image(self, image, filename="ComfyUI_output", sub_folder="", meta_data=None,prompt=None, extra_pnginfo=None):
        """
        保存圖像，支持：
        - 根據 filename 後綴自動選擇格式（.png / .jpg / .jpeg / .webp 等）
        - 優先使用傳入的 meta_data（字符串 JSON 格式，包含原始 PNG 元數據）
        - 其次使用 ComfyUI 標準的 prompt / extra_pnginfo
        - PNG 格式完整保留元數據（文本 + ICC Profile）
        - 其他格式（如 JPG/WEBP）不寫元數據（因為不支援或不推薦）
        """
        full_output_folder = Path(folder_paths.get_output_directory()) / sub_folder
        full_output_folder.mkdir(parents=True, exist_ok=True)

        results = []
        img = None
        for (batch_number, image) in enumerate(image):
            # 轉換為 Pillow Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            break

        filename = os.path.basename(filename)

        # 解析文件名與格式
        file_path = Path(filename)
        suffix = file_path.suffix.lower()  # 如 ".png", ".jpg"
        stem = file_path.stem              # 無後綴名

        # 默認 PNG（如果無後綴）
        if not suffix:
            suffix = ".png"
            file_name = f"{stem}.png"
        else:
            file_name = filename

        full_file_path = full_output_folder / file_name

        # 準備保存參數
        save_kwargs = {
            "compress_level": 5,
            # "optimize": True
        }

        pnginfo = None
        icc_profile = None

        # 優先級 1：用戶傳入的 meta_data（字符串 JSON）
        if meta_data:
            try:
                if suffix == ".png":
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
        if not args.disable_metadata and suffix == ".png" and pnginfo is None:
            pnginfo = PngInfo()

            # 添加 prompt
            if prompt is not None:
                pnginfo.add_text("prompt", json.dumps(prompt))

            # 添加 extra_pnginfo（如 workflow）
            if extra_pnginfo is not None:
                for key in extra_pnginfo:
                    pnginfo.add_text(key, json.dumps(extra_pnginfo[key]))

        # 只有 PNG 才傳 pnginfo 和 icc_profile
        if suffix == ".png":
            if pnginfo:
                save_kwargs["pnginfo"] = pnginfo
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            save_kwargs["format"] = "PNG"
        elif suffix in {".jpg", ".jpeg"}:
            save_kwargs["format"] = "JPEG"
            save_kwargs["quality"] = 95      # 高品質 JPEG
            save_kwargs["optimize"] = True
        elif suffix == ".webp":
            save_kwargs["format"] = "WEBP"
            save_kwargs["quality"] = 95
            save_kwargs["method"] = 6         # 最高壓縮質量
        else:
            # 不支援的格式，強制轉 PNG
            logger.warning(f"[SaveImageToFileName] Unknown format {suffix}，強制保存為 PNG")
            full_file_path = full_output_folder / f"{stem}.png"
            save_kwargs["format"] = "PNG"
            if pnginfo:
                save_kwargs["pnginfo"] = pnginfo
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile

        # 執行保存
        img.save(full_file_path, **save_kwargs)

        # 返回 ComfyUI 標準格式
        results.append({
            "filename": full_file_path.name,
            "subfolder": sub_folder,
            "type": "output"  # 或 self.type，根據你的節點類型調整
        })

        logger.info(f"[SaveImageToFileName] full_output_folder:{full_output_folder} full_file_path:{full_file_path}")

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
        logger.info(f"[ExtractSubFolder] parts:{parts}")
        
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
                    "tooltip": "Float A (Toggle ON)"
                }),
                "float_b": ("FLOAT", {
                    "default": 0.55,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.02,
                    "round": 0.01,
                    "display": "slider",
                    "tooltip": "Float B (Toggle OFF)"
                }),
                "toggle": ("BOOLEAN", {
                    "default": False,
                    "label_on": "ON",
                    "label_off": "OFF",
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
                    "tooltip": "Float C - 如果 > 0 则强制使用此值"
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

##############################################

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./js"

# Add custom API routes, using router

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     print(f"""[get_hello]: request: {request}""")
#     return web.json_response("hello")

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
        include_subfolders = data.get("include_subfolders", False)
        filter_type = data.get("filter_type", "none")
        sort_method = data.get("sort_method", "date_modified")
        success = True

        # # Get files
        # if not os.path.isabs(data["input_path"]):
        #     abs_path = os.path.abspath(os.path.join(folder_paths.get_output_directory(), data["input_path"]))
        # else:
        #     abs_path = data["input_path"]
        #
        # # Get all files and sort them
        # all_files = ImageManager.get_image_files(abs_path, include_subfolders, filter_type)
        # if not all_files:
        #     return web.json_response({
        #         "success": False,
        #         "error": "No valid images found"
        #     })
        #
        # all_files = sorted(all_files,
        #                    key=numerical_sort_key if sort_method == "numerical"
        #                    else os.path.getctime if sort_method == "date_created"
        #                    else os.path.getmtime if sort_method == "date_modified"
        #                    else str)

        # total_available = len(all_files)
        #
        # # Validate and adjust indices
        # start_index = min(max(0, start_index), total_available)
        # stop_index = min(max(start_index + 1, stop_index), total_available)
        #
        # # Calculate how many images to actually load
        # num_images = min(stop_index - start_index, load_limit)
        #
        # # Create thumbnails only for the selected range
        # success, message, thumbnails, image_order = ImageManager.create_thumbnails(
        #     data["input_path"],
        #     include_subfolders=include_subfolders,
        #     filter_type=filter_type,
        #     sort_method=sort_method,
        #     start_index=start_index,
        #     max_images=num_images  # Only create thumbnails for the range we want
        # )

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

# class SlowargoExtensions(ComfyExtension):
#     @override
#     async def get_node_list(self) -> list[type[io.ComfyNode]]:
#         return [
#             # Example,
#             FloatSwitch,
#             LoadImageFromOutputsPlus,
#         ]
#
#
# async def comfy_entrypoint() -> ComfyExtension:  # ComfyUI calls this to load your extension and its nodes.
#     return SlowargoExtensions()

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
    "LoadImageFromOutputPlusV1": "Load Image (from Outputs) Plus V1",
    "LoadImageFromAnyPath": "Load Image (from Any Path)",
    "LoadRecentImagePlusV1": "Load Recent Image Plus V1",
    "SaveImageToFileName": "Save Image to Specified File Name",
    "ExtractSubFolder": "Extract Sub Folder",
}