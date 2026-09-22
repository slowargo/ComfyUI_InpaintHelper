"""图像读取：把磁盘上的图变成 IMAGE 张量，以及枚举「最近的文件」。

三个 Load 节点共用 process_image_to_tensor 与 get_recent_image_files，前端刷新
预览的三条路由也在这里——它们各自只服务一个 Load 节点。

resolve_image_widget_source_path 被 nodes_save 的 ServerFileTransfer 复用，
是本包里唯一跨模块的辅助函数；放在这里因为它解析的就是图像 widget 的来源路径。

路由是 import 时注册的副作用，__init__.py 里对本模块的导入不能删。
"""

import logging
import heapq
import os
import time
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageOps, ImageSequence
from aiohttp import web
from comfy_api.latest import io
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
