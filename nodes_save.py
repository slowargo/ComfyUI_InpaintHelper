"""图像保存、按文件名定位原文件、服务端文件搬运，以及前端触发搬运的那条 API 路由。

路由是 import 时注册的副作用，__init__.py 里对本模块的导入不能删。
"""

import logging
import json
import os
import shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
from aiohttp import web
from comfy.cli_args import args
import folder_paths
import nodes
import numpy as np
from server import PromptServer

from .nodes_image_io import resolve_image_widget_source_path

logger = logging.getLogger(__name__)


class SaveImageToFileName(nodes.SaveImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"default": "ComfyUI_output", "tooltip": (
                    "Output file name. If no suffix, default to .png. Parent paths will be "
                    "ignored."
                )}),
                "sub_folder": ("STRING", {"default": ""}),
                "meta_data": ("STRING", {"default": ""}),
                "force_format": ("COMBO", {"options": ["PNG", "JPEG", "WEBP","auto"], "default": "auto"}),
                "auto_open": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Open After Save",
                    "label_off": "Don't Open After Save",
                })
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_TOOLTIPS = ("Absolute path of the saved file.",)
    FUNCTION = "save_image"
    CATEGORY = "Slowargo"

    def save_image(self, image, filename="ComfyUI_output", sub_folder="", meta_data=None,
                   force_format="auto", auto_open=False, prompt=None, extra_pnginfo=None):
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

        # logger.info(f"[SaveImageToFileName] full_output_folder:{full_output_folder} "
        #             f"full_file_path:{full_file_path} args:{save_kwargs}")
        if auto_open:
            PromptServer.instance.send_sync("slowargo.js.extension.SaveImageToFileName", {"results": results})

        return {"ui": {"images": results}, "result": (str(full_file_path),)}


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


def resolve_server_file_source_path(source_path):
    source_text = str(source_path or "").strip()
    if not source_text:
        raise ValueError("source_path is required")

    source = Path(source_text).expanduser()
    if source.exists() or source.is_absolute():
        return source

    output_dir = Path(folder_paths.get_output_directory())
    return Path(folder_paths.get_annotated_filepath(source_text, output_dir)).expanduser()


def plan_server_file_transfer(source_path, target_dir, target_filename):
    target_dir_text = str(target_dir or "").strip()
    target_filename_text = str(target_filename or "").strip()

    if not target_dir_text:
        raise ValueError("target_dir is required")

    source = resolve_server_file_source_path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Source path is not a file: {source}")

    filename = os.path.basename(target_filename_text) if target_filename_text else source.name
    if not filename:
        raise ValueError("target_filename is required when source_path has no file name")

    return source, Path(target_dir_text).expanduser() / filename


def transfer_server_file(source_path, target_dir, target_filename, move_file=False):
    source, target = plan_server_file_transfer(source_path, target_dir, target_filename)
    target.parent.mkdir(parents=True, exist_ok=True)

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
                "source_path": ("STRING", {
                    "default": "",
                    "tooltip": "Source file path on the server. Empty can use image_source."
                }),
                "target_dir": ("STRING", {"default": "", "tooltip": "Target directory on the server."}),
                "target_filename": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Target file name. Empty uses the source file name. Parent paths "
                        "are ignored."
                    )
                }),
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
                    "tooltip": (
                        "Auto runs during prompt execution. Manual only resolves the source and "
                        "target during execution; the node button then performs that transfer."
                    )
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

    def execute(self, source_path="", target_dir="", target_filename="", move_file=False,
                auto_execute=True, image_source=None, prompt=None, node_id=None):
        try:
            source_path = resolve_image_widget_source_path(prompt or {}, node_id, "image_source")
        except Exception:
            # 连了 image_source 时 source_path 控件是隐藏的，不能回退到里面可能过期的值
            image_source_linked = "image_source" in (prompt or {}).get(str(node_id), {}).get("inputs", {})
            if image_source_linked or (auto_execute and not str(source_path or "").strip()):
                raise

        if not auto_execute:
            # 只算出这次要搬的源和目标，交给节点按钮去执行，执行前可以先看清楚要覆盖哪个文件
            try:
                source, target = plan_server_file_transfer(source_path, target_dir, target_filename)
            except (OSError, ValueError) as e:
                return {"ui": {"transfer_plan": []}, "result": (str(source_path), "", False, f"Not planned: {e}")}
            plan = {"source_path": str(source), "target_dir": str(target.parent), "target_filename": target.name}
            return {
                "ui": {"transfer_plan": [plan]},
                "result": (str(source), str(target), False, "Planned: click Run File Transfer"),
            }

        result = transfer_server_file(source_path, target_dir, target_filename, move_file)
        return (
            result["source_path"],
            result["target_path"],
            result["success"],
            result["message"],
        )


# LocateFileByName 上次定位到的目录，下次先在这里找
LAST_LOCATED_DIR_FILE = Path(folder_paths.get_user_directory()) / "slowargo_locate_file.json"


def load_last_located_dir():
    try:
        state = json.loads(LAST_LOCATED_DIR_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    last_dir = state.get("last_dir") if isinstance(state, dict) else None
    return last_dir if isinstance(last_dir, str) else ""


def save_last_located_dir(directory):
    try:
        LAST_LOCATED_DIR_FILE.write_text(
            json.dumps({"last_dir": str(directory)}, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning(f"[LocateFileByName] failed to save last dir: {e}")


# 至少这么多比例的像素各通道差值都不超过容差，才算同一张图。inpaint 只改局部、
# 上一轮覆盖过的原图也能过；同名同尺寸的别的照片基本过不了
SAME_IMAGE_TOLERANCE = 8
SAME_IMAGE_MIN_FRACTION = 0.5


def matches_reference(path, reference):
    """reference 是 HxWx3 的 int16 像素数组，按 LoadImage 的方式（EXIF 转正、RGB）解码 path 再比较。"""
    height, width = reference.shape[:2]
    try:
        with Image.open(path) as img:
            if img.size not in ((width, height), (height, width)):
                return False
            pixels = np.asarray(ImageOps.exif_transpose(img).convert("RGB"), dtype=np.int16)
    except OSError:
        return False
    if pixels.shape != reference.shape:
        return False
    close = (np.abs(pixels - reference) <= SAME_IMAGE_TOLERANCE).all(axis=2)
    return close.mean() >= SAME_IMAGE_MIN_FRACTION


def locate_file_by_name(filename, search_roots, reference=None):
    """先查上次定位到的目录，再递归搜索 search_roots（";" 分隔），要求恰好一个同名文件。

    reference 给了就只认内容与它一致的图片；ComfyUI 自己的 input/output/temp 目录不搜，那里只有上传的副本。
    """
    def accepted(path):
        return reference is None or matches_reference(path, reference)

    last_dir = load_last_located_dir()
    if last_dir:
        candidate = Path(last_dir) / filename
        if candidate.is_file() and accepted(candidate):
            return candidate

    roots = [r.strip() for r in str(search_roots or "").split(";") if r.strip()]
    if not roots:
        raise FileNotFoundError(f"{filename} is not in the last dir ({last_dir or 'none'}) and search_roots is empty")

    comfy_dirs = {os.path.normcase(os.path.realpath(d)) for d in (
        folder_paths.get_input_directory(), folder_paths.get_output_directory(), folder_paths.get_temp_directory())}
    wanted = os.path.normcase(filename)
    matches, rejected, unreadable = [], [], []
    for root in roots:
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            raise FileNotFoundError(f"Search root does not exist: {root_path}")
        for dir_path, dir_names, file_names in os.walk(root_path, onerror=lambda e: unreadable.append(e.filename)):
            dir_names[:] = [
                d for d in dir_names
                if not os.path.isjunction(os.path.join(dir_path, d))
                and os.path.normcase(os.path.realpath(os.path.join(dir_path, d))) not in comfy_dirs
            ]
            for f in file_names:
                if os.path.normcase(f) == wanted:
                    path = Path(dir_path) / f
                    (matches if accepted(path) else rejected).append(path)

    if unreadable:
        logger.warning(f"[LocateFileByName] skipped unreadable dirs: {'; '.join(unreadable)}")
    if not matches:
        hint = f" ({len(rejected)} same-named file(s) differ from reference)" if rejected else ""
        raise FileNotFoundError(f"{filename} was not found under: {'; '.join(roots)}{hint}")
    if len(matches) > 1:
        raise ValueError(f"{filename} was found in multiple places: {'; '.join(str(m) for m in matches)}")
    return matches[0]


class LocateFileByName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "", "tooltip": "File name to look for. Parent paths are ignored."}),
                "search_roots": ("STRING", {"default": "", "tooltip": (
                    "';'-separated folders searched recursively when the file is not in the last "
                    "located folder. ComfyUI's input/output/temp folders are skipped."
                )}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": (
                    "Only accept an image whose pixels mostly match this one (an inpaint of it still "
                    "matches)."
                )}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("directory", "file_path")
    DESCRIPTION = (
        "Find the single file with this name: the last located folder first, then search_roots. "
        "Fails when nothing or more than one file matches."
    )
    FUNCTION = "execute"
    CATEGORY = "Slowargo"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 结果取决于磁盘上的文件，不能用缓存
        return float("nan")

    def execute(self, filename, search_roots, reference=None):
        filename = os.path.basename(str(filename or "").strip())
        if not filename:
            raise ValueError("filename is required")
        if reference is not None:
            reference = (reference[0, :, :, :3].cpu().numpy() * 255).round().astype(np.int16)
        path = locate_file_by_name(filename, search_roots, reference)
        save_last_located_dir(path.parent)
        return (str(path.parent), str(path))


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
