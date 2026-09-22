"""字符串记忆节点，以及前端读写它历史记录的三条 API 路由。

路由是 import 时注册的副作用，所以 __init__.py 里对本模块的导入不能因为
「看起来没用到 XXX_api」而删掉。
"""

import logging
import json
import os
from aiohttp import web
import folder_paths
from server import PromptServer
import re

logger = logging.getLogger(__name__)


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


@PromptServer.instance.routes.get("/slowargo_api/get_string_history")
async def get_string_history_api(request):
    store_file = request.query.get("store_file", "")
    _, stored_entries = RememberStrings.read_stored_strings(store_file)
    return web.json_response({"entries": stored_entries})


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
