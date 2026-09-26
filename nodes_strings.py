"""字符串记忆节点，前端读写它历史记录的三条 API 路由，以及入队时记录字符串的 on_prompt 钩子。

路由和钩子都是 import 时注册的副作用，所以 __init__.py 里对本模块的导入不能因为
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
        string = string.strip()

        if not string: return ("",)

        RememberStrings.remember(string, store_file, max_entries)
        return (string,)

    @staticmethod
    def remember(string, store_file, max_entries):
        # 读入已记忆列表，已记忆则移到顶部，否则添加到顶部，记录条数不超过 max_entries，之后保存回文件。
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

        # 3. 淘汰逻辑（只淘汰，不重排）
        # 文件里存的始终是「最近使用」顺序，Pinned 置顶交给读取端做视图变换，由 View History
        # 弹窗的 Pin to top 开关控制。条目没有时间戳字段，顺序就是唯一的时间信息，写入时按
        # pinned 重排会把它永久破坏：pin 再 unpin 之后，最旧的条目会留在列表最前面。
        # 我们需要保留所有 pinned=True 的，以及排在前面的非 pinned 条目，总数不超过 max_entries
        pinned_count = sum(1 for item in stored_entries if item.get("pinned"))

        # 计算还能容纳多少个非置顶条目
        # 即使 pinned 很多，我们也至少保证总数逻辑或优先保证 pinned
        allowed_unpinned_count = max(0, max_entries - pinned_count)

        # 保序遍历：pinned 全留，非 pinned 只留最靠前（即最近使用）的若干条
        final_entries = []
        kept_unpinned = 0
        for item in stored_entries:
            if item.get("pinned"):
                final_entries.append(item)
            elif kept_unpinned < allowed_unpinned_count:
                final_entries.append(item)
                kept_unpinned += 1

        # 保存文件
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(final_entries, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"[RememberStrings] Save error: {e}")

    @staticmethod
    def read_stored_strings(store_file):
        store_file = store_file.strip()
        # 使用正则表达式解析 store_file 格式，支持类似 "filename.json[output]" 的格式
        # 忽略文件名和 [ 之间的空格
        match = re.match(r"^(.+?)\s*\[([^\]]+)\]$", store_file)
        if match:
            file_name = match.group(1).strip()  # 提取文件名部分并去除空格
            folder_type = match.group(2).lower()  # 提取 [] 中的内容并转为小写

            base_dir = (folder_paths.get_input_directory() if folder_type == "input"
                        else folder_paths.get_output_directory())

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


# 入队时就记录：默认的 RAM 缓存会保留多次运行的结果，从历史里选回一个旧字符串时节点命中缓存、
# remember_strings 根本不执行，顺序也就不会更新。用 IS_CHANGED 强制重跑又会连带下游（CLIP 编码等）
# 失效，所以改在这里处理。只处理三个输入都是字面值的节点；string 来自连线的情况仍靠节点执行时记录。
# 这一步在校验之前，所以校验失败的 prompt、以及没接到任何输出的 RememberStrings 节点也会被记录。
def remember_strings_on_prompt(json_data):
    for node in json_data.get("prompt", {}).values():
        if node.get("class_type") != "RememberStrings":
            continue
        inputs = node.get("inputs", {})
        string, store_file = inputs.get("string"), inputs.get("store_file")
        max_entries = inputs.get("max_entries", 10)
        if isinstance(string, str) and isinstance(store_file, str) and isinstance(max_entries, int):
            string = string.strip()
            if string:
                RememberStrings.remember(string, store_file, max_entries)
    return json_data


PromptServer.instance.add_on_prompt_handler(remember_strings_on_prompt)


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

    # 不重排：Pin 只翻转一个布尔值，文件顺序保持「最近使用」不变，置顶由前端视图变换负责。
    # 这样 pin / unpin 是完全可逆的，不会像重排那样丢掉条目的新旧信息。

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
