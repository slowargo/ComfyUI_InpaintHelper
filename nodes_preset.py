"""Widget Preset 节点：把多个节点上的一组 widget 值存成命名预设，一键切换。

功能全部在前端 js/widgetPreset.js 里。这里只是声明节点，让它出现在节点库、
并声明两个随 workflow 保存的 widget。前端会把节点标成 isVirtualNode，
所以它不会被提交给后端，noop 永远不会被调用。设计说明见
reports/WIDGET_PRESET_NODE_DESIGN.md。
"""


class WidgetPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filter": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "One regex per line, matched against 'node title/widget name' "
                        "(partial, case-insensitive). Lines starting with '-' exclude, lines "
                        "starting with '#' are comments. A line is appended automatically when "
                        "a node is connected."
                    ),
                }),
                # JSON of all presets, managed by the frontend and hidden from the node body
                "presets": ("STRING", {"default": ""}),
            },
            "optional": {},
        }

    RETURN_TYPES = ()
    DESCRIPTION = (
        "Connect any nodes to the inputs, pick widgets with regex rules, then save their current "
        "values as named presets. Each preset gets a button that writes the values back; the "
        "preset matching the current values is highlighted."
    )
    FUNCTION = "noop"
    CATEGORY = "Slowargo"

    def noop(self, filter, presets):
        return ()
