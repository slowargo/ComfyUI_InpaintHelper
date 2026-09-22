"""零散的小工具节点：浮点数开关/选择器、刷新与运行触发器、历史清理、SSIM 比较。

这些彼此无依赖，也不依赖本包其他模块，凑在一起只是为了不给每个几十行的节点
单开一个文件。ImageSimilaritySSIM 严格说不属于「工具」，如果以后图像比较这类
节点变多，应该单独成模块。
"""

from decimal import Decimal, InvalidOperation
from comfy_api.latest import io
from server import PromptServer
import torch
import torch.nn.functional as F


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
