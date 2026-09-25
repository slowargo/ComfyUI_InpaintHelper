"""采样循环内的 inpaint 偏色抑制：在每一步的 x0 预测上动手，而不是出图后补救。

低 denoise inpaint 的偏色来自去噪器自身：每一步的 x0 预测都会把重绘区往模型偏好的
影调/白平衡上拉，逐步累加。出图后的色彩校正（见 nodes_color.py）只能事后平移均值，
后续步骤生成的细节已经建立在偏掉的颜色上。本模块把校正挪进采样循环：每一步拿到 x0
后先把遮罩内的低频拉回原图低频，再交给采样器走下一步，后续步骤就在正确的颜色上细化。

实测（Krea2 Turbo，denoise 0.32）遮罩内偏色与遮罩外测得的模型偏差方向不同——它跟
内容和提示词相关，不是全局常量，所以没法用遮罩外的实测值扣除，只能锚定低频。这等于
在循环内编码「重绘不改变 blur_sigma 以上尺度的色彩」。偏色主要在后几步落定，只锚定前几步
会被后续步骤重新带偏，所以每一步都锚定；需要大面积改色时降低 strength 或调大 blur_sigma。
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

LOG_PREFIX = "[X0DriftGuard]"
# 只用于 measure 日志：遮罩值不超过它的 latent 像素视为未重绘，拿来观察模型在该 sigma 下的偏差
KNOWN_THRESHOLD = 0.05


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(round(sigma * 3)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """对最后两维做可分离高斯模糊，其余维度原样保留。边界用 replicate 填充。"""
    shape = x.shape
    h, w = shape[-2], shape[-1]
    flat = x.reshape(-1, 1, h, w)
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = k.numel() // 2
    flat = F.conv2d(F.pad(flat, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    flat = F.conv2d(F.pad(flat, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return flat.reshape(shape)


def _mask_like(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 MASK 缩放到 ref 的空间尺寸，并整形成可与 ref 广播的 [B,1,(1...),H,W]。"""
    m = mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    h, w = ref.shape[-2], ref.shape[-1]
    m = F.interpolate(m.unsqueeze(1).to(ref.device), size=(h, w), mode="area")
    m = m.view(m.shape[0], 1, *([1] * (ref.ndim - 4)), h, w)
    if m.shape[0] != ref.shape[0]:
        m = m.expand(ref.shape[0], *m.shape[1:])
    return m.clamp(0, 1)


def _weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """按空间权重求每个 (batch, channel) 的均值，返回 [B,C]。"""
    dims = tuple(range(2, x.ndim))
    return (x * w).sum(dims) / w.sum(dims).clamp(min=1e-6)


class InpaintX0DriftGuard:
    """在 sampler_post_cfg_function 里把重绘区 x0 的低频锚定到原图，从源头抑制 inpaint 偏色。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT", {"tooltip": (
                    "The VAE-encoded original (the same latent fed to the sampler)."
                )}),
                "mask": ("MASK", {"tooltip": (
                    "The inpaint mask (the one given to SetLatentNoiseMask)."
                )}),
                "mode": (["lowfreq_anchor", "measure"], {
                    "default": "lowfreq_anchor",
                    "tooltip": (
                        "lowfreq_anchor: pull the masked region's low frequencies back to the "
                        "original at every step. measure: leave sampling untouched and only log "
                        "the per-step drift."
                    ),
                }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blur_sigma": ("FLOAT", {
                    "default": 12.0, "min": 1.0, "max": 64.0, "step": 0.5,
                    "tooltip": (
                        "Gaussian sigma in latent pixels (x8 for image pixels). Colors above this "
                        "scale are anchored; smaller values hold color tighter but also constrain "
                        "more detail."
                    ),
                }),
                "log": ("BOOLEAN", {"default": False, "tooltip": (
                    "Log per-step drift (approximate RGB) to the console."
                )}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Slowargo"
    DESCRIPTION = (
        "Anchor the inpaint region's low frequencies to the original inside the sampling loop to "
        "stop color drift at its source."
    )

    def patch(self, model, latent, mask, mode, strength, blur_sigma, log):
        m = model.clone()
        latent_format = m.get_model_object("latent_format")
        orig_raw = latent["samples"]
        rgb_factors = latent_format.latent_rgb_factors
        cache = {}

        def to_rgb(r_bc):
            # 残差不带偏置项；latent_rgb_factors 输出约在 [-1,1]，乘 127.5 换算成 0..255 量程
            f = torch.tensor(rgb_factors, device=r_bc.device, dtype=r_bc.dtype)
            return "[" + ", ".join(f"{v:+.2f}" for v in (r_bc @ f * 127.5)[0].tolist()) + "]"

        def post_cfg(args):
            den = args["denoised"]
            key = (den.shape, den.device)
            if key not in cache:
                orig = latent_format.process_in(orig_raw.to(den.device, torch.float32)).reshape(den.shape)
                cache.clear()
                cache[key] = (orig, _mask_like(mask, den), _blur(orig, blur_sigma))
            orig, msk, orig_low = cache[key]
            sigma = float(args["sigma"].flatten()[0])
            x0 = den.float()

            anchor = mode == "lowfreq_anchor"
            corr = strength * msk * (orig_low - _blur(x0, blur_sigma)) if anchor else None

            if log:
                resid = x0 - orig
                inside = (msk >= 0.95).float()
                known = (msk <= KNOWN_THRESHOLD).float()
                corr_rgb = to_rgb(_weighted_mean(corr, inside)) if anchor else "off"
                logger.info(f"{LOG_PREFIX} sigma={sigma:.4f} known_resid_rgb={to_rgb(_weighted_mean(resid, known))} "
                            f"inside_resid_rgb={to_rgb(_weighted_mean(resid, inside))} inside_corr_rgb={corr_rgb}")

            if corr is None:
                return den
            return (x0 + corr).to(den.dtype)

        m.set_model_sampler_post_cfg_function(post_cfg)
        return (m,)
