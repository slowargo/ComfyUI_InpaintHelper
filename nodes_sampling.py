"""采样循环内的 inpaint 偏色抑制：在每一步的 x0 预测上动手，而不是出图后补救。

低 denoise inpaint 的偏色来自去噪器自身：每一步的 x0 预测都会把重绘区往模型偏好的
影调/白平衡上拉，逐步累加。出图后的色彩校正（见 nodes_color.py）只能事后平移均值，
后续步骤生成的细节已经建立在偏掉的颜色上。本模块把校正挪进采样循环：每一步拿到 x0
后先把遮罩内的低频拉回原图低频（只按遮罩内像素求局部平均），再交给采样器走下一步，
后续步骤就在正确的颜色上细化。

实测（Krea2 Turbo，denoise 0.32）遮罩内偏色与遮罩外测得的模型偏差方向不同——它跟
内容和提示词相关，不是全局常量，所以没法用遮罩外的实测值扣除，只能锚定低频。这等于
在循环内编码「重绘不改变 blur_sigma 以上尺度的色彩」。偏色主要在后几步落定，只锚定前几步
会被后续步骤重新带偏，所以每一步都锚定；需要大面积改色时降低 strength 或调大 blur_sigma。
"""

import logging

import torch
import torch.nn.functional as F

import comfy.sampler_helpers

logger = logging.getLogger(__name__)

LOG_PREFIX = "[X0DriftGuard]"
# 只用于日志：遮罩值不超过 KNOWN_THRESHOLD 的 latent 像素视为未重绘，拿来观察模型在该 sigma 下的偏差；
# 不低于 INSIDE_THRESHOLD 的视为重绘核心区
KNOWN_THRESHOLD = 0.05
INSIDE_THRESHOLD = 0.95


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
                "mode": (["lowfreq_anchor", "measure"], {
                    "default": "lowfreq_anchor",
                    "tooltip": (
                        "lowfreq_anchor: pull the masked region's low frequencies back to the "
                        "original at every step. measure: leave sampling untouched and only log "
                        "the per-step drift."
                    ),
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 leaves sampling untouched, 1 fully anchors the low frequencies.",
                }),
                "blur_sigma": ("FLOAT", {
                    "default": 12.0, "min": 1.0, "max": 64.0, "step": 0.5,
                    "tooltip": (
                        "Gaussian sigma in latent pixels (x8 for image pixels). Colors above this "
                        "scale are anchored; smaller values hold color tighter but also constrain "
                        "more detail. Capped at a quarter of the latent's short side."
                    ),
                }),
                "log": ("BOOLEAN", {"default": False, "tooltip": (
                    "Log per-step drift (approximate RGB of batch 0) to the console. Always on in "
                    "measure mode."
                )}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": (
                    "The inpaint mask. Defaults to the latent's noise mask (from SetLatentNoiseMask)."
                )}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Slowargo"
    DESCRIPTION = (
        "Anchor the inpaint region's low frequencies to the original inside the sampling loop to "
        "stop color drift at its source. Place it after other post-CFG patches, which would "
        "otherwise run after it and can reintroduce the drift."
    )

    def patch(self, model, latent, mode, strength, blur_sigma, log, mask=None):
        if mask is None:
            mask = latent.get("noise_mask")
        if mask is None:
            raise ValueError("InpaintX0DriftGuard needs a mask: connect one or use a latent from SetLatentNoiseMask.")
        orig_raw = latent["samples"]
        # 采样器不对全零 latent 做 process_in（见 comfy.samplers），这里若照做会锚定到一个不存在的色调
        if torch.count_nonzero(orig_raw) == 0:
            raise ValueError("InpaintX0DriftGuard got an empty latent; connect the VAE-encoded original.")

        m = model.clone()
        latent_format = m.get_model_object("latent_format")
        anchor = mode == "lowfreq_anchor" and strength > 0
        log = log or mode == "measure"
        rgb_factors = None
        if latent_format.latent_rgb_factors is not None and latent_format.latent_rgb_factors_reshape is None:
            rgb_factors = torch.tensor(latent_format.latent_rgb_factors)
            if rgb_factors.shape[0] != orig_raw.shape[1]:
                rgb_factors = None
        cache = {}

        def fmt(r_bc):
            # 残差不带偏置项；latent_rgb_factors 输出约在 [-1,1]，乘 127.5 换算成 0..255 量程。
            # 没有可用因子的模型退回逐通道均值的 L2 范数
            if rgb_factors is None:
                return f"|c|={float(r_bc[0].norm()):.4f}"
            rgb = r_bc @ rgb_factors.to(r_bc.device, r_bc.dtype) * 127.5
            return "[" + ", ".join(f"{v:+.2f}" for v in rgb[0].tolist()) + "]"

        def post_cfg(args):
            den = args["denoised"]
            key = (den.shape, den.device)
            if key not in cache:
                if orig_raw.shape != den.shape:
                    raise ValueError(
                        f"InpaintX0DriftGuard: connected latent is {tuple(orig_raw.shape)} but the sampler runs "
                        f"{tuple(den.shape)}; use this patched model only with the sampler that takes that latent."
                    )
                orig = latent_format.process_in(orig_raw.to(den.device, torch.float32))
                msk = comfy.sampler_helpers.prepare_mask(mask, den.shape, den.device).float()[:, :1]
                sigma_px = min(blur_sigma, min(den.shape[-2:]) / 4)
                cache.clear()
                cache[key] = (orig, msk, sigma_px, _blur(msk, sigma_px).clamp(min=1e-3))
            orig, msk, sigma_px, msk_low = cache[key]
            x0 = den.float()
            # 归一化卷积：只用遮罩内像素求局部平均偏差。直接模糊整张图时，细长或细小的遮罩会被周围
            # 未重绘的像素稀释，修正量只剩一部分
            corr = strength * msk * _blur(msk * (orig - x0), sigma_px) / msk_low if anchor else None

            if log:
                resid = x0 - orig
                inside = (msk >= INSIDE_THRESHOLD).float()
                known = (msk <= KNOWN_THRESHOLD).float()
                corr_msg = fmt(_weighted_mean(corr, inside)) if anchor else "off"
                logger.info(f"{LOG_PREFIX} sigma={float(args['sigma'].flatten()[0]):.4f} "
                            f"known_resid={fmt(_weighted_mean(resid, known))} "
                            f"inside_resid={fmt(_weighted_mean(resid, inside))} inside_corr={corr_msg}")

            if not anchor:
                return den
            return (x0 + corr).to(den.dtype)

        m.set_model_sampler_post_cfg_function(post_cfg)
        return (m,)
