"""Lab 色彩校正节点：遮罩外用真实测量，遮罩内用假设，按遮罩值在两者间过渡。

从 __init__.py 拆出。这一组只依赖 torch 与 kornia，不碰 ComfyUI 的服务端、
文件系统或前端，所以能独立测试——把本文件的 _resize_bchw_to 到文件末尾
exec 出来就能跑，不需要 ComfyUI 运行时。
"""

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _resize_bchw_to(tensor_bchw: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """把 [B,C,H,W] 缩放到目标尺寸；尺寸已一致时原样返回，不产生拷贝。"""
    if tensor_bchw.shape[-2:] == size:
        return tensor_bchw
    return F.interpolate(tensor_bchw, size=size, mode="bilinear", align_corners=False)


def _broadcast_batch(tensor: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    """batch 为 1 时零拷贝展开到目标 batch；既不为 1 也不匹配则报错。"""
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(f"Batch size mismatch: image batch={batch}, {name} batch={tensor.shape[0]}")


def _normalize_mask(mask: torch.Tensor, height: int, width: int, device) -> torch.Tensor:
    """把常见形状的 MASK 归一成 [B,H,W]、值域 [0,1]、对齐到目标尺寸。

    不做 batch 广播：调用方通常要先做形态学再广播，否则会把展开后的副本实体化。
    """
    msk = mask.float().to(device)
    if msk.ndim == 2:
        msk = msk.unsqueeze(0)
    elif msk.ndim == 4:
        # [B,1,H,W] 或 [B,H,W,1] 都归一到 [B,H,W]
        if msk.shape[1] == 1:
            msk = msk.squeeze(1)
        elif msk.shape[-1] == 1:
            msk = msk.squeeze(-1)
    if msk.ndim != 3:
        raise ValueError(f"Expected MASK tensor with 3 dims [B,H,W], got shape: {tuple(mask.shape)}")
    if msk.shape[1:3] != (height, width):
        msk = _resize_bchw_to(msk.unsqueeze(1), (height, width)).squeeze(1)
    return msk.clamp(0.0, 1.0)


class MaskedColorMatch:
    """基于遮罩外区域的线性色彩回归，校正 inpaint 的 VAE 重建偏差。

    VAE encode/decode 往返带有系统性偏差，与内容无关、全图一致，迭代式修补下会逐轮累积。

    本节点只用遮罩外（内容理论上未被改动）的像素拟合逐通道的 reference = k * image + c，
    再把该变换施加到整张图上。

    作用范围仅限上述 VAE 分量。重绘区内采样器自身还会叠加一层偏移，那部分在遮罩外没有
    任何可观测样本，本节点无法、也不试图消除它——强行把重绘区均值拉回原图均值会破坏
    "重绘本来就该改变颜色"的正常情况。见 InpaintRegionColorFix。
    """

    # Guard rails for the per-channel least-squares fit.
    # 样本区纹理太弱时 gain 会被 regression dilution 系统性压低（x 自带 VAE 重建噪声，
    # k → var_true/(var_true+var_noise)），把这种被低估的 k 施加到全图等于压对比度，
    # 危害远大于它要修的那点偏移。std 低于阈值就降级为纯 offset。
    MIN_SAMPLE_STD = 0.12          # ≈ 30/255
    MIN_SAMPLE_RATIO = 0.002       # 样本数下限取 max(MIN_SAMPLES, 像素总数 * 该比例)
    MIN_SAMPLES = 1024
    GAIN_LIMITS = (0.9, 1.1)
    EPS = 1e-12

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The VAE-decoded inpaint result to correct."}),
                "reference": ("IMAGE", {"tooltip": "The original image before VAE encode, same framing as image."}),
                "mode": (["offset_only", "gain_offset"], {
                    "default": "offset_only",
                    "tooltip": "offset_only fits a per-channel constant shift (recommended, matches the measured VAE bias). gain_offset also fits a slope, which needs a well-textured sample region to be reliable."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blends the correction: 0 leaves the image untouched, 1 applies the full fit."
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Only pixels whose mask value is at or below this are used to fit the correction. Keep it low so that repainted pixels never enter the fit."
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "The inpaint mask. Without it the fit uses the whole image, which lets repainted content distort the result."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "match_color"
    CATEGORY = "Slowargo"
    DESCRIPTION = "Remove the VAE roundtrip color drift from an inpaint result by fitting a per-channel linear correction on the pixels outside the mask, then applying it to the whole image."

    def _sample_region(self, mask, batch, height, width, mask_threshold, device):
        """遮罩外像素的选取。返回 [B,H,W] 的 bool。"""
        if mask is None:
            logger.warning(
                "[MaskedColorMatch] no mask connected, fitting on the whole image - "
                "repainted content will distort the correction"
            )
            return torch.ones((batch, height, width), dtype=torch.bool, device=device)

        msk = _normalize_mask(mask, height, width, device)
        # mask 缩小时 bilinear 是点采样，细羽化带会被跳过而让重绘像素混进样本池，
        # 用 3x3 max 保守膨胀挡掉这种漏采。注意它挡不住 VAE 解码感受野造成的
        # 边界渗透（尺度是 8px 量级），那部分只占样本池极小比例，对均值影响可忽略。
        msk = F.max_pool2d(msk.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        return _broadcast_batch(msk, batch, "mask") <= mask_threshold

    def match_color(self, image, reference, mode="offset_only", strength=1.0, mask_threshold=0.05, mask=None):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor with 4 dims [B,H,W,C], got shape: {tuple(image.shape)}")

        img = image.float()
        batch, height, width, channels = img.shape

        ref = reference.float().to(img.device)
        if ref.ndim != 4:
            raise ValueError(f"Expected reference IMAGE with 4 dims [B,H,W,C], got shape: {tuple(ref.shape)}")
        if ref.shape[-1] != channels:
            raise ValueError(f"Channel mismatch: image has {channels}, reference has {ref.shape[-1]}")
        if ref.shape[1:3] != (height, width):
            logger.warning(
                f"[MaskedColorMatch] reference size {tuple(ref.shape[1:3])} != image size {(height, width)}, "
                f"resizing reference - the interpolation blur it adds will bias the fit"
            )
            ref = _resize_bchw_to(ref.permute(0, 3, 1, 2), (height, width)).permute(0, 2, 3, 1)
        ref = _broadcast_batch(ref, batch, "reference")

        sample_region = self._sample_region(mask, batch, height, width, mask_threshold, img.device)
        min_samples = max(self.MIN_SAMPLES, int(height * width * self.MIN_SAMPLE_RATIO))
        gain_lo, gain_hi = self.GAIN_LIMITS

        corrected = torch.empty_like(img)
        report_lines = [f"mode={mode} strength={strength:.2f} mask_threshold={mask_threshold:.2f}"]

        for b in range(batch):
            weight = sample_region[b].to(img.dtype)
            sample_count = int(weight.sum().item())
            if sample_count < min_samples:
                logger.warning(
                    f"[MaskedColorMatch] batch {b}: only {sample_count} usable pixels "
                    f"(need {min_samples}, mask_threshold={mask_threshold}), skipping correction"
                )
                corrected[b] = img[b]
                report_lines.append(f"[{b}] SKIPPED n={sample_count} < {min_samples}")
                continue

            # 逐通道加权统计，一次算完所有通道，避免 per-channel 的布尔 gather
            wc = weight.unsqueeze(-1)
            x, y = img[b], ref[b]
            x_mean = (x * wc).sum(dim=(0, 1)) / sample_count
            y_mean = (y * wc).sum(dim=(0, 1)) / sample_count
            x_dev, y_dev = x - x_mean, y - y_mean
            x_var = ((x_dev * x_dev) * wc).sum(dim=(0, 1)) / sample_count
            y_var = ((y_dev * y_dev) * wc).sum(dim=(0, 1)) / sample_count
            covariance = ((x_dev * y_dev) * wc).sum(dim=(0, 1)) / sample_count

            # 上游任何一个坏像素都会让均值/方差变成 NaN，再逐像素相加就是整图报废
            if not bool(torch.isfinite(torch.stack([x_mean, y_mean, x_var, y_var, covariance])).all()):
                logger.warning(
                    f"[MaskedColorMatch] batch {b}: non-finite statistics in the sample region "
                    f"(NaN/Inf in image or reference), skipping correction"
                )
                corrected[b] = img[b]
                report_lines.append(f"[{b}] SKIPPED non-finite statistics")
                continue

            x_std = x_var.sqrt()
            unit_gain = torch.ones_like(x_std)
            if mode == "gain_offset":
                fit_ok = x_std >= self.MIN_SAMPLE_STD
                raw_gain = covariance / x_var.clamp_min(self.EPS)
                bounded = raw_gain.clamp(gain_lo, gain_hi)
                gain = torch.where(fit_ok, bounded, unit_gain)
                was_clamped = fit_ok & (bounded != raw_gain)
                if not bool(fit_ok.all()):
                    logger.warning(
                        f"[MaskedColorMatch] batch {b}: sample region too flat "
                        f"(std={[round(v * 255.0, 2) for v in x_std.tolist()]}/255 < "
                        f"{self.MIN_SAMPLE_STD * 255.0:.0f}/255), falling back to offset_only on those channels"
                    )
                if bool(was_clamped.any()):
                    logger.warning(
                        f"[MaskedColorMatch] batch {b}: gain clamped to {self.GAIN_LIMITS} "
                        f"(raw={[round(v, 4) for v in raw_gain.tolist()]}), the fit is not trustworthy"
                    )
            else:
                gain = unit_gain
                was_clamped = torch.zeros_like(unit_gain, dtype=torch.bool)

            offset = y_mean - gain * x_mean
            # strength 插值：out = (1-s)*img + s*(k*img+c)
            gain_eff = 1.0 + (gain - 1.0) * strength
            offset_eff = offset * strength
            # clamp 会削掉本该提亮的高光，8bit 管线下无解，多轮迭代后高光会略微压平
            corrected[b] = (x * gain_eff + offset_eff).clamp_(0.0, 1.0)

            # 诊断量：均值处的修正是拟合的恒等式（直线必过样本均值点），说明不了任何问题，
            # 所以额外报 ±2σ 两端的修正量——gain 一旦失真，这两个数会立刻劈叉。
            lo = (x_mean - 2.0 * x_std).clamp(0.0, 1.0)
            hi = (x_mean + 2.0 * x_std).clamp(0.0, 1.0)
            stats = torch.stack([
                gain,
                offset * 255.0,
                x_std * 255.0,
                (covariance * covariance) / (x_var * y_var).clamp_min(self.EPS),
                (gain_eff * lo + offset_eff - lo) * 255.0,
                (gain_eff * hi + offset_eff - hi) * 255.0,
                was_clamped.to(x_std.dtype),
            ]).t().tolist()
            channel_reports = [
                f"k={k:.5f} c={c:+.3f} std={s:.1f} R2={r2:.4f} @-2s={s_lo:+.2f} @+2s={s_hi:+.2f}"
                + (" CLAMPED" if clamped else "")
                for k, c, s, r2, s_lo, s_hi, clamped in stats
            ]
            report_lines.append(f"[{b}] n={sample_count} " + " | ".join(channel_reports))

        report = "\n".join(report_lines)
        logger.debug(f"[MaskedColorMatch]\n{report}")
        return (corrected, report)


class InpaintRegionColorFix:
    """把重绘区的色彩基准拉回原图，按遮罩值在「遮罩外」和「遮罩内」两个基准之间过渡。

    与 MaskedColorMatch 的本质区别：遮罩外有 ground truth（内容未改动，原图就是答案），
    拟合出来的是测量值；遮罩内没有 ground truth，任何校正都是在编码一条假设——
    「重绘不应该改变该区域的平均亮度（components 选 luminance_and_chroma 时还包括平均色度）」。

    默认只校正亮度：亮度损失跨内容、跨采样器都表现为同向的系统性偏移，而色度偏移的方向
    随内容变化，与真实重绘意图分不开，一并拉回就会撤销重绘本该带来的颜色变化。

    核心区基准默认取均值：只做细化时它更准，而它「跟随内容变化」的风险已经由 max_excess
    封顶。区域可能被大面积重绘时改用中位数，它不会被那部分带跑。

    高 denoise 的内容替换场景上述假设不成立（重绘本来就该改变颜色）。那种情况下把
    inside_source 切到 manual：改用手工标定的常量偏移，完全不看重绘区内容，于是假设从
    「内容属性」变成「管道属性」——代价是换一组 模型/采样器/步数/denoise 就要重新标定。

    校正量按遮罩值插值，与偏移量随遮罩强度递增的实际趋势同向，羽化带不会被过校正，
    也不会引入新的接缝。
    """

    # 遮罩外总体是整张图的大头，下限跟 MaskedColorMatch 对齐。
    MIN_OUTSIDE_SAMPLES = 1024
    MIN_OUTSIDE_RATIO = 0.002
    # 核心区下限只做数值保护，不做统计准入。退回遮罩外基准不是中性的「更安全」，
    # 它是一个已知偏了整整一个 drift 量的选择；而局部估计的样本外误差远小于此，
    # 即使在很小的区域上也是如此。区域越小，局部测量相对全局常数的优势反而越大，
    # 所以这里不该有一个会把小重绘挡在外面的门槛。
    MIN_INSIDE_SAMPLES = 64
    # estimator=clip 迭代时，某通道截断后剩余像素少于此值就冻结该通道。与
    # MIN_INSIDE_SAMPLES 同值纯属巧合：那个管核心区总体准入，这个管「还剩多少
    # 样本才敢再走一步」。
    MIN_CLIP_INLIERS = 64

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The VAE-decoded inpaint result to correct."}),
                "reference": ("IMAGE", {"tooltip": "The original image before VAE encode, same framing as image."}),
                "mask": ("MASK", {"tooltip": "The inpaint mask. Its feather is used as the blend ramp between the outside and inside baselines."}),
                "components": (["luminance", "luminance_and_chroma"], {
                    "default": "luminance",
                    "tooltip": "luminance shifts L* only, so hue and saturation are left alone except where the shift pushes a pixel out of gamut - the safe default, since L* drift is the only part that measures as systematic. luminance_and_chroma also pulls a*/b* back, which fixes colour casts but undoes intentional colour changes in the repainted area."
                }),
                "inside_source": (["fit", "manual"], {
                    "default": "fit",
                    "tooltip": "fit measures the repainted core's baseline from the image, which assumes repainting should not change the region's mean. manual ignores the core entirely and uses inside_offset_l as the baseline shift, so it never fights an intentional colour change - but it has to be calibrated per model/sampler/steps/denoise."
                }),
                "inside_offset_l": ("FLOAT", {
                    "default": 0.0, "min": -50.0, "max": 50.0, "step": 0.1,
                    "tooltip": "Baseline shift for the repainted core, in Lab L* units; positive brightens. Added on top of the measured shift in fit mode, used as the whole shift in manual mode."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blends the correction: 0 returns the input untouched, 1 applies the full baseline shift."
                }),
                "outside_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask values at or below this define the outside population, whose baseline shift is a real measurement. Must stay below inside_threshold."
                }),
                "inside_threshold": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask values at or above this define the repainted core, whose baseline shift is an assumption. A mask whose peak never reaches this value gets no inside correction - which is correct, since such a mask barely repaints anything."
                }),
                "max_excess": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Cap, in Lab units, on how far the repainted core's baseline may sit from the outside one; 0 disables the cap. Drift alone stays well inside it, so the cap only binds when the repaint really changed the content - it then limits the damage instead of letting the correction undo that change. For heavy content replacement, inside_source=manual is still better."
                }),
                "estimator": (["mean", "median", "clip"], {
                    "default": "mean",
                    "tooltip": "How the repainted core's baseline is summarised. mean is marginally more accurate when the repaint only refined detail, but it tracks any real content change one-for-one and is unbounded. median gives that up for a small, fixed cost and stays close to the drift even when part of the region was genuinely repainted. clip assumes the repaint only altered part of the region: it locks onto the dominant offset and discards pixels that moved far from it, controlled by clip_k. Pick it when a minority of the core was really repainted and the rest only drifted. If instead the whole region was reworked there is no untouched majority to lock onto, and it latches onto whichever mode dominates and over-corrects past mean - so this is a scenario switch, not a safer default. Only affects the core - the outside baseline always uses the mean, where there is no content change to be robust against."
                }),
                "clip_k": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "For estimator=clip: how far a core pixel may sit from the current offset estimate before it stops counting towards it, in units of the outside population's robust dispersion. That dispersion is small, so this is the width of a narrow mode-seeking window rather than an outlier threshold - the default already leaves most of the core out, and raising k widens the window without converging on the plain mean. Smaller rejects harder. Taking the unit from the outside keeps it meaningful across images of similar detail, but a core whose texture differs a lot from its surroundings will still need a different k."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "fix_region"
    CATEGORY = "Slowargo"
    DESCRIPTION = "Pull the repainted region's colour baseline back to the original in Lab space, ramping the correction by mask value so the feathered edge stays seamless. Corrects the sampler-induced drift that MaskedColorMatch cannot see."

    @staticmethod
    def _masked_mean(lab_hwc: torch.Tensor, selection: torch.Tensor, count: int) -> torch.Tensor:
        return (lab_hwc * selection.unsqueeze(-1)).sum(dim=(0, 1)) / count

    @staticmethod
    def _masked_median(diff_hwc: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
        # 必须是「逐像素差值的中位数」，不能是「两个中位数之差」——后者不再是配对
        # 估计量，内容方差不会抵消。
        return diff_hwc[selection].median(dim=0).values

    @classmethod
    def _masked_clip(cls, diff_hwc: torch.Tensor, selection: torch.Tensor,
                     out_diff: torch.Tensor, k: float, iters: int = 200):
        """中心自由的迭代截断：找核心区的主导偏移，把离它太远的像素当成重绘内容剔除。

        中心必须是自由参数。用「接近遮罩外基准」来筛像素会把待估的量当成筛选条件，
        估计量被结构性地拉回遮罩外值，无论真实偏移是多少都得到零——看着很稳，实则退化。

        尺度取自遮罩外的离散度。那里内容没有改动，所以离散度只反映重建误差而非重绘，
        用它当单位比凭空给一个 Lab 绝对值合理；但它随局部细节密度变化，核心区的纹理
        统计与遮罩外不同时，同一个 k 的实际松紧程度也会跟着变。

        迭代是单调慢漂而非震荡，轮数上限必须给够：截断过早等于让「走了几步」成为
        估计量定义的一部分，同一张图换个上限就是另一个答案。

        返回 (center, info)。info 里带每通道保留像素数、实际轮数、以及哪些通道在第一
        轮就被冻结（即该通道实际退化成了中位数），供 report 暴露出来——否则用户无法
        知道自己选的 clip 到底跑成了什么。
        """
        din = diff_hwc[selection]
        center = din.median(dim=0).values
        # 稳健 sigma：MAD 换算到正态等效标准差，不被遮罩外的少量坏像素带跑
        mad = (out_diff - out_diff.median(dim=0).values).abs().median(dim=0).values
        scale = (1.4826 * mad * k).clamp(min=1e-6)

        active = torch.ones_like(center, dtype=torch.bool)
        n_keep = torch.full_like(center, float(din.shape[0]))
        froze_at_once = torch.zeros_like(active)
        used = 0
        for it in range(iters):
            keep = (din - center).abs() <= scale
            cnt = keep.sum(dim=0)
            # 逐通道独立冻结，不是任一通道不行就整体放弃：默认 components=luminance
            # 下 a*/b* 随后会被 keep 向量归零，让一个注定被丢弃的通道有权取消 L* 的
            # 精修是错的。而 clip 的目标场景（少数区域被真重绘）往往正是色相替换，
            # a*/b* 分布宽、L* 才是要估的量。
            newly_frozen = active & (cnt < cls.MIN_CLIP_INLIERS)
            if it == 0:
                froze_at_once = newly_frozen.clone()
            # 在冻结判定之后、break 之前记录：此时 active 仍含刚冻结的通道，report
            # 里才看得到是「剩太少」才停的。全通道同时冻结时也不会漏记。
            n_keep = torch.where(active, cnt.to(n_keep.dtype), n_keep)
            active = active & ~newly_frozen
            if not bool(active.any()):
                break
            nxt = torch.where(active, (din * keep).sum(dim=0) / cnt.clamp(min=1), center)
            moved = (nxt - center).abs() >= 1e-4
            center = nxt
            used = it + 1
            if not bool((active & moved).any()):
                break
        return center, {"n_keep": n_keep, "iters": used, "froze": froze_at_once}

    @staticmethod
    def _fmt_delta(delta: torch.Tensor) -> str:
        return "L*={:+.3f} a*={:+.3f} b*={:+.3f}".format(*delta.tolist())

    def fix_region(self, image, reference, mask, components="luminance", inside_source="fit",
                   inside_offset_l=0.0, strength=1.0, outside_threshold=0.05, inside_threshold=0.95,
                   max_excess=8.0, estimator="mean", clip_k=3.0):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor with 4 dims [B,H,W,C], got shape: {tuple(image.shape)}")
        if image.shape[-1] != 3:
            raise ValueError(f"Expected a 3-channel IMAGE for Lab conversion, got {image.shape[-1]} channels")
        if reference.ndim != 4 or reference.shape[-1] != 3:
            raise ValueError(f"Expected reference IMAGE with shape [B,H,W,3], got: {tuple(reference.shape)}")
        if outside_threshold >= inside_threshold:
            raise ValueError(
                f"outside_threshold ({outside_threshold}) must stay below inside_threshold "
                f"({inside_threshold}); otherwise the two populations overlap and the outside "
                f"baseline gets contaminated by repainted content"
            )

        # 整条链路要过一次 RGB->Lab->RGB 往返，本身就不是无损的；strength 为 0 时
        # 在校验之后短路，保证原样返回，同时不让非法输入蒙混过关。
        if strength == 0:
            return (image.float(), "strength=0, unchanged")

        # kornia 是 comfy_extras/nodes_post_processing.py 的模块级依赖，必然可用；
        # 这里延迟导入只是不想让本扩展的加载路径多挂一个硬依赖，不是可选降级。
        import kornia

        # kornia 的 rgb_to_lab 假定输入在 [0,1]，越界不会产 NaN 但会让 Lab 值跑飞、
        # 统计量随之失真。上游并非所有 VAE 分支都 clamp，这里自己兜住。
        img = image.float().clamp(0.0, 1.0)
        batch, height, width, _ = img.shape

        ref = reference.float().to(img.device).clamp(0.0, 1.0)
        if ref.shape[1:3] != (height, width):
            logger.warning(
                f"[InpaintRegionColorFix] reference size {tuple(ref.shape[1:3])} != image size {(height, width)}, resizing reference"
            )
            ref = _resize_bchw_to(ref.permute(0, 3, 1, 2), (height, width)).permute(0, 2, 3, 1)
        ref = _broadcast_batch(ref, batch, "reference")

        msk = _normalize_mask(mask, height, width, img.device)
        # 两个统计总体都往保守方向收一格：外部用 max 膨胀遮罩后再取阈，核心区用 min
        # 腐蚀后再取阈，避免过渡带同时混进两边。ramp 用的是未经形态学的原始遮罩，
        # 所以核心边缘会欠校正约一格、外部边缘会渗入约一格，量级可忽略。
        grown = F.max_pool2d(msk.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        outside = _broadcast_batch(grown <= outside_threshold, batch, "mask")
        manual_inside = inside_source == "manual"
        if manual_inside:
            # 不看重绘区内容，核心区总体连算都不用算
            inside = None
        else:
            shrunk = -F.max_pool2d(-msk.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
            inside = _broadcast_batch(shrunk >= inside_threshold, batch, "mask")
        msk = _broadcast_batch(msk, batch, "mask")

        lab_img = kornia.color.rgb_to_lab(img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        lab_ref = kornia.color.rgb_to_lab(ref.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        chroma = components == "luminance_and_chroma"
        keep = torch.tensor([1.0, 1.0, 1.0] if chroma else [1.0, 0.0, 0.0],
                            dtype=lab_img.dtype, device=lab_img.device)
        offset = torch.tensor([inside_offset_l, 0.0, 0.0], dtype=lab_img.dtype, device=lab_img.device)
        min_outside = max(self.MIN_OUTSIDE_SAMPLES, int(height * width * self.MIN_OUTSIDE_RATIO))

        lab_out = torch.empty_like(lab_img)
        est_desc = f"{estimator}(k={clip_k:.1f})" if estimator == "clip" else estimator
        report_lines = [f"components={components} inside_source={inside_source} estimator={est_desc} "
                        f"inside_offset_l={inside_offset_l:+.2f} strength={strength:.2f} "
                        f"outside<={outside_threshold:.2f} inside>={inside_threshold:.2f} "
                        f"max_excess={max_excess:.1f}"]

        for b in range(batch):
            n_out = int(outside[b].sum().item())
            flags = []
            clip_note = ""

            outside_ok = n_out >= min_outside
            if outside_ok:
                delta_out = (self._masked_mean(lab_ref[b], outside[b], n_out)
                             - self._masked_mean(lab_img[b], outside[b], n_out))
            else:
                logger.warning(
                    f"[InpaintRegionColorFix] batch {b}: outside population too small "
                    f"({n_out} < {min_outside}), treating the outside baseline as unshifted"
                )
                delta_out = torch.zeros_like(keep)
                flags.append("OUTSIDE_ZEROED")

            if manual_inside:
                # 基准完全由 inside_offset_l 给出，它在下面统一加上，这里先置零
                n_in = "manual"
                delta_in = torch.zeros_like(keep)
            else:
                n_in = int(inside[b].sum().item())
                if n_in < self.MIN_INSIDE_SAMPLES:
                    logger.warning(
                        f"[InpaintRegionColorFix] batch {b}: repainted core too small "
                        f"({n_in} < {self.MIN_INSIDE_SAMPLES}), falling back to the outside baseline"
                    )
                    delta_in = delta_out
                    flags.append("INSIDE_FALLBACK")
                else:
                    if estimator == "median":
                        delta_in = self._masked_median(lab_ref[b] - lab_img[b], inside[b])
                    elif estimator == "clip":
                        diff_b = lab_ref[b] - lab_img[b]
                        if outside_ok:
                            delta_in, info = self._masked_clip(
                                diff_b, inside[b], diff_b[outside[b]], clip_k)
                            clip_note = " clip[keep={} it={}]".format(
                                "/".join(f"{int(v)}" for v in info["n_keep"].tolist()),
                                info["iters"])
                            if bool(info["froze"].any()):
                                flags.append("CLIP_AT_MEDIAN")
                        else:
                            # 尺度取自遮罩外的离散度；那里的样本连均值都测不出来，
                            # 更给不出可信的离散度。退回中位数而不是崩掉。
                            delta_in = self._masked_median(diff_b, inside[b])
                            flags.append("CLIP_NO_SCALE")
                    else:
                        delta_in = (self._masked_mean(lab_ref[b], inside[b], n_in)
                                    - self._masked_mean(lab_img[b], inside[b], n_in))
                    # 判据取 delta_in 超出 delta_out 的部分：delta_in 自身含全局 VAE
                    # 偏置，上游没接 MaskedColorMatch 时直接拿它比会误报。
                    # 也不受 components 过滤——luminance 模式下纯色相替换的 ΔL* 很小，
                    # 只有 a*/b* 会暴露「假设不成立」，过滤后就永远看不到了。
                    excess = delta_in - delta_out
                    if max_excess > 0 and bool((excess.abs() > max_excess).any()):
                        logger.warning(
                            f"[InpaintRegionColorFix] batch {b}: inside baseline exceeds the outside one by "
                            f"{[round(v, 2) for v in excess.tolist()]}, over max_excess={max_excess} - "
                            f"this is more likely a real content change than drift, capping the correction. "
                            f"Use inside_source=manual with a calibrated offset for heavy content replacement."
                        )
                        # 限幅而不是放弃：退回 delta_out 等于主动接受一整个 drift 量的
                        # 偏差，而封顶只是不让校正跟着内容变化跑远，方向仍然是对的。
                        delta_in = delta_out + excess.clamp(-max_excess, max_excess)
                        flags.append("CLAMPED")

            # 上游任何一个坏像素都会让均值变成 NaN，再逐像素相加就是整图报废
            if not bool(torch.isfinite(torch.stack([delta_out, delta_in])).all()):
                logger.warning(
                    f"[InpaintRegionColorFix] batch {b}: non-finite baseline shift "
                    f"(NaN/Inf in image or reference), skipping correction"
                )
                lab_out[b] = lab_img[b]
                report_lines.append(f"[{b}] SKIPPED non-finite statistics")
                continue

            # offset 不受 components 过滤：暴露出来的就只有 L*，用户设了就该生效
            delta_out = delta_out * keep
            delta_in = delta_in * keep + offset
            # 按遮罩值在两个基准之间线性过渡。下游 ImageCompositeMasked 是
            # mask * source + (1 - mask) * destination，所以相对「未校正合成图」的净
            # 修正是 m * delta_out + m^2 * (delta_in - delta_out)：一次项对应全局 VAE
            # 分量，二次项对应内外基准差。m→0 时连续趋零，不产生新接缝。
            ramp = msk[b].unsqueeze(-1)
            lab_out[b] = lab_img[b] + (delta_out + (delta_in - delta_out) * ramp) * strength

            suffix = (" " + " ".join(flags)) if flags else ""
            report_lines.append(
                f"[{b}] n_out={n_out} n_in={n_in} | outside {self._fmt_delta(delta_out)} "
                f"| inside {self._fmt_delta(delta_in)}{clip_note}{suffix}"
            )

        # L* 平移会把高光推出色域：R/G 截到 1.0 而 B 仍在上升，净效果是色相/饱和度
        # 位移而非单纯提亮，迭代多轮后高光会逐步压平。8bit 管线下无解。
        # 截断实际发生在 kornia 内部（lab_to_rgb 的 clip 默认为 True），外面这层
        # clamp 只是防它哪天改默认值。
        corrected = kornia.color.lab_to_rgb(lab_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).clamp_(0.0, 1.0)
        report = "\n".join(report_lines)
        logger.debug(f"[InpaintRegionColorFix]\n{report}")
        return (corrected, report)
