"""CLIPTextEncode with a persistent disk cache.

A cache hit skips the text encoder entirely, so the model never has to be loaded into VRAM just to
re-encode a prompt seen before (e.g. after a restart, or when switching between prompts).

Layout: user/slowargo/clip_text_encode/<model file names>/<sha256>.pt
Each model group is an independent LRU (by file mtime) capped at MAX_CACHE_ITEMS.

The key covers everything carried on the CLIP object that changes the output: model file names,
the loader's simple args (clip_type, ...), LoRA patches (sampled fingerprint), clip skip
(layer_idx), tokenizer options and the prompt text. It is intentionally loose (no mtime/size,
sampled LoRA tensors) -- replacing a model file in place under the same name needs a manual cleanup
of its group directory.
"""

import enum
import hashlib
import logging
import os
import re
from collections import OrderedDict

import folder_paths
import nodes
import torch

logger = logging.getLogger(__name__)

CACHE_ROOT = os.path.join(folder_paths.get_user_directory(), "slowargo", "clip_text_encode")
MAX_CACHE_ITEMS = 100
# Elements sampled per LoRA tensor; enough to tell different LoRAs apart, not a strict hash
SAMPLES_PER_TENSOR = 16

# patches_uuid -> fingerprint. The uuid changes on every add_patches and is copied by clone(),
# so the (relatively costly) tensor sampling runs once per LoRA stack per session.
_patch_fp_memo: "OrderedDict[object, str]" = OrderedDict()
_PATCH_FP_MEMO_SIZE = 64
# group dir -> number of .pt files, so pruning does not list the directory on every write
_group_counts: dict[str, int] = {}


def _stable_repr(value):
    """repr() for values that are stable across restarts; None for anything else (skipped)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return repr(value)
    if isinstance(value, enum.Enum):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = [_stable_repr(v) for v in value]
        return None if None in parts else "[" + ",".join(parts) + "]"
    return None


def _iter_tensors(value):
    """Yield the tensors inside a patch value (WeightAdapterBase, legacy tuples, or a bare tensor)."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_tensors(v)
    elif hasattr(value, "weights"):
        yield from _iter_tensors(value.weights)


def _tensor_fingerprint(t):
    flat = t.detach().reshape(-1)
    n = flat.numel()
    header = f"{tuple(t.shape)}:{t.dtype}:"
    if n == 0:
        return header.encode("utf-8")
    idx = torch.linspace(0, n - 1, steps=min(n, SAMPLES_PER_TENSOR), device=flat.device).long()
    sample = flat[idx].to("cpu", torch.float32)
    return header.encode("utf-8") + sample.numpy().tobytes()


def _patches_fingerprint(patcher):
    uuid = getattr(patcher, "patches_uuid", None)
    if uuid is not None and uuid in _patch_fp_memo:
        _patch_fp_memo.move_to_end(uuid)
        return _patch_fp_memo[uuid]

    hasher = hashlib.sha256()
    for key in sorted(patcher.patches.keys()):
        hasher.update(key.encode("utf-8"))
        for strength_patch, value, strength_model, offset, function in patcher.patches[key]:
            func_name = getattr(function, "__qualname__", repr(function)) if function is not None else ""
            hasher.update(f"|{strength_patch}|{strength_model}|{offset}|{func_name}|".encode("utf-8"))
            hasher.update(type(value).__name__.encode("utf-8"))
            for t in _iter_tensors(value):
                hasher.update(_tensor_fingerprint(t))
    fp = hasher.hexdigest()

    if uuid is not None:
        _patch_fp_memo[uuid] = fp
        while len(_patch_fp_memo) > _PATCH_FP_MEMO_SIZE:
            _patch_fp_memo.popitem(last=False)
    return fp


def _group_name(paths):
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)
    name = re.sub(r"[^\w.\-]+", "_", "+".join(names))
    if len(name) > 100:
        # Long multi-encoder names: keep it readable but unique
        name = name[:90] + "-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return name


def _cache_location(clip, text):
    """Return (group_dir, key), or raise if this CLIP cannot be identified."""
    patcher = clip.patcher
    cached_init = getattr(patcher, "cached_patcher_init", None)
    if cached_init is None:
        raise ValueError("CLIP has no cached_patcher_init (not loaded by a core loader)")

    args = cached_init[1]
    paths = args[0]
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("cached_patcher_init has no model paths")

    hasher = hashlib.sha256()
    hasher.update(getattr(cached_init[0], "__name__", "").encode("utf-8"))
    for p in sorted(os.path.basename(p) for p in paths):
        hasher.update(p.encode("utf-8") + b"\x00")
    # clip_type and other simple loader args; dicts/objects (model_options...) are skipped
    for i, arg in enumerate(args[1:]):
        r = _stable_repr(arg)
        if r is not None:
            hasher.update(f"a{i}={r}\x00".encode("utf-8"))
    hasher.update(f"layer={clip.layer_idx}\x00".encode("utf-8"))
    hasher.update(f"tok={sorted((k, repr(v)) for k, v in clip.tokenizer_options.items())}\x00".encode("utf-8"))
    hasher.update(f"patches={_patches_fingerprint(patcher)}\x00".encode("utf-8"))
    hasher.update(text.encode("utf-8"))

    return os.path.join(CACHE_ROOT, _group_name(paths)), hasher.hexdigest()


def _prune_group(group_dir):
    """Drop the least recently used files once the group exceeds MAX_CACHE_ITEMS."""
    count = _group_counts.get(group_dir)
    if count is not None and count <= MAX_CACHE_ITEMS:
        return
    try:
        entries = [e for e in os.scandir(group_dir) if e.name.endswith(".pt")]
    except FileNotFoundError:
        _group_counts[group_dir] = 0
        return
    if len(entries) > MAX_CACHE_ITEMS:
        entries.sort(key=lambda e: e.stat().st_mtime)
        for e in entries[: len(entries) - MAX_CACHE_ITEMS]:
            try:
                os.remove(e.path)
                logger.debug("SimpleCachedCLIPTextEncode: pruned %s", e.path)
            except OSError:
                logger.warning("SimpleCachedCLIPTextEncode: failed to prune %s", e.path, exc_info=True)
    _group_counts[group_dir] = min(len(entries), MAX_CACHE_ITEMS)


class SimpleCachedCLIPTextEncode(nodes.CLIPTextEncode):
    CATEGORY = "Slowargo"
    DESCRIPTION = (
        "CLIP Text Encode with a disk cache (per model, 100 most recently used prompts). A cache hit "
        "skips loading the text encoder. Bypassed when the CLIP carries hooks."
    )

    def encode(self, clip, text):
        if clip is None:
            return super().encode(clip, text)
        # Scheduled hooks make the output depend on more than the CLIP state, and put hook objects
        # into the conditioning; don't cache those.
        if clip.patcher.forced_hooks is not None or clip.apply_hooks_to_conds:
            return super().encode(clip, text)

        try:
            group_dir, key = _cache_location(clip, text)
        except Exception:
            logger.warning("SimpleCachedCLIPTextEncode: cannot identify CLIP, cache disabled", exc_info=True)
            return super().encode(clip, text)
        cache_path = os.path.join(group_dir, f"{key}.pt")

        try:
            conditioning = torch.load(cache_path, weights_only=True)
            try:
                os.utime(cache_path, None)
            except OSError:
                pass
            logger.info("SimpleCachedCLIPTextEncode: cache hit %s/%s", os.path.basename(group_dir), key[:12])
            return (conditioning,)
        except FileNotFoundError:
            pass
        except Exception:
            logger.warning("SimpleCachedCLIPTextEncode: failed to load %s, re-encoding", cache_path, exc_info=True)

        logger.info("SimpleCachedCLIPTextEncode: cache miss %s/%s", os.path.basename(group_dir), key[:12])
        conditioning = super().encode(clip, text)[0]

        try:
            os.makedirs(group_dir, exist_ok=True)
            existed = os.path.exists(cache_path)
            # Write-then-rename so an interrupted save never leaves a truncated cache file
            tmp_path = cache_path + ".tmp"
            torch.save(conditioning, tmp_path)
            os.replace(tmp_path, cache_path)
            if not existed and group_dir in _group_counts:
                _group_counts[group_dir] += 1
            _prune_group(group_dir)
        except Exception:
            logger.warning("SimpleCachedCLIPTextEncode: failed to write %s", cache_path, exc_info=True)

        return (conditioning,)
