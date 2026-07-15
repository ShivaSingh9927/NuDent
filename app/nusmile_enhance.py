"""AI polish pass for Nu Smile — Stable Diffusion img2img on CPU.

We keep the CAD arch geometry intact (low denoise), and rely on the base
model to add micro-detail: enamel highlights, subsurface scatter, and a
believable lip/tooth transition. Runs offline once the model is cached.
"""
import os
import numpy as np
import cv2

_PIPE = None
_MODEL_ID = os.environ.get("NUSMILE_SD_MODEL",
                           "stable-diffusion-v1-5/stable-diffusion-v1-5")


def is_available():
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except Exception:
        return False


def _get_pipeline(progress_cb=None):
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
    if progress_cb:
        progress_cb("Loading Stable Diffusion pipeline (first run downloads ~4 GB)…")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        _MODEL_ID, torch_dtype=torch.float32, safety_checker=None,
    )
    # UniPC converges much faster than the default DDIM/PNDM — ~10 steps
    # matches ~22 with the default scheduler at similar quality.
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    _PIPE = pipe
    return pipe


def _crop_around_lips(img_bgr, lip_poly, pad_frac=0.6):
    """Return (crop_bgr, (x0, y0, x1, y1)) — square-ish region centred on
    the lips, padded by ``pad_frac`` of the lip extent so SD sees enough
    facial context to blend cleanly."""
    h, w = img_bgr.shape[:2]
    xs, ys = lip_poly[:, 0], lip_poly[:, 1]
    lx0, lx1 = float(xs.min()), float(xs.max())
    ly0, ly1 = float(ys.min()), float(ys.max())
    cx, cy = 0.5 * (lx0 + lx1), 0.5 * (ly0 + ly1)
    half = 0.5 * max(lx1 - lx0, ly1 - ly0) * (1.0 + pad_frac)
    half = max(half, 128.0)
    x0 = int(max(0, cx - half)); y0 = int(max(0, cy - half))
    x1 = int(min(w, cx + half)); y1 = int(min(h, cy + half))
    return img_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def _resize_pad(img_bgr, target=512):
    """Pad-to-square then resize to (target, target)."""
    h, w = img_bgr.shape[:2]
    s = max(h, w)
    pad = np.zeros((s, s, 3), dtype=img_bgr.dtype)
    dy = (s - h) // 2
    dx = (s - w) // 2
    pad[dy:dy + h, dx:dx + w] = img_bgr
    small = cv2.resize(pad, (target, target), interpolation=cv2.INTER_AREA)
    return small, (dy, dx, s)


def _unpad_resize(img_bgr, meta, out_hw):
    dy, dx, s = meta
    up = cv2.resize(img_bgr, (s, s), interpolation=cv2.INTER_LANCZOS4)
    h, w = out_hw
    return up[dy:dy + h, dx:dx + w]


def enhance(composite_bgr, lip_poly, lip_mask,
            denoise=0.28, steps=12, guidance=6.0, seed=1234,
            progress_cb=None):
    """Polish the composited image around the lips with SD img2img.

    Only the lip region is passed to the model (for speed and to avoid
    altering the rest of the face). The result is blended back into the
    original composite with a soft mask centred on the lips.
    """
    from PIL import Image
    import torch

    pipe = _get_pipeline(progress_cb)
    if progress_cb:
        progress_cb(f"Cropping and preparing input ({steps} steps)…")

    crop, (x0, y0, x1, y1) = _crop_around_lips(composite_bgr, lip_poly)
    ch, cw = crop.shape[:2]
    small, meta = _resize_pad(crop, target=512)
    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    prompt = (
        "close-up photograph of front teeth in a natural smile, "
        "realistic enamel with soft saliva highlights, subtle "
        "translucency at the incisal edge, healthy pink lips, "
        "soft studio lighting, sharp focus, dental photography"
    )
    negative = (
        "cartoon, illustration, painting, plastic, blurry, extra teeth, "
        "missing teeth, deformed, low quality, wax figure"
    )

    if progress_cb:
        progress_cb("Running Stable Diffusion (this will take 30–60 s on CPU)…")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out = pipe(prompt=prompt, negative_prompt=negative,
               image=pil, strength=float(denoise),
               num_inference_steps=int(steps),
               guidance_scale=float(guidance),
               generator=gen).images[0]

    out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    up = _unpad_resize(out_bgr, meta, (ch, cw))

    result = composite_bgr.copy()
    result[y0:y1, x0:x1] = up

    # Soft blend near the lip region only — don't let SD edits leak onto the
    # eyes / forehead. Mask = dilated lip window with a wide feather.
    lip_soft = cv2.dilate(
        lip_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    ).astype(np.float32) / 255.0
    lip_soft = cv2.GaussianBlur(lip_soft, (61, 61), 0)
    m3 = lip_soft[:, :, None]
    blended = composite_bgr.astype(np.float32) * (1.0 - m3) + \
              result.astype(np.float32) * m3
    return np.clip(blended, 0, 255).astype(np.uint8)
