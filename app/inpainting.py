"""Image inpainting using Stable Diffusion (local GPU)."""
import numpy as np
import cv2
from PIL import Image

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import AutoPipelineForInpainting
        _pipe = AutoPipelineForInpainting.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        _pipe = _pipe.to("cuda")
        _pipe.enable_attention_slicing()
    return _pipe


def inpaint(image_path: str, mask_path: str, output_path: str) -> str:
    """Remove masked areas using Stable Diffusion Inpainting.

    Resize to 512x512 for SD, then blend result back into original
    so only masked areas are replaced.
    """
    original = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    orig_w, orig_h = original.size

    # SD optimal size
    sd = 512
    img_sd = original.resize((sd, sd), Image.LANCZOS)
    mask_sd = mask.resize((sd, sd), Image.NEAREST)

    result_sd = _get_pipe()(
        prompt="natural clean background, photorealistic, seamless, no people, matching environment",
        negative_prompt="blurry, distorted, artifacts, text, watermark, ugly, discolored",
        image=img_sd,
        mask_image=mask_sd,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=1.0,
    ).images[0]

    # Resize back and blend
    result_full = np.array(result_sd.resize((orig_w, orig_h), Image.LANCZOS))
    original_np = np.array(original)
    mask_np = np.array(mask)

    if mask_np.shape[:2] != (orig_h, orig_w):
        mask_np = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Feather mask edges for smooth transition
    alpha = cv2.GaussianBlur(
        np.stack([mask_np] * 3, axis=-1).astype(np.float32) / 255.0, (11, 11), 5
    )

    blended = (result_full * alpha + original_np * (1 - alpha)).astype(np.uint8)
    Image.fromarray(blended).save(output_path, quality=95)
    return output_path
