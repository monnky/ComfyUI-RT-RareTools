# ====================================================================================================
# [RT] Interactive Two Image Compare Suite
# Real-time visual A/B image comparator with interactive slide wipe, click toggle,
# side-by-side view, difference blend, and fullscreen inspection.
# 100% English ASCII Only
# ====================================================================================================

import os
import random
import folder_paths
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class RT_Image_Compare:
    """
    RT Image Compare (Interactive)
    Real-time interactive two-image visual comparison node.
    Features:
      - Interactive wipe slider (drag or hover to scrub between images)
      - Instant click-to-toggle (A/B comparison for retouching/VFX)
      - Side-by-side split & stacked view
      - GPU difference blend mode (mix-blend-mode: difference)
      - Smooth fade / dissolve slider
      - Fullscreen lightbox modal with deep zoom and pan
      - Auto-conforming for mismatched image dimensions
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Primary image A (Base / Before / Reference)."}),
                "image_b": ("IMAGE", {"tooltip": "Secondary image B (Modified / After / Comparison)."}),
            },
            "optional": {
                "initial_mode": (["Slide (Wipe)", "Difference", "Click (Toggle)", "Side-by-Side", "Fade (Dissolve)"], {
                    "default": "Slide (Wipe)",
                    "tooltip": "Default interactive visual comparison mode."
                }),
                "split_direction": (["Horizontal", "Vertical"], {
                    "default": "Horizontal",
                    "tooltip": "Wipe divider orientation."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image_a", "image_b", "comparison")
    OUTPUT_NODE = True
    FUNCTION = "compare_images"
    CATEGORY = "RareTutor/Image"
    TITLE = "RT Image Compare"
    DESCRIPTION = "Interactive visual two-image comparator. Drag the slider on the node to wipe between images, click to toggle, or switch to side-by-side and difference views."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def compare_images(self, image_a, image_b, initial_mode="Slide (Wipe)", split_direction="Horizontal"):
        output_dir = folder_paths.get_temp_directory()
        rand_id = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8))

        # 1. Ensure 4D tensors: [B, H, W, C]
        if image_a.ndim == 3:
            image_a = image_a.unsqueeze(0)
        if image_b.ndim == 3:
            image_b = image_b.unsqueeze(0)

        # 2. Conform color channels: ComfyUI standard is 3-channel RGB [B, H, W, 3]
        # Strips alpha channel or extra channels so images are 100% opaque RGB with no transparency
        if image_a.shape[-1] > 3:
            image_a = image_a[..., :3]
        elif image_a.shape[-1] == 1:
            image_a = image_a.repeat(1, 1, 1, 3)

        if image_b.shape[-1] > 3:
            image_b = image_b[..., :3]
        elif image_b.shape[-1] == 1:
            image_b = image_b.repeat(1, 1, 1, 3)

        # 3. Sanitize mode & direction inputs
        if isinstance(initial_mode, list):
            initial_mode = initial_mode[0] if initial_mode else "Slide (Wipe)"
        if isinstance(split_direction, list):
            split_direction = split_direction[0] if split_direction else "Horizontal"

        # 4. Determine batch count to compare
        b_a = image_a.shape[0]
        b_b = image_b.shape[0]
        max_b = max(b_a, b_b)

        # Broadcast if needed
        if b_a < max_b:
            repeats = (max_b + b_a - 1) // b_a
            image_a = image_a.repeat(repeats, 1, 1, 1)[:max_b]
        if b_b < max_b:
            repeats = (max_b + b_b - 1) // b_b
            image_b = image_b.repeat(repeats, 1, 1, 1)[:max_b]

        results_a = []
        results_b = []

        # 5. Save temp frames for frontend interactive display (guaranteed 100% opaque RGB)
        for idx in range(max_b):
            frame_a = (255.0 * image_a[idx, :, :, :3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)
            img_a = Image.fromarray(frame_a, mode="RGB")
            fname_a = f"rt_cmp_a_{rand_id}_{idx}.png"
            path_a = os.path.join(output_dir, fname_a)
            img_a.save(path_a, compress_level=1)
            results_a.append({
                "filename": fname_a,
                "subfolder": "",
                "type": "temp",
                "width": img_a.width,
                "height": img_a.height,
            })

            frame_b = (255.0 * image_b[idx, :, :, :3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)
            img_b = Image.fromarray(frame_b, mode="RGB")
            fname_b = f"rt_cmp_b_{rand_id}_{idx}.png"
            path_b = os.path.join(output_dir, fname_b)
            img_b.save(path_b, compress_level=1)
            results_b.append({
                "filename": fname_b,
                "subfolder": "",
                "type": "temp",
                "width": img_b.width,
                "height": img_b.height,
            })

        # 6. Generate a standard 50/50 split comparison tensor for downstream node chaining
        # Align image_b to image_a dimensions if they differ
        _, h_a, w_a, _ = image_a.shape
        _, h_b, w_b, _ = image_b.shape

        if (h_a, w_a) != (h_b, w_b):
            b_chw = image_b.permute(0, 3, 1, 2)
            b_resized = F.interpolate(b_chw, size=(h_a, w_a), mode="bicubic", align_corners=False)
            b_aligned = b_resized.permute(0, 2, 3, 1)
        else:
            b_aligned = image_b

        comp_tensor = b_aligned.clone()
        if split_direction == "Horizontal":
            mid = w_a // 2
            comp_tensor[:, :, :mid, :] = image_a[:, :, :mid, :]
            # Draw subtle 2px white divider line
            line_start = max(0, mid - 1)
            line_end = min(w_a, mid + 1)
            comp_tensor[:, :, line_start:line_end, :] = 1.0
        else:
            mid = h_a // 2
            comp_tensor[:, :mid, :, :] = image_a[:, :mid, :, :]
            line_start = max(0, mid - 1)
            line_end = min(h_a, mid + 1)
            comp_tensor[:, line_start:line_end, :, :] = 1.0

        ui_payload = {
            "images_a": results_a,
            "images_b": results_b,
            "mode": [initial_mode],
            "direction": [split_direction],
        }

        print(f"[RT Image Compare] Executed successfully: {max_b} comparison image(s) generated.")

        return {
            "ui": ui_payload,
            "result": (image_a, image_b, comp_tensor)
        }


NODE_CLASS_MAPPINGS = {
    "RT_Image_Compare": RT_Image_Compare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_Image_Compare": "RT Image Compare",
}
