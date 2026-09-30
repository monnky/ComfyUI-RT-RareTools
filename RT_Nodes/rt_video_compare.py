# ====================================================================================================
# [RT] Interactive Two Video Compare Suite
# Real-time synchronized dual-video visual comparator with interactive wipe slider,
# click-to-toggle, side-by-side, real-time difference blend, timeline scrubber, and frame stepping.
# 100% English ASCII Only
# ====================================================================================================

import os
import random
import shutil
import subprocess
import folder_paths
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def encode_video_mp4(tensor_fhwc, fps, output_filepath):
    """
    Encodes an [F, H, W, C] float32 tensor in [0, 1] to an MP4 video using ffmpeg stdin pipe.
    Falls back to animated WebP if ffmpeg is unavailable.
    """
    if tensor_fhwc.shape[-1] > 3:
        tensor_fhwc = tensor_fhwc[..., :3]
    elif tensor_fhwc.shape[-1] == 1:
        tensor_fhwc = tensor_fhwc.repeat(1, 1, 1, 3)

    f_num, h_num, w_num, c_num = tensor_fhwc.shape
    ffmpeg_exe = shutil.which("ffmpeg")

    # If ffmpeg is available, encode high-speed MP4
    if ffmpeg_exe:
        raw_bytes = (tensor_fhwc.detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8).tobytes()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w_num}x{h_num}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-crf", "19",
            output_filepath
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.communicate(input=raw_bytes)
            if proc.returncode == 0 and os.path.exists(output_filepath):
                return True, "mp4"
        except Exception:
            pass

    # Fallback to animated WebP
    webp_path = os.path.splitext(output_filepath)[0] + ".webp"
    frames = []
    for i in range(f_num):
        arr = (tensor_fhwc[i].detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
        frames.append(Image.fromarray(arr, mode="RGB"))

    duration_ms = max(10, int(1000 / fps))
    frames[0].save(
        webp_path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=85
    )
    return True, "webp"


class RT_Video_Compare:
    """
    RT Video Compare (Interactive)
    Real-time interactive two-video visual comparison node.
    Features:
      - Synchronized dual-video playback
      - Interactive wipe slider (drag or hover across playing video)
      - Instant click-to-toggle between Video A and Video B
      - Synchronized timeline scrubber and frame-by-frame stepping
      - Real-time GPU difference blend mode
      - Side-by-side synchronized video playback
      - Auto-alignment for mismatched frame counts and resolutions
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_a": ("IMAGE", {"tooltip": "Primary video frames (IMAGE batch [F, H, W, C])."}),
                "video_b": ("IMAGE", {"tooltip": "Secondary video frames (IMAGE batch [F, H, W, C])."}),
            },
            "optional": {
                "fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "tooltip": "Playback framerate in frames per second."
                }),
                "initial_mode": (["Slide (Wipe)", "Click (Toggle)", "Side-by-Side", "Difference", "Fade (Dissolve)"], {
                    "default": "Slide (Wipe)",
                    "tooltip": "Default interactive video comparison mode."
                }),
                "match_frames": (["Trim to Shorter", "Loop Shorter", "Hold Last Frame", "Pad Black"], {
                    "default": "Trim to Shorter",
                    "tooltip": "How to align videos if they have different frame counts."
                }),
                "match_resolution": (["Resize to A", "Resize to B", "Resize to Larger", "Resize to Smaller", "Center Crop"], {
                    "default": "Resize to A",
                    "tooltip": "How to conform videos if dimensions differ."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("video_a", "video_b", "comparison")
    OUTPUT_NODE = True
    FUNCTION = "compare_videos"
    CATEGORY = "RareTutor/Video"
    TITLE = "RT Video Compare"
    DESCRIPTION = "Interactive synchronized two-video comparator with wipe slider, timeline scrubber, and frame stepping."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def compare_videos(
        self,
        video_a,
        video_b,
        fps=24,
        initial_mode="Slide (Wipe)",
        match_frames="Trim to Shorter",
        match_resolution="Resize to A"
    ):
        output_dir = folder_paths.get_temp_directory()
        rand_id = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8))

        # 1. Ensure 4D tensors: [F, H, W, C]
        if video_a.ndim == 3:
            video_a = video_a.unsqueeze(0)
        if video_b.ndim == 3:
            video_b = video_b.unsqueeze(0)

        # 2. Conform color channels: ComfyUI standard is 3-channel RGB [F, H, W, 3]
        # Strips alpha channel or extra channels so videos are 100% opaque RGB with no transparency
        if video_a.shape[-1] > 3:
            video_a = video_a[..., :3]
        elif video_a.shape[-1] == 1:
            video_a = video_a.repeat(1, 1, 1, 3)

        if video_b.shape[-1] > 3:
            video_b = video_b[..., :3]
        elif video_b.shape[-1] == 1:
            video_b = video_b.repeat(1, 1, 1, 3)

        # 3. Sanitize mode & options inputs
        if isinstance(initial_mode, list):
            initial_mode = initial_mode[0] if initial_mode else "Slide (Wipe)"
        if isinstance(match_frames, list):
            match_frames = match_frames[0] if match_frames else "Trim to Shorter"
        if isinstance(match_resolution, list):
            match_resolution = match_resolution[0] if match_resolution else "Resize to A"

        f_a, h_a, w_a, _ = video_a.shape
        f_b, h_b, w_b, _ = video_b.shape

        # Step 1: Align Frame Counts
        if f_a != f_b:
            if match_frames == "Trim to Shorter":
                target_f = min(f_a, f_b)
                video_a = video_a[:target_f]
                video_b = video_b[:target_f]
            elif match_frames == "Loop Shorter":
                target_f = max(f_a, f_b)
                if f_a < target_f:
                    repeats = (target_f // f_a) + 1
                    video_a = video_a.repeat(repeats, 1, 1, 1)[:target_f]
                if f_b < target_f:
                    repeats = (target_f // f_b) + 1
                    video_b = video_b.repeat(repeats, 1, 1, 1)[:target_f]
            elif match_frames == "Hold Last Frame":
                target_f = max(f_a, f_b)
                if f_a < target_f:
                    last_frame = video_a[-1:].repeat(target_f - f_a, 1, 1, 1)
                    video_a = torch.cat([video_a, last_frame], dim=0)
                if f_b < target_f:
                    last_frame = video_b[-1:].repeat(target_f - f_b, 1, 1, 1)
                    video_b = torch.cat([video_b, last_frame], dim=0)
            elif match_frames == "Pad Black":
                target_f = max(f_a, f_b)
                if f_a < target_f:
                    pad = torch.zeros(target_f - f_a, h_a, w_a, 3, device=video_a.device, dtype=video_a.dtype)
                    video_a = torch.cat([video_a, pad], dim=0)
                if f_b < target_f:
                    pad = torch.zeros(target_f - f_b, h_b, w_b, 3, device=video_b.device, dtype=video_b.dtype)
                    video_b = torch.cat([video_b, pad], dim=0)

        # Step 2: Align Resolutions if needed
        f_aligned, h_a, w_a, _ = video_a.shape
        _, h_b, w_b, _ = video_b.shape

        if (h_a, w_a) != (h_b, w_b):
            if match_resolution == "Resize to A":
                b_chw = video_b.permute(0, 3, 1, 2)
                video_b = F.interpolate(b_chw, size=(h_a, w_a), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
            elif match_resolution == "Resize to B":
                a_chw = video_a.permute(0, 3, 1, 2)
                video_a = F.interpolate(a_chw, size=(h_b, w_b), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
            elif match_resolution == "Resize to Larger":
                th = max(h_a, h_b)
                tw = max(w_a, w_b)
                a_chw = video_a.permute(0, 3, 1, 2)
                b_chw = video_b.permute(0, 3, 1, 2)
                video_a = F.interpolate(a_chw, size=(th, tw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
                video_b = F.interpolate(b_chw, size=(th, tw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
            elif match_resolution == "Resize to Smaller":
                th = min(h_a, h_b)
                tw = min(w_a, w_b)
                a_chw = video_a.permute(0, 3, 1, 2)
                b_chw = video_b.permute(0, 3, 1, 2)
                video_a = F.interpolate(a_chw, size=(th, tw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
                video_b = F.interpolate(b_chw, size=(th, tw), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
            elif match_resolution == "Center Crop":
                th = min(h_a, h_b)
                tw = min(w_a, w_b)
                y1_a = (h_a - th) // 2
                x1_a = (w_a - tw) // 2
                video_a = video_a[:, y1_a:y1_a + th, x1_a:x1_a + tw, :]
                y1_b = (h_b - th) // 2
                x1_b = (w_b - tw) // 2
                video_b = video_b[:, y1_b:y1_b + th, x1_b:x1_b + tw, :]

        # Step 3: Render Synchronized 50/50 Split Video Tensor for Output Pin
        _, final_h, final_w, _ = video_a.shape
        comp_tensor = video_b.clone()
        mid = final_w // 2
        comp_tensor[:, :, :mid, :] = video_a[:, :, :mid, :]
        line_start = max(0, mid - 1)
        line_end = min(final_w, mid + 1)
        comp_tensor[:, :, line_start:line_end, :] = 1.0  # 2px white divider line

        # Step 4: Encode Video Files for Frontend Interactive Playback
        file_a_path = os.path.join(output_dir, f"rt_vid_a_{rand_id}.mp4")
        file_b_path = os.path.join(output_dir, f"rt_vid_b_{rand_id}.mp4")

        success_a, fmt_a = encode_video_mp4(video_a, fps, file_a_path)
        success_b, fmt_b = encode_video_mp4(video_b, fps, file_b_path)

        final_fname_a = f"rt_vid_a_{rand_id}.{fmt_a}"
        final_fname_b = f"rt_vid_b_{rand_id}.{fmt_b}"

        ui_payload = {
            "video_a": [{"filename": final_fname_a, "subfolder": "", "type": "temp", "format": fmt_a, "width": final_w, "height": final_h}],
            "video_b": [{"filename": final_fname_b, "subfolder": "", "type": "temp", "format": fmt_b, "width": final_w, "height": final_h}],
            "fps": [fps],
            "frame_count": [f_aligned],
            "mode": [initial_mode],
        }

        return {
            "ui": ui_payload,
            "result": (video_a, video_b, comp_tensor)
        }


NODE_CLASS_MAPPINGS = {
    "RT_Video_Compare": RT_Video_Compare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_Video_Compare": "RT Video Compare",
}
