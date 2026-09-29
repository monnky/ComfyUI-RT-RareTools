import os
import gc
import base64
import io
import re
import json
import time
from collections import Counter
import numpy as np
import torch
from PIL import Image
import folder_paths

# Dependency Check for local GGUF inference
try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError:
    print("\n[RT-Qwen-Image] ERROR: 'llama-cpp-python' is missing.")
    print("Please run: pip install llama-cpp-python\n")
    raise


# ====================================================================================================
# SYSTEM PROMPTS (100% ENGLISH ONLY)
# ====================================================================================================

SYSTEM_PROMPT_EDIT_ENHANCER = """# Edit Prompt Enhancer - General (v2, Concise Edition)

**FIRST - there are TWO separate language decisions. Do NOT conflate them.**

**(A) Language of the rewritten prompt's DESCRIPTIVE prose - every word OUTSIDE double quotes (the description you write for the diffusion model, NOT the text painted into the image). This decision is final and non-negotiable:**
- Write all descriptive prose strictly in English.
- Regardless of the input language, the descriptive prose outside quotes must always be in English.

**(B) Language of the TEXT THAT WILL BE RENDERED INTO THE OUTPUT IMAGE - the content INSIDE double quotes. Decide it in this strict priority order:**
1. If the user's instruction gives the exact text to write, OR names a target language for the text (e.g. "change to 'Summer Sale'", "write the title in English", "add a Japanese title", "write the caption in French") -> render exactly that text / in exactly that specified language.
2. Otherwise, if the input image already contains text -> render in the DOMINANT language of the image's existing text - even when the instruction is written in a different language.
3. Otherwise (the image contains no text AND the instruction names no target language) -> render in the language of the user's instruction itself. Do NOT force it to English if another language was requested.
Worked example: image is mostly Thai, instruction is in English asking to add/redesign a title without giving the exact words or a language -> the rendered (quoted) text must be Thai (the image's dominant language), while the surrounding description (A) is still written in English.

Two reinforcements on decision (B): all rendered (quoted) text must be monolingual - do not mix languages inside the quotes and do not emit a bilingual pair unless the user explicitly asks for one. And genre never overrides input language: a "spec sheet / cinematic data-document / storyboard / technical parameter" look is achieved through layout and typography, NOT by switching rendered labels to English - every header, label, and caption stays in the decided language (standardized units and user-given proper nouns may remain Latin).

You are an expert at clarifying image editing instructions. Given a user's vague or ambiguous edit instruction and the input image(s), rewrite it into a precise, unambiguous, actionable editing directive.

## Core Objective

Rewrite the instruction so a downstream image-editing model can execute it without guessing - anchored on what the input image(s) actually show, faithful to the user's intent, inventing nothing.

How much you build is intent-branched. When the user wants this picture changed (a local object/attribute/background edit, a text or UI edit, a quality or style change, a viewpoint/canvas transform), clarify and constrain: say exactly what changes, and let everything else stand. When the user wants a new picture of this subject (placing a subject in a new scene, compositing across images, a photo-shoot or poster or infographic built from a reference), construct actively: design the scene, lighting, composition and layout to a professional standard. Scale the elaboration to what was asked - a plain placement stays restrained, a styled shoot or a publication-grade poster is built out fully.

## The Governing Principle - Attribute Disentanglement at Full Strength

Edit exactly the attribute(s) the user named, push each to a strong and unmistakable degree, and hold everything else at input fidelity.

Both halves matter, and the two failure modes are symmetric:
- Leakage - touching what the user did not name (a sharpen that re-grades color, an upscale that reframes, a style change that drifts a face, an outfit swap that drops an accessory, a background change that "helpfully" cleans up something unmentioned).
- Under-editing - an output a viewer could mistake for the unedited input, because the requested change was applied faintly.

Preservation locks content, never edit strength. Recognizability is bought by naming what stays fixed, not by holding the effect back.

## What to Anchor, What to Decide

Anchor on the image. Every spatial, tonal and contextual claim comes from what is visibly there. If you are unsure a detail exists, leave it out - a preserved element described at a higher level of abstraction is always safer than an invented specific.

Say what stays, without repainting it. Name the untargeted content by type, position and role rather than describing its appearance, and prefer one blanket preservation clause over walking the frame. A preservation description reads to the model as a generation instruction: the more concretely you describe something you meant to keep, the more likely it drifts. Describe appearance concretely only for what you are actually changing, or when it is the only way to disambiguate between similar objects.

Identity is the hardest invariant. A person's facial identity and the personal accessories that make them recognizable; a product's exact design, markings and count; and the input's rendering medium (photograph, anime, illustration, sketch, 3D render, painting) all survive every edit unless the user explicitly targets them. When identity comes from a reference image, point at that image rather than describing features in words - verbal descriptions make the model regenerate and degrade the likeness.

Resolve ambiguity, then commit. Turn vague intent, imprecise spatial reference and unparameterized style words into something concrete and observable. Translate abstract quality language into the visual properties it implies. Where the instruction offers alternatives or contradicts itself, pick the most reasonable reading and state it as a decision. Keep the user's own action verb, spatial relations and described state intact, and treat anything they asked to preserve as absolute. Preserve creative or physically impossible intent rather than correcting it.

Only what was asked. Do not add operations the user did not request, and do not clean up unmentioned defects, overlays or clutter however prominent they look. When an edit removes, moves or reveals something, say enough about the newly exposed region that the result stays physically coherent.

Text in the image is literal. Whenever readable text will appear in the output, commit to the exact characters - every element, quoted, nothing summarized or abbreviated away. Text you cannot commit to should not be added at all. Match the typography and language the input establishes unless the user asks otherwise. When the operation extends the canvas outward, name it as outpainting explicitly.

Write it as an instruction. Lead with the operation, not a description of the finished picture, and write from the perspective of someone holding only the input image(s).

## Thinking Process

Before emitting JSON, reason through: what the image(s) actually contain (including a complete reading of any text present); what the user is asking for and which attributes that names; what must therefore stay fixed; the output size; and finally the composed directive. Close with a check that every visible element is either the target of the edit or covered by what stays fixed, that the requested change is unmistakable, that nothing outside the target was touched, and that every quoted string obeys language decision (B).

## Image Reference Rules

For Multi-Image Input (N >= 2), the rewritten instruction MUST use <image1>, <image2>, ... to refer to each input image. Do not use natural language references like "image 1", "the first image", or "image A". This tagging format is mandatory and non-negotiable. For single-image input (N = 1), do NOT use tags - refer to the image naturally ("the image", "in the image").

State each image's role explicitly - which one is the canvas whose composition and untargeted content survive, and which supply material to transfer - and say what is taken from each. For scene generation with no canvas (group photo and the like), all images serve as identity sources. Describe every referenced image individually; never compress several into a range or a group to avoid describing them one by one.

## Output Size Determination

You must determine two output fields: wh_ratio and ratio_follow. These two fields are mutually exclusive - when one has a value, the other must be empty string "".

### Step 1: Check if the user explicitly specified a size or aspect ratio

Look for any of the following in the user's edit instruction:
- Exact pixel dimensions: "1920x1080", "800x600", "1080p"
- Aspect ratios: "16:9", "4:3", "3:2", "9:16", "1:1"
- Descriptive terms mapped to aspect ratios:
  - "square" / "avatar" / "profile picture" / "album cover" -> "1:1"
  - "landscape" / "desktop wallpaper" / "widescreen" / "video thumbnail" / "slide" / "presentation" -> "16:9"
  - "portrait" / "phone wallpaper" / "phone screen" / "Instagram story" / "Stories" / "Reels" / "short video cover" -> "9:16"
  - "full phone screen" / "iPhone screen" -> "18:39"
  - "Android screen" -> "9:20"
  - "ultrawide" -> "7:3"
  - "cinematic" / "movie ratio" / "cinemascope" -> "21:9"
  - "poster" -> "2:3"
  - "ID photo" / "passport photo" -> "3:4"
  - "tablet" / "iPad screen" -> "4:3"
  - "panoramic" / "panorama" -> "2:1"
  - "business card" -> "9:5"
  - "A4" -> "5:7" (vertical) or "7:5" (horizontal)
  - "1080p" / "720p" -> "16:9"

High-resolution keywords ("2K", "4K", "8K") are quality descriptors, NOT aspect ratio indicators. When the user mentions "2K", "4K", or "8K", these only express a desire for high image quality. They must NOT be used to infer or determine the aspect ratio. The aspect ratio should still be determined by other explicit cues or by the input image's ratio.

If the user specified a size or ratio:
-> wh_ratio = the corresponding ratio (e.g., "16:9", "1:1", "3:2")
-> ratio_follow = ""

If the user specified exact pixel dimensions (e.g., "1920x1080"), convert to the simplest integer ratio (1920:1080 = 16:9).

### Step 2: If the user did NOT specify any size or ratio

#### Single-image editing (1 input image):
The output should follow the input image's resolution.
-> wh_ratio = ""
-> ratio_follow = "<image1>"

Exception - Single-image scene generation: If the task generates a new scene from scratch using the input image only as an identity reference (e.g., "photo shoot", "cosplay as X", "travel to ancient times"), do NOT follow the input image's ratio - the output is a new composition, not an edit of the existing image. Instead, choose wh_ratio by scene semantics:
- Portrait / photo shoot / half-body: "2:3"
- Full-body scene / outdoor activity: "3:4"
- Landscape-oriented scene: "3:2"
- No clear orientation hint: Follow the input image's ratio (set ratio_follow to "<image1>", wh_ratio to "")

#### Multi-image editing (N >= 2 input images):
You must identify the canvas image (the image whose composition and framing the output should follow), then set ratio_follow to that image's tag.
- Compositing - transfer subject into a scene ("composite A into B", "place into", "add to"): The target scene image -> "<imageX>" (scene image number)
- Face/head swap ("swap face", "swap head"): The body image -> "<imageX>" (body image number)
- Clothing swap ("swap clothes", "change outfit"): The person image -> "<imageX>" (person image number)
- Style transfer ("draw in style of X", "style transfer"): The content image (not the style reference) -> "<imageX>" (content image number)
- Background replacement: The foreground subject image -> "<imageX>" (subject image number)
- Local object replacement: The original image being edited -> "<imageX>" (original image number)
- Scene generation - no canvas ("group photo", "eat together", "age together"): No canvas - choose a ratio:
  - Group photo: "3:2"
  - Portrait / photo shoot: "2:3"
  - Poster: "2:3"
  - Desktop wallpaper: "16:9"
  - Phone wallpaper: "9:16"
  - No clear orientation hint: Follow the last input image's ratio (set ratio_follow to the last image, wh_ratio to "")

#### Outpainting (extend image):
For outpainting tasks where the user did NOT specify a target aspect ratio, infer the new ratio from the extension direction:
- Extend right only or left only: widen the ratio (e.g., 1:1 input -> "3:2"; 3:4 input -> "1:1" or "4:3").
- Extend both left and right: widen more aggressively (e.g., 1:1 input -> "16:9" or "2:1").
- Extend down only or up only: make the ratio taller (e.g., 1:1 input -> "2:3"; 16:9 input -> "4:3" or "1:1").
- Extend both up and down: make the ratio significantly taller (e.g., 1:1 input -> "9:16").
- Extend all sides: keep the original ratio (the image grows uniformly).
Set ratio_follow = "" and wh_ratio = the inferred ratio.

#### Panoramic generation:
- Standard panorama: "2:1"
- Wide panorama: "3:1"
- 360 / VR panorama: "2:1"
- User specified a different ratio: Use user's ratio.
Set ratio_follow = "".

#### Three-view drawings and multi-grid generation:
- Three side-by-side views of a standing person: "1:1"
- Three side-by-side views of a car: "3:1" or "9:2"
- 2x2 grid of a square object: "1:1"
- 3x3 grid of square panels: "1:1"
Set ratio_follow = "" and wh_ratio = the adaptively determined ratio.

## Output Format
Output a valid JSON object with exactly three fields:
```json
{
  "rewritten_prompt": "Transform the daytime sky into a clear starry night sky filled with luminous constellations, a glowing crescent moon, and the subtle deep indigo glow of the Milky Way, while keeping the terrain, foreground trees, and ground lighting completely unchanged.",
  "wh_ratio": "",
  "ratio_follow": "<image1>"
}
```

rewritten_prompt formatting rules:
- The entire rewritten prompt must be a single continuous paragraph with NO line breaks or newline characters.
- All text that should appear as visible, readable content in the output image must be enclosed in double quotes (""). Descriptive or structural language that does not appear as rendered text should NOT be quoted.
- Never include any resolution or aspect ratio information in rewritten_prompt (e.g., "2:3", "16:9", "1920x1080", "2K", "4K").
- Write it out in full - no ellipsis, no truncation, and never output placeholder text or angle brackets.
- State requirements affirmatively ("keep background identical to input image") rather than as prohibitions ("do not change background"). Standard preservation phrasing "keep/preserve [X] unchanged" is fine.
- Be precise and decisive: no hedging, no unresolved alternatives.
- Language-purge self-check: re-scan every double-quoted string - the text that will be RENDERED in the image - and enforce language decision (B). No quoted string may mix languages or carry parenthetical translation glosses.

Rules for each field:
- rewritten_prompt: The rewritten editing instruction in English prose. Retain proper nouns and domain-specific terms in English double quotes.
- wh_ratio: The target aspect ratio as "W:H". Set to "" when the output resolution should follow an input image instead.
- ratio_follow: Which input image's resolution the output should follow ("<image1>", "<image2>", ...). Set to "" when a specific aspect ratio is provided in wh_ratio.

Mutual exclusivity rule:
- If wh_ratio has a value -> ratio_follow must be ""
- If ratio_follow is "<imageX>" -> wh_ratio must be ""

Do not include any text outside the JSON object - no greetings, no explanations, no markdown code fences."""


SYSTEM_PROMPT_T2I_EXPANSION = """# Text-to-Image Prompt Enhancer (v2, High-Fidelity Visual Expansion)

You are an expert prompt engineer for cutting-edge text-to-image diffusion and autoregressive models (such as Qwen-Image 2.1, Flux, and Wan 2.1).

Given a concise, vague, or short user prompt, expand it into a rich, vivid, professional photographic or artistic description written in natural English prose.

## Core Rules:
1. Always write the expanded prompt in clear, evocative English prose as a single continuous paragraph.
2. Structure the description to thoroughly establish:
   - Primary Subject: Specific anatomy, posture, clothing, expression, materials, and textures.
   - Environment and Background: Spatial depth, architectural or natural surroundings, atmosphere, and weather.
   - Lighting and Cinematography: Lighting direction, color temperature, shadows, camera focal length, depth of field, and angle.
   - Artistic Style and Medium: CRITICAL - You MUST strictly preserve and enhance the user's requested medium (e.g., 3D Render, Pixar, Anime, Oil Painting, Sketch, Digital Art). Do NOT force non-photographic prompts into photography. If the user asks for "3D" or "Pixar", describe it as a pristine 3D render, NOT a 35mm photograph.
3. Text Rendering: If the user explicitly asks for specific text, logos, or signage to appear inside the image, enclose the exact text in double quotes ("").
4. Aspect Ratio Determination:
   - Determine the most visually compelling aspect ratio (wh_ratio) based on the subject:
     - Portraits / single human figures: "2:3" or "3:4"
     - Landscapes, cinematic scenes, wildlife in habitat: "16:9" or "21:9"
     - Still life, macro, symmetrical logos: "1:1"
     - Mobile phone wallpaper / vertical posters: "9:16"

## Worked Example:
Input: "a lone astronaut on mars at sunset"
Output:
```json
{
  "rewritten_prompt": "A lone astronaut in a weathered white and gold exploration spacesuit stands on the crest of a rust-red Martian sand dune, boots sunk slightly into the fine iron-rich dust. In the background, towering jagged basalt formations and distant crater rims stretch toward the horizon beneath a pale butterscotch sky. A dramatic blue-tinted sunset casts long sharp shadows across the terracotta terrain, with warm golden rays glinting off the reflective gold visor of the helmet. The scene is captured in cinematic 35mm photography with a 50mm lens at f/2.8, sharply focusing on the fabric textures and mechanical dials of the suit while rendering the alien desert in soft atmospheric haze, evoking intense solitude and epic discovery.",
  "wh_ratio": "16:9",
  "ratio_follow": ""
}
```

## Mandatory Generation Rules:
- You must write a complete original description (100 to 250 words) in 'rewritten_prompt' describing the user's specific subject in full. Never output placeholder words, template brackets, or copy the example.
- Output ONLY the raw JSON object containing 'rewritten_prompt', 'wh_ratio', and 'ratio_follow'.
- Do not include markdown code fences or explanatory conversational text."""


# ====================================================================================================
# MAIN NODE CLASS: RT_QwenImagePromptEnhancer
# ====================================================================================================
class RT_QwenImagePromptEnhancer:
    """
    RT Qwen Image 2.1 Prompt Enhancer:
    * 100% written from scratch with zero external node dependencies.
    * Faithfully implements the 'Edit Prompt Enhancer - General (v2, Concise Edition)' system prompt.
    * Supports both Image-to-Image Edit Instruction Rewriting and Text-to-Image Prompt Expansion.
    * Multi-image batch support: automatically formats image batches as <image1>, <image2>, ...
    * Strict Attribute Disentanglement: preserves identity, untouched objects, and rendering style.
    * Intelligent Aspect Ratio Resolution: evaluates user intent or image semantics with mutual exclusivity.
    * Bulletproof JSON recovery parser: extracts continuous single-line prompts and outputs clean JSON.
    """

    @staticmethod
    def get_supported_models():
        unique_models = set()
        search_dirs = []
        for folder in ["clip", "text_encoders", "llm", "unet"]:
            if folder in folder_paths.folder_names_and_paths:
                search_dirs.extend(folder_paths.get_folder_paths(folder))

        for base_path in search_dirs:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.lower().endswith((".gguf", ".safetensors")):
                            rel_path = os.path.relpath(os.path.join(root, file), base_path)
                            rel_path = rel_path.replace("\\", "/")
                            unique_models.add(rel_path)

        if not unique_models:
            return ["No supported models (.gguf or .safetensors) found in clip/text_encoders/llm"]
        return sorted(list(unique_models))

    @classmethod
    def INPUT_TYPES(s):
        valid_models = s.get_supported_models()
        return {
            "required": {
                "llm_model": (valid_models, {
                    "default": valid_models[0],
                    "tooltip": "Select a GGUF text model or multimodal base model."
                }),
                "vision_model": (valid_models, {
                    "default": valid_models[0],
                    "tooltip": "Select a GGUF vision projector / mmproj model (required when input images are used)."
                }),
                "user_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter editing instruction (e.g. 'swap background to sunny beach', 'change shirt to leather jacket') or prompt to expand..."
                }),
                "mode": ([
                    "01. Edit Prompt Enhancer (Image-to-Image / Multi-Image)",
                    "02. Text-to-Image Prompt Enhancer (T2I Expansion)"
                ], {
                    "default": "01. Edit Prompt Enhancer (Image-to-Image / Multi-Image)",
                    "tooltip": "Choose between Image Editing Instruction Rewriter (strict attribute disentanglement) or Text-to-Image Prompt Expansion."
                }),
                "thinking_mode": ([
                    "Disabled (Fast / Direct JSON - Recommended)",
                    "Enabled (Deep Reasoning / Verbose Thoughts)"
                ], {
                    "default": "Disabled (Fast / Direct JSON - Recommended)",
                    "tooltip": "Disable internal <think> reasoning for instant generation (3-5 seconds) and to avoid token budget truncation. Enable only if you want full chain-of-thought analysis."
                }),
                "image_resolution": ([
                    "1.0 Megapixel (1008x1008 - Ultra Detail, Recommended)",
                    "0.6 Megapixel (784x784 - High Fidelity)",
                    "0.3 Megapixel (560x560 - Balanced Multi-Image)",
                    "0.15 Megapixel (392x392 - Low VRAM)",
                    "Auto (Smart Budget by Image Count)"
                ], {
                    "default": "1.0 Megapixel (1008x1008 - Ultra Detail, Recommended)",
                    "tooltip": "Visual resolution for reference images. 1.0 MP (1008x1008) provides ~1296 tokens per image, preserving small text, facial features, and subtle textures. 28-pixel patch alignment is strictly enforced."
                }),
                "max_tokens": (["512", "1024", "1536", "2048", "3072", "4096", "6144", "8192"], {
                    "default": "4096",
                    "tooltip": "Maximum output tokens to generate. Default 4096 provides ample headroom for reasoning models."
                }),
                "creativity": ([
                    "0.6 - Literal / Exact",
                    "0.8 - Balanced (Recommended)",
                    "1.0 - Creative",
                    "1.2 - Artistic"
                ], {
                    "default": "0.8 - Balanced (Recommended)",
                    "tooltip": "Sampling temperature. Lower is more literal; higher is more elaborative."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible generation (-1 for random)."
                }),
                "n_ctx": ("INT", {
                    "default": 8192,
                    "min": 2048,
                    "max": 32768,
                    "tooltip": "Context window size in tokens."
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 128,
                    "tooltip": "Number of model layers to offload to GPU VRAM (-1 = all layers to GPU, 0 = CPU only)."
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model loaded in VRAM across workflow runs to avoid reloading latency."
                }),
                "debug_console": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print detailed progress and diagnostic metrics to the terminal console."
                }),
            },
            "optional": {
                "image_1": ("IMAGE", {
                    "tooltip": "Reference image 1 (<image1>). Primary canvas, target scene, or main subject."
                }),
                "image_2": ("IMAGE", {
                    "tooltip": "Reference image 2 (<image2>). Secondary reference, style, face, clothing, or background."
                }),
                "image_3": ("IMAGE", {
                    "tooltip": "Reference image 3 (<image3>)."
                }),
                "image_4": ("IMAGE", {
                    "tooltip": "Reference image 4 (<image4>)."
                }),
                "image_5": ("IMAGE", {
                    "tooltip": "Reference image 5 (<image5>)."
                }),
                "image_6": ("IMAGE", {
                    "tooltip": "Reference image 6 (<image6>)."
                }),
                "image_7": ("IMAGE", {
                    "tooltip": "Reference image 7 (<image7>)."
                }),
                "image_8": ("IMAGE", {
                    "tooltip": "Reference image 8 (<image8>)."
                }),
                "image_9": ("IMAGE", {
                    "tooltip": "Reference image 9 (<image9>)."
                }),
                "image_10": ("IMAGE", {
                    "tooltip": "Reference image 10 (<image10>)."
                }),
                "image": ("IMAGE", {
                    "tooltip": "Alternative single image or batched image input."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "json_output", "wh_ratio", "ratio_follow", "diagnostics")
    FUNCTION = "enhance_prompt"
    CATEGORY = "RareTutor/Prompt"
    TITLE = "RT Qwen Image 2.1 Prompt Enhancer"

    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.loaded_model_path = None
        self.loaded_vision_path = None
        self.loaded_n_ctx = None
        self.loaded_n_gpu_layers = None
        self.hardware_info = {}

    def _resize_to_patch_grid(self, pil_img, total_images=1, resolution_mode="1.0 Megapixel"):
        """
        Automatically resizes reference images maintaining original aspect ratio.
        - Patch Grid Alignment: Width and height MUST be multiples of 28 (14x14 patch * 2x2 merge),
          strictly required by Qwen2-VL / Qwen2.5-VL / Qwen-Image-PE vision projectors.
        - High-Fidelity Multi-Resolution Control:
            * 1.0 MP: 1008x1008 (~1,016,064 px, ~1296 tokens) - preserves fine text, logos, facial details
            * 0.6 MP: 784x784 (~614,656 px, ~784 tokens) - high fidelity
            * 0.3 MP: 560x560 (~313,600 px, ~400 tokens) - balanced
            * 0.15 MP: 392x392 (~153,664 px, ~196 tokens) - low memory
            * Auto: scales dynamically based on total connected images
        """
        orig_w, orig_h = pil_img.size
        if orig_w <= 0 or orig_h <= 0:
            return pil_img

        mode_lower = str(resolution_mode).lower()
        if "1.0" in mode_lower or "ultra" in mode_lower:
            target_pixels = 1008 * 1008
        elif "0.6" in mode_lower:
            target_pixels = 784 * 784
        elif "0.3" in mode_lower:
            target_pixels = 560 * 560
        elif "0.15" in mode_lower or "low" in mode_lower:
            target_pixels = 392 * 392
        else:
            # Auto smart budgeting based on count
            if total_images <= 1:
                target_pixels = 1008 * 1008
            elif total_images <= 2:
                target_pixels = 784 * 784
            elif total_images <= 4:
                target_pixels = 560 * 560
            else:
                target_pixels = 392 * 392

        current_pixels = orig_w * orig_h
        scale = (target_pixels / float(current_pixels)) ** 0.5

        # Align strictly to multiples of 28 (mandatory for Qwen2.5-VL patch geometry)
        new_w = max(28, int(round((orig_w * scale) / 28.0)) * 28)
        new_h = max(28, int(round((orig_h * scale) / 28.0)) * 28)

        return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _resize_to_1mp(self, pil_img, target_pixels=1008 * 1008):
        return self._resize_to_patch_grid(pil_img, total_images=1, resolution_mode="1.0 Megapixel")

    def _tensor_batch_to_base64_list(self, image_tensor, total_images=1, resolution_mode="1.0 Megapixel"):
        """Converts an image tensor [B, H, W, C] to a list of base64 JPEG data URI strings with patch alignment."""
        if image_tensor is None:
            return []
        
        # Ensure 4D shape [B, H, W, C]
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        base64_list = []
        batch_size = image_tensor.shape[0]
        
        for i in range(batch_size):
            img_slice = image_tensor[i].detach().cpu().numpy()
            img_slice = (img_slice * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_slice)
            
            # Resize with 28-pixel patch alignment and user-specified resolution
            pil_img = self._resize_to_patch_grid(pil_img, total_images=total_images, resolution_mode=resolution_mode)
            
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=92)
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_list.append(f"data:image/jpeg;base64,{encoded}")
            
        return base64_list

    def _find_absolute_path(self, filename, search_dirs):
        for base_path in search_dirs:
            direct_path = os.path.join(base_path, filename)
            if os.path.exists(direct_path):
                return direct_path
        for base_path in search_dirs:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        if file == filename or full_path.replace("\\", "/").endswith(filename.replace("\\", "/")):
                            return full_path
        return None

    def _create_multimodal_handler(self, vision_path, llm_name):
        """
        Dynamically selects and instantiates the optimal multimodal chat handler.
        Fully compatible with JamePeng/llama-cpp-python's MTMD architecture:
        - Dedicated Qwen-VL handlers (Qwen25VLChatHandler, Qwen2VLChatHandler)
        - GenericMTMDChatHandler (JamePeng Jinja2 template-driven MTMD engine)
        - Llava16ChatHandler / Llava15ChatHandler (Classic fallbacks)
        Handles differing constructor signatures across llama-cpp-python versions.
        """
        try:
            import llama_cpp.llama_chat_format as chat_formats
        except ImportError:
            raise RuntimeError("Failed to import 'llama_cpp.llama_chat_format'.")

        model_lower = os.path.basename(llm_name).lower()
        handler_instance = None
        handler_name_used = None

        def try_instantiate(cls, is_generic=False):
            # Attempt 1: JamePeng v0.3.48+ positional signature: (None, mmproj_path, verbose=False)
            if is_generic:
                try:
                    return cls(None, vision_path, verbose=False)
                except Exception:
                    pass
                try:
                    return cls(chat_format=None, mmproj_path=vision_path, verbose=False)
                except Exception:
                    pass

            # Attempt 2: mmproj_path keyword argument
            try:
                return cls(mmproj_path=vision_path, verbose=False)
            except Exception:
                pass
            try:
                return cls(mmproj_path=vision_path)
            except Exception:
                pass

            # Attempt 3: clip_model_path keyword argument
            try:
                return cls(mmproj_path=vision_path, verbose=False)
            except Exception:
                pass
            try:
                return cls(mmproj_path=vision_path)
            except Exception:
                pass

            # Attempt 4: single positional argument
            try:
                return cls(vision_path)
            except Exception:
                pass
            return None

        # Priority 1: If model is Qwen, try dedicated Qwen vision handlers first
        if "qwen" in model_lower:
            for qwen_cls_name in ["Qwen25VLChatHandler", "Qwen2VLChatHandler", "QwenVLChatHandler"]:
                if hasattr(chat_formats, qwen_cls_name):
                    cls = getattr(chat_formats, qwen_cls_name)
                    handler_instance = try_instantiate(cls)
                    if handler_instance is not None:
                        handler_name_used = qwen_cls_name
                        break

        # Priority 2: GenericMTMDChatHandler (JamePeng template-driven MTMD engine)
        if handler_instance is None and hasattr(chat_formats, "GenericMTMDChatHandler"):
            cls = getattr(chat_formats, "GenericMTMDChatHandler")
            handler_instance = try_instantiate(cls, is_generic=True)
            if handler_instance is not None:
                handler_name_used = "GenericMTMDChatHandler (JamePeng MTMD)"

        # Priority 3: Llava16ChatHandler
        if handler_instance is None and hasattr(chat_formats, "Llava16ChatHandler"):
            cls = getattr(chat_formats, "Llava16ChatHandler")
            handler_instance = try_instantiate(cls)
            if handler_instance is not None:
                handler_name_used = "Llava16ChatHandler"

        # Priority 4: Llava15ChatHandler (Universal fallback)
        if handler_instance is None and hasattr(chat_formats, "Llava15ChatHandler"):
            cls = getattr(chat_formats, "Llava15ChatHandler")
            handler_instance = try_instantiate(cls)
            if handler_instance is not None:
                handler_name_used = "Llava15ChatHandler"

        if handler_instance is None:
            raise RuntimeError(
                f"Could not initialize a vision chat handler for '{llm_name}'. "
                f"Please verify that 'llama-cpp-python' has vision support and the mmproj model is valid."
            )

        print(f"[RT-Qwen-Image] Multimodal Chat Handler: {handler_name_used}")
        return handler_instance

    def load_model(self, llm_name, vision_name, n_ctx, n_gpu_layers, has_image):
        if llm_name.lower().endswith(".safetensors") or vision_name.lower().endswith(".safetensors"):
            raise ValueError("SAFETENSORS_ERROR")

        search_dirs = []
        for folder in ["clip", "text_encoders", "llm", "unet"]:
            if folder in folder_paths.folder_names_and_paths:
                search_dirs.extend(folder_paths.get_folder_paths(folder))

        llm_path = self._find_absolute_path(llm_name, search_dirs)
        vision_path = self._find_absolute_path(vision_name, search_dirs)

        if not llm_path:
            raise FileNotFoundError(f"LLM model '{llm_name}' not found.")
        if has_image and not vision_path:
            raise FileNotFoundError(f"Vision projector '{vision_name}' not found.")

        active_vision_path = vision_path if has_image else None
        if (self.llm is not None and
            self.loaded_model_path == llm_path and
            self.loaded_vision_path == active_vision_path and
            self.loaded_n_ctx == n_ctx and
            self.loaded_n_gpu_layers == n_gpu_layers):
            return

        self.unload_model()

        model_lower = llm_name.lower()
        if "qwen" in model_lower:
            detected_chat_format = "chatml"
            model_family = "Qwen (ChatML)"
        elif "gemma" in model_lower:
            detected_chat_format = "gemma"
            model_family = "Gemma"
        else:
            detected_chat_format = None
            model_family = "Auto (Jinja2 / Model-Native)"

        # Check GPU offload support in llama-cpp-python
        gpu_supported = False
        try:
            if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                gpu_supported = llama_cpp.llama_supports_gpu_offload()
            
            # Some newer custom wheels (like JamePeng 0.4.x) might return False here
            # despite having CUDA compiled in. Fall back to PyTorch's detection.
            if not gpu_supported and torch.cuda.is_available():
                gpu_supported = True
        except Exception:
            gpu_supported = False

        device_name = "N/A"
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:
                device_name = "CUDA Device"

        self.hardware_info = {
            "gpu_supported": gpu_supported,
            "cuda_available": cuda_available,
            "device_name": device_name,
            "n_gpu_layers": n_gpu_layers
        }

        vision_status = "Enabled" if has_image else "Disabled (Text Only)"
        layer_disp = "All Layers (-1)" if n_gpu_layers == -1 else f"{n_gpu_layers} Layers"

        if not gpu_supported:
            print("\n[RT-Qwen-Image] ========================================================")
            print("[RT-Qwen-Image] WARNING: Installed llama-cpp-python is a CPU-ONLY build!")
            print("[RT-Qwen-Image] 'llama_supports_gpu_offload()' returned False.")
            if cuda_available:
                print(f"[RT-Qwen-Image] GPU detected by PyTorch: {device_name}")
                print("[RT-Qwen-Image] Notice: PyTorch detects CUDA, but llama-cpp-python was installed without CUDA acceleration.")
            print("[RT-Qwen-Image] Model will execute on CPU until a CUDA-enabled wheel is installed.")
            print("[RT-Qwen-Image] ========================================================\n")
        else:
            print(f"[RT-Qwen-Image] Hardware Backend: CUDA / GPU Accelerated ({device_name})")
            print(f"[RT-Qwen-Image] GPU Offload Target: {layer_disp}")

        print(f"[RT-Qwen-Image] Loading Models... [Vision: {vision_status}] [Format: {model_family}] [Context: {n_ctx}]")

        try:
            if has_image:
                self.chat_handler = self._create_multimodal_handler(vision_path, llm_name)
            else:
                self.chat_handler = None

            is_legacy_llava = self.chat_handler is not None and "llava" in type(self.chat_handler).__name__.lower()

            n_batch_val = min(max(n_ctx, 2048), 4096)
            n_ubatch_val = min(n_batch_val, 2048)

            effective_gpu_layers = n_gpu_layers if gpu_supported else 0

            llama_kwargs = {
                "model_path": llm_path,
                "chat_handler": self.chat_handler,
                "n_gpu_layers": effective_gpu_layers,
                "n_ctx": n_ctx,
                "n_batch": n_batch_val,
                "n_ubatch": n_ubatch_val,
                "main_gpu": 0,
                "logits_all": True if is_legacy_llava else False,
                "verbose": False
            }

            if detected_chat_format and not has_image:
                llama_kwargs["chat_format"] = detected_chat_format

            # Try loading with Flash Attention if supported for acceleration
            try:
                llama_kwargs["flash_attn"] = True
                self.llm = Llama(**llama_kwargs)
            except (TypeError, ValueError):
                llama_kwargs.pop("flash_attn", None)
                try:
                    self.llm = Llama(**llama_kwargs)
                except TypeError:
                    llama_kwargs.pop("n_ubatch", None)
                    llama_kwargs.pop("n_batch", None)
                    self.llm = Llama(**llama_kwargs)

            self.loaded_model_path = llm_path
            self.loaded_vision_path = active_vision_path
            self.loaded_n_ctx = n_ctx
            self.loaded_n_gpu_layers = n_gpu_layers
            print(f"[RT-Qwen-Image] Successfully loaded model: {os.path.basename(llm_path)}")

        except Exception as e:
            self.unload_model()
            raise RuntimeError(f"[RT-Qwen-Image] Failed to load model '{llm_name}': {e}")

    def unload_model(self):
        if self.llm is not None:
            del self.llm
            self.llm = None
        if self.chat_handler is not None:
            del self.chat_handler
            self.chat_handler = None
        self.loaded_model_path = None
        self.loaded_vision_path = None
        self.loaded_n_ctx = None
        self.loaded_n_gpu_layers = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _parse_bulletproof_json(self, raw_text, image_count):
        """
        Extracts and validates JSON output from the model.
        Features multi-layer fallback:
        1. Strips closed (<think>...</think>) and unclosed (<think>...) reasoning blocks.
        2. Detects if the prompt was drafted in the reasoning phase and rescues it if JSON was truncated.
        3. Parses JSON from markdown blocks or bare braces.
        4. Regex recovery for individual fields, supporting both closed and truncated strings.
        5. Flattens rewritten_prompt to a single continuous paragraph with NO line breaks.
        6. Enforces mutual exclusivity between wh_ratio and ratio_follow.
        """
        # 1. Strip reasoning/thinking tokens (both closed and unclosed tags)
        cleaned_text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
        
        # Handle unclosed <think> tag if model was truncated or omitted </think>
        if "<think>" in cleaned_text.lower():
            json_pos = cleaned_text.find("{")
            if json_pos != -1:
                cleaned_text = cleaned_text[json_pos:].strip()
            else:
                rp_pos = cleaned_text.lower().find('"rewritten_prompt"')
                if rp_pos != -1:
                    cleaned_text = "{" + cleaned_text[rp_pos:].strip()
                else:
                    cleaned_text = re.sub(r"<think>[\s\S]*$", "", cleaned_text, flags=re.IGNORECASE).strip()

        parsed_obj = None

        # 2. Look for JSON markdown block ```json ... ```
        json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned_text, flags=re.IGNORECASE)
        if json_match:
            try:
                parsed_obj = json.loads(json_match.group(1).strip())
            except Exception:
                pass

        # 3. Look for outermost bare braces { ... }
        if parsed_obj is None:
            brace_match = re.search(r"(\{[\s\S]*\})", cleaned_text)
            if brace_match:
                try:
                    parsed_obj = json.loads(brace_match.group(1).strip())
                except Exception:
                    pass

        # 4. Regex extraction fallback if JSON parsing failed
        rewritten_prompt = ""
        wh_ratio = ""
        ratio_follow = ""

        if isinstance(parsed_obj, dict):
            rewritten_prompt = str(parsed_obj.get("rewritten_prompt", "")).strip()
            wh_ratio = str(parsed_obj.get("wh_ratio", "")).strip()
            ratio_follow = str(parsed_obj.get("ratio_follow", "")).strip()
        else:
            # Field-by-field regex recovery (closed string first)
            rp_match = re.search(r'"rewritten_prompt"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned_text)
            if rp_match:
                try:
                    rewritten_prompt = bytes(rp_match.group(1), "utf-8").decode("unicode_escape").strip()
                except Exception:
                    rewritten_prompt = rp_match.group(1).strip()
            else:
                # Truncated string: match from "rewritten_prompt": " to end of line/string
                trunc_match = re.search(r'"rewritten_prompt"\s*:\s*"((?:\\.|[^"\\])*)$', cleaned_text)
                if trunc_match:
                    try:
                        rewritten_prompt = bytes(trunc_match.group(1), "utf-8").decode("unicode_escape").strip()
                    except Exception:
                        rewritten_prompt = trunc_match.group(1).strip()

            # If the model drafted a longer prose description in its thinking steps before getting truncated, rescue it
            draft_match = re.search(r'(?:\*Drafting prose:\*|Drafting prose:|Draft:)\s*(.+?)(?=\s*\d+\.\s*\*\*|\s*\d+\.|\Z)', raw_text, flags=re.DOTALL | re.IGNORECASE)
            if draft_match:
                draft_text = draft_match.group(1).strip()
                if len(draft_text) > len(rewritten_prompt):
                    rewritten_prompt = draft_text

            # If still empty, clean up conversational prefixes from cleaned_text
            if not rewritten_prompt:
                fallback_clean = re.sub(r"^(?:Enhanced|Output|Rewritten|Instruction|Result):\s*", "", cleaned_text, flags=re.IGNORECASE).strip()
                fallback_clean = re.sub(r"^Thinking Process:[\s\S]*?(?=[A-Z][a-z]|\Z)", "", fallback_clean, flags=re.IGNORECASE).strip()
                rewritten_prompt = fallback_clean

            wr_match = re.search(r'"wh_ratio"\s*:\s*"([^"]*)"', cleaned_text)
            if wr_match:
                wh_ratio = wr_match.group(1).strip()

            rf_match = re.search(r'"ratio_follow"\s*:\s*"([^"]*)"', cleaned_text)
            if rf_match:
                ratio_follow = rf_match.group(1).strip()

        def _rescue_from_reasoning(text):
            # Tier 1: Look for explicit double-quoted draft inside reasoning
            double_quote_match = re.search(
                r'(?:Let me design|Design|Drafting prose|Draft|Composed directive|Rewritten directive|Directive|Prompt draft):\s*"([^"]{50,})"',
                text,
                flags=re.DOTALL | re.IGNORECASE
            )
            if double_quote_match:
                return double_quote_match.group(1).strip()

            # Tier 2: Look for single-quoted draft (only if no internal apostrophe collision)
            single_quote_match = re.search(
                r"(?:Let me design|Design|Drafting prose|Draft|Composed directive|Rewritten directive|Directive|Prompt draft):\s*'([^'\n]{50,})'",
                text,
                flags=re.DOTALL | re.IGNORECASE
            )
            if single_quote_match:
                return single_quote_match.group(1).strip()

            # Tier 3: Look for explicit scene design / composition / drafting section stopping before reflection words
            design_match = re.search(
                r'(?:Let me design|Design|Drafting prose|Draft|Composed directive|Rewritten directive|Directive|Prompt draft):\s*(.+?)(?=(?:\n\s*(?:Hmm|Wait|But\s+wait|Note|Environment|Lighting|Preservation|Style|Forbidden|Spatial|Composition|Task|Output|```|\{)|\Z))',
                text,
                flags=re.DOTALL | re.IGNORECASE
            )
            main_block = design_match.group(1).strip() if design_match else ""
            if main_block.startswith('"') and '"' in main_block[1:]:
                main_block = main_block[1:main_block.find('"', 1)].strip()

            env_match = re.search(
                r'Environment:\s*(.+?)(?=(?:\n\s*(?:Hmm|Wait|Lighting|Preservation|Style|Let me design|Design|Spatial|Composition)|\Z))',
                text,
                flags=re.DOTALL | re.IGNORECASE
            )
            light_match = re.search(
                r'Lighting:\s*(.+?)(?=(?:\n\s*(?:Hmm|Wait|Preservation|Style|Environment|Let me design|Design)|\Z))',
                text,
                flags=re.DOTALL | re.IGNORECASE
            )

            parts = []
            if main_block:
                parts.append(main_block)
            if env_match:
                env_text = env_match.group(1).strip()
                if "forest path" not in main_block.lower() and "dirt path" not in main_block.lower():
                    parts.append("Environment: " + env_text)
                elif not main_block:
                    parts.append("Environment: " + env_text)
            if light_match:
                light_text = light_match.group(1).strip()
                if "sunlight" not in main_block.lower() and "lighting" not in main_block.lower():
                    parts.append("Lighting: " + light_text)

            if parts:
                return " ".join(parts)

            # Tier 4: Look for paragraphs containing image tags (<image1>, <image2>, etc.)
            paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 80]
            tag_paras = [
                p for p in paras
                if re.search(r"<image\d+>", p) and not p.startswith(("-", "*", "Input images", "Note:", "Task mode:", "Hmm", "Wait"))
            ]
            if tag_paras:
                res = max(tag_paras, key=len)
                if res.startswith('"') and '"' in res[1:]:
                    res = res[1:res.find('"', 1)].strip()
                return res

            # Tier 5: Longest substantial descriptive paragraph
            general_paras = [
                p for p in paras
                if not p.startswith(("{", "}", "<", "*", "-", "1.", "2.", "3.", "4.", "5.", "Thinking", "Input", "Hmm", "Wait"))
            ]
            if general_paras:
                return max(general_paras, key=len)

            return ""

        # Extra safety: detect and reject placeholder hallucinations or empty output
        is_placeholder = (
            "expanded prompt" in rewritten_prompt.lower() or
            "rewritten editing instruction" in rewritten_prompt.lower() or
            (rewritten_prompt.strip().startswith("<") and rewritten_prompt.strip().endswith(">")) or
            len(rewritten_prompt.strip()) == 0
        )
        if is_placeholder:
            rescued_text = _rescue_from_reasoning(raw_text)
            if rescued_text:
                rewritten_prompt = rescued_text

        # Extra safety: strip any residual thinking artifacts from rewritten_prompt
        rewritten_prompt = re.sub(r"<think>[\s\S]*?</think>", "", rewritten_prompt, flags=re.IGNORECASE).strip()
        rewritten_prompt = re.sub(r"^Thinking Process:[\s\S]*?(?=[A-Z][a-z]|\Z)", "", rewritten_prompt, flags=re.IGNORECASE).strip()

        # Normalize common unicode punctuation to standard ASCII
        rewritten_prompt = rewritten_prompt.replace('\u2014', ' - ').replace('\u2013', ' - ')
        rewritten_prompt = rewritten_prompt.replace('\u2018', "'").replace('\u2019', "'")
        rewritten_prompt = rewritten_prompt.replace('\u201c', '"').replace('\u201d', '"')
        rewritten_prompt = rewritten_prompt.replace('\u2026', '...')

        # 5. Enforce strict single-line continuous paragraph (NO line breaks)
        rewritten_prompt = re.sub(r"[\r\n]+", " ", rewritten_prompt).strip()
        rewritten_prompt = re.sub(r"\s{2,}", " ", rewritten_prompt)

        # 6. Enforce mutual exclusivity rule
        if wh_ratio and wh_ratio != '""':
            ratio_follow = ""
        elif ratio_follow and ratio_follow != '""':
            wh_ratio = ""
        else:
            # Default sizing logic if both are empty
            if image_count == 1:
                ratio_follow = "<image1>"
                wh_ratio = ""
            elif image_count > 1:
                ratio_follow = "<image1>"
                wh_ratio = ""
            else:
                wh_ratio = "16:9"
                ratio_follow = ""

        # Normalize empty representations
        if wh_ratio in ['""', "''", "none", "null"]:
            wh_ratio = ""
        if ratio_follow in ['""', "''", "none", "null"]:
            ratio_follow = ""

        final_json_obj = {
            "rewritten_prompt": rewritten_prompt,
            "wh_ratio": wh_ratio,
            "ratio_follow": ratio_follow
        }
        json_output_str = json.dumps(final_json_obj, indent=2, ensure_ascii=True)

        return rewritten_prompt, json_output_str, wh_ratio, ratio_follow

    def enhance_prompt(
        self, llm_model, vision_model, user_input, mode, thinking_mode="Disabled (Fast / Direct JSON - Recommended)",
        image_resolution="1.0 Megapixel (1008x1008 - Ultra Detail, Recommended)", max_tokens="4096",
        creativity="0.8 - Balanced (Recommended)", seed=-1, n_ctx=8192, n_gpu_layers=-1,
        keep_model_loaded=True, debug_console=True,
        image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
        image_6=None, image_7=None, image_8=None, image_9=None, image_10=None,
        image=None, **kwargs
    ):
        t_start = time.perf_counter()
        
        # Collect all connected images in order (supports 0 to 10 images with zero errors)
        raw_images = []
        for img in [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9, image_10]:
            if img is not None:
                raw_images.append(img)
        if not raw_images and image is not None:
            raw_images.append(image)

        # Count total slices across all connected images
        total_image_count = sum([img.shape[0] if img.ndim == 4 else 1 for img in raw_images])

        # Convert each image tensor to base64 JPEG with patch grid alignment and selected resolution
        base64_images = []
        for img_tensor in raw_images:
            base64_images.extend(self._tensor_batch_to_base64_list(
                img_tensor,
                total_images=total_image_count,
                resolution_mode=image_resolution
            ))

        image_count = len(base64_images)
        has_image = image_count > 0

        # Dynamic context window scaling:
        # Avoid "failed to find a memory slot" or context limit when multiple images are loaded at 1.0 MP
        mode_lower = str(image_resolution).lower()
        if "1.0" in mode_lower or "ultra" in mode_lower:
            est_tokens_per_img = 1296
        elif "0.6" in mode_lower:
            est_tokens_per_img = 784
        elif "0.3" in mode_lower:
            est_tokens_per_img = 400
        elif "0.15" in mode_lower:
            est_tokens_per_img = 196
        else:
            est_tokens_per_img = 1296 if total_image_count <= 1 else (784 if total_image_count <= 2 else (400 if total_image_count <= 4 else 196))

        token_val = int(max_tokens)
        required_ctx = (total_image_count * est_tokens_per_img) + token_val + 2048
        effective_n_ctx = max(int(n_ctx), required_ctx)
        effective_n_ctx = min(32768, int(((effective_n_ctx + 1023) // 1024) * 1024))

        if effective_n_ctx > n_ctx and debug_console:
            print(f"[RT-Qwen-Image] Auto-scaled context window from {n_ctx} to {effective_n_ctx} tokens for {total_image_count} image(s) at {image_resolution.split(' (')[0]}.")

        # 1. Load model with VRAM caching
        try:
            self.load_model(llm_model, vision_model, effective_n_ctx, n_gpu_layers, has_image)
        except Exception as e:
            err_msg = str(e)
            if "SAFETENSORS_ERROR" in err_msg:
                user_msg = "[ERROR] A .safetensors file was selected. Please select a standard .gguf LLM model."
            else:
                user_msg = f"[ERROR] Failed to load model '{llm_model}': {e}"
            print(f"\n[RT-Qwen-Image] {user_msg}\n")
            diag = (
                f"=== RT Qwen Image 2.1 Prompt Enhancer Diagnostics ===\n"
                f"{user_msg}\n\n"
                f"Troubleshooting Guide:\n"
                f"1. Model Compatibility: Make sure the selected model is a standard GGUF LLM (e.g. Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-3B, or Gemma-2) in standard quantization (Q4_K_M, Q5_K_M, Q8_0, Q6_K, FP16).\n"
                f"2. Unsupported Quantizations: Experimental 1-bit or ternary formats (such as PTQ1_0 / Ternary-Bonsai) and standalone CLIP text encoders are not supported by standard llama.cpp.\n"
                f"3. Vision Models: When connecting images, ensure 'vision_model' points to a matching mmproj GGUF file."
            )
            return (user_msg, "{}", "", "", diag)

        # 2. Select system prompt based on mode and thinking_mode
        is_edit_mode = "01. Edit" in mode
        system_prompt = SYSTEM_PROMPT_EDIT_ENHANCER if is_edit_mode else SYSTEM_PROMPT_T2I_EXPANSION
        is_direct_mode = "Disabled" in str(thinking_mode) or "Direct" in str(thinking_mode)

        if is_direct_mode:
            system_prompt = re.sub(
                r"## Thinking Process[\s\S]*?## Image Reference Rules",
                "## Direct Output Requirement\nCRITICAL: DO NOT emit <think> tags. Do NOT output internal reasoning, analysis, or monologue. Emit ONLY the raw JSON object starting immediately with '{'.\n\n## Image Reference Rules",
                system_prompt,
                flags=re.IGNORECASE
            )
            system_prompt = system_prompt.replace(
                "Output ONLY the raw JSON object containing 'rewritten_prompt', 'wh_ratio', and 'ratio_follow'.",
                "CRITICAL: DO NOT use <think> tags. Output ONLY the raw JSON object starting immediately with '{'."
            )

        # 3. Build user message context
        user_text = user_input.strip() if user_input and user_input.strip() else "Enhance and refine the visual composition."

        if is_edit_mode:
            if is_direct_mode:
                format_suffix = (
                    "\n\nTask: Rewrite this edit instruction into a precise directive. "
                    "CRITICAL: Output ONLY a valid JSON object starting immediately with '{'. "
                    "Do NOT output <think> tags, reasoning, or preamble. "
                    "Write your complete instructions in full prose inside 'rewritten_prompt'."
                )
            else:
                format_suffix = (
                    "\n\nTask: Rewrite this edit instruction into a precise directive. "
                    "Output ONLY a valid JSON object with 'rewritten_prompt', 'wh_ratio', 'ratio_follow'. "
                    "Write your complete instructions in full prose inside 'rewritten_prompt'. Never output template brackets."
                )
            if image_count == 0:
                annotated_instruction = (
                    f"Edit Instruction: {user_text}\n\n"
                    f"Note: No source image connected. Treat as an image synthesis instruction to clarify and specify."
                    f"{format_suffix}"
                )
            elif image_count == 1:
                annotated_instruction = (
                    f"Edit Instruction: {user_text}\n\n"
                    f"Note: Single input image provided. Refer to the image naturally without tag numbers."
                    f"{format_suffix}"
                )
            else:
                tag_list_str = ", ".join([f"<image{i+1}>" for i in range(image_count)])
                annotated_instruction = (
                    f"Edit Instruction: {user_text}\n\n"
                    f"Note: Multi-image input provided with {image_count} images: {tag_list_str}. "
                    f"You MUST use tags like <image1>, <image2> to refer to each image in the rewritten instruction."
                    f"{format_suffix}"
                )
        else:
            annotated_instruction = (
                f"User Prompt: {user_text}\n\n"
                f"Task: Expand this prompt into a rich, vivid, highly detailed description. "
                f"Output ONLY a valid JSON object with keys 'rewritten_prompt', 'wh_ratio', 'ratio_follow'. "
                f"Write your full original descriptive paragraph in 'rewritten_prompt'. Never output template brackets or placeholder text."
            )

        # 4. Construct chat messages (images first, then instruction text)
        if image_count > 0:
            user_content = []
            for b64 in base64_images:
                user_content.append({"type": "image_url", "image_url": {"url": b64}})
            user_content.append({"type": "text", "text": annotated_instruction})
        else:
            user_content = annotated_instruction

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # 5. Sampling parameters
        token_val = int(max_tokens)
        temp_val = float(creativity.split(" - ")[0])
        safe_seed = (int(seed) % (2 ** 32)) if seed != -1 else None

        # Stop tokens for Qwen and standard chat templates
        stop_tokens = [
            "<|im_end|>", "<|im_start|>", "<|endoftext|>",
            "<end_of_turn>", "<eos>", "<|eot_id|>", "USER:", "ASSISTANT:"
        ]

        # 6. Execute inference
        if debug_console:
            mode_name = "Edit Instruction Rewriter" if is_edit_mode else "T2I Prompt Expander"
            print(f"[RT-Qwen-Image] Generating prompt ({mode_name}) | Images: {image_count} | Temp: {temp_val}...")

        try:
            chat_kwargs = {
                "messages": messages,
                "max_tokens": token_val,
                "temperature": temp_val,
                "stop": stop_tokens,
                "seed": safe_seed
            }
            if is_direct_mode:
                try:
                    chat_kwargs["response_format"] = {"type": "json_object"}
                    response = self.llm.create_chat_completion(**chat_kwargs)
                except Exception:
                    chat_kwargs.pop("response_format", None)
                    response = self.llm.create_chat_completion(**chat_kwargs)
            else:
                response = self.llm.create_chat_completion(**chat_kwargs)
            raw_result = response["choices"][0]["message"]["content"].strip()

        except Exception as e:
            err = f"Inference Error: {e}"
            if not keep_model_loaded:
                self.unload_model()
            return (err, "{}", "", "", err)

        if not keep_model_loaded:
            self.unload_model()

        # 7. Parse bulletproof JSON output
        enhanced_prompt, json_output_str, wh_ratio, ratio_follow = self._parse_bulletproof_json(raw_result, image_count)

        duration = time.perf_counter() - t_start

        # 8. Construct comprehensive diagnostic report
        res_display = image_resolution.split(" (")[0]
        hw = getattr(self, "hardware_info", {})
        if hw.get("gpu_supported", False):
            backend_str = f"GPU / CUDA Accelerated ({hw.get('device_name', 'CUDA Device')})"
            layer_val = hw.get("n_gpu_layers", n_gpu_layers)
            layers_str = f"{layer_val} (All layers offloaded to GPU)" if layer_val == -1 else f"{layer_val} layers offloaded to GPU"
        else:
            cuda_dev = hw.get("device_name", "N/A")
            backend_str = f"CPU ONLY (llama-cpp-python lacks CUDA support | PyTorch GPU: {cuda_dev})"
            layers_str = "0 layers (CPU fallback - wheel needs CUDA support)"

        diagnostics = (
            f"=== RT Qwen Image 2.1 Prompt Enhancer Diagnostics ===\n"
            f"Mode: {'Edit Prompt Enhancer (Image-to-Image)' if is_edit_mode else 'Text-to-Image Prompt Enhancer'}\n"
            f"Thinking Mode: {'Disabled (Direct Fast JSON)' if is_direct_mode else 'Enabled (Deep Reasoning)'}\n"
            f"Hardware Backend: {backend_str}\n"
            f"GPU Layers Offloaded: {layers_str}\n"
            f"Images Ingested: {image_count}\n"
            f"Image Resolution: {res_display}\n"
            f"Context Window: {effective_n_ctx} tokens (Configured: {n_ctx})\n"
            f"Aspect Ratio (wh_ratio): '{wh_ratio}'\n"
            f"Follow Canvas (ratio_follow): '{ratio_follow}'\n"
            f"Execution Time: {duration:.2f}s\n"
            f"Seed: {safe_seed if safe_seed is not None else 'Random'}\n\n"
            f"--- Enhanced Single-Line Prompt ---\n{enhanced_prompt}\n\n"
            f"--- Raw Model Response ---\n{raw_result}"
        )

        if debug_console:
            print(f"[RT-Qwen-Image] Complete in {duration:.2f}s | wh_ratio: '{wh_ratio}' | ratio_follow: '{ratio_follow}'")
            print(f"[RT-Qwen-Image] Prompt: {enhanced_prompt[:120]}...\n")

        return (enhanced_prompt, json_output_str, wh_ratio, ratio_follow, diagnostics)


# ====================================================================================================
# NODE REGISTRATION MAPPINGS
# ====================================================================================================
NODE_CLASS_MAPPINGS = {
    "RT_QwenImagePromptEnhancer": RT_QwenImagePromptEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_QwenImagePromptEnhancer": "RT Qwen Image 2.1 Prompt Enhancer"
}

