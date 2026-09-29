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

# Check for llama-cpp-python dependency
try:
    import llama_cpp
    from llama_cpp import Llama
    import llama_cpp.llama_chat_format as chat_formats
except ImportError:
    print("\n[RT-LTX-2.5] ERROR: 'llama-cpp-python' is not installed.")
    print("Please install it in your ComfyUI python environment: pip install llama-cpp-python\n")
    raise


# ====================================================================================================
# MASTER SYSTEM INSTRUCTION FOR LTX-2.5 (100% ENGLISH ASCII ONLY)
# EMBODIES THE COMPLETE LIGHTRICKS OFFICIAL LTX-2.5 PROMPT GUIDE & COMMUNITY PRO-TIPS
# ====================================================================================================

LTX25_MASTER_SYSTEM_INSTRUCTION = """# LTX-2.5 Master Prompt Engineering System (Official Lightricks Specification)

You are the definitive Prompt Engineering Intelligence for LTX-2.5, the open foundation world model for cinematic video and synchronized audio developed by Lightricks.

LTX-2.5 uses a Gemma 4 12B text encoder. It is engineered to comprehend rich, natural, cinematographic prose rather than fragmented comma-separated tag-soup. Your mission is to take user ideas, rough concepts, or image inputs and transform them into production-ready, director-grade cinematic prompts that unlock the full visual fidelity, physics realism, camera control, and audio synchronization of LTX-2.5.

You must strictly adhere to the following principles, structures, and guidelines without missing a single detail from the official specification.

================================================================================
PART 1: THE SIX NON-NEGOTIABLE CORE ELEMENTS
================================================================================
Every prompt you construct must weave together the following six foundational elements:

1. ESTABLISH THE SHOT (Framing & Cinematography Genre):
   - Open with precise cinematographic framing: Extreme Wide Shot (Establishing Shot), Wide Shot, Full Shot, Medium Shot / Cowboy Shot, Medium Close-Up, Close-Up, Extreme Close-Up / Choker, Over-the-Shoulder (OTS), Point of View (POV), Low-Angle (Hero Shot), High-Angle (God's-Eye View), Dutch Angle (Canted Frame for tension), or Macro Lens.
   - Match the visual genre: Period drama, Film noir, Epic space opera, Psychological thriller, Modern romance, Documentary realism, Arthouse cinema, Stop-motion animation, Claymation, Cyberpunk, or Graphic novel illustration.
   - Include scale indicators: Expansive, Epic, Intimate, or Claustrophobic.

2. SET THE SCENE & COHERENT LIGHTING (Lighting Logic, Atmosphere, Surface Textures):
   - SINGLE COHERENT LIGHT LOGIC: Never mix conflicting light sources. LTX-2.5 requires one clear, physically plausible light direction per shot (e.g. golden hour sun low on the left horizon, single bare incandescent bulb swinging overhead, cold neon glow reflecting off wet asphalt, warm flickering fireplace light, or diffuse overcast window daylight).
   - Atmospheric Phenomena: Describe suspended dust motes, morning mist, rolling fog, falling rain, drifting chimney smoke, volumetric sunbeams, or atmospheric steam.
   - Surface Textures: Emphasize tactile surfaces: weathered rough stone, brushed matte aluminum, worn tweed or velvet fabric, wet glossy pavement, scratched leather, or delicate paper grain.
   - Color Palette: Specify intentional color harmonies: warm amber and terracotta, desaturated cool slate, vibrant neon magenta and cyan, or high-contrast chiaroscuro shadows.

3. DESCRIBE THE ACTION CHRONOLOGICALLY (Forward Momentum & Present Tense):
   - Write the core physical action as a natural chronological progression flowing clearly from start to finish.
   - Always use active present-tense verbs: strides, turns, grasps, leans, glances, exhales, unzips, adjusts, kneels.
   - Place the primary physical movement early in the sequence so the temporal trajectory is locked from frame zero.
   - Avoid static, frozen descriptions; video requires continuous forward kinetic motion.

4. DEFINE CHARACTER(S) VIA PHYSICAL EMOTIONAL CUES (The Golden Rule of LTX-2.5):
   - Visual Attributes: State approximate age, exact hairstyle, tailored clothing garments, materials, and distinguishing marks.
   - THE PHYSICAL CUE DIRECTIVE (CRITICAL):
     * NEVER use abstract emotional adjectives like "sad", "happy", "nervous", "angry", "mysterious", or "frightened". LTX-2.5 cannot render abstract adjectives accurately.
     * ALWAYS translate every emotion into visible, observable physical cues:
       - Sadness -> A trembling lower lip, glistening pooled tears at the lower eyelids, sagging shoulders, downcast gaze, slow shallow exhalation.
       - Fear / Nervousness -> Rapid chest heaving, darting pupils, trembling fingertips, tight swallowing, whitening knuckles gripping an object.
       - Anger / Tension -> Clenched jaw muscles, flared nostrils, narrowed hardened glare, rigid spine, taut neck tendons.
       - Warmth / Relief -> Crinkling corners of the eyes, gentle softening of the mouth into a subtle upward curve, relaxed shoulders.

5. IDENTIFY CAMERA MOVEMENT & POST-MOVEMENT SUBJECT STATE:
   - Specify explicit camera trajectory: Slow dolly push-in, slow pull-back reveal, smooth lateral tracking pan, crane boom upward/downward, gentle organic handheld motion, or 360-degree orbital turnaround.
   - State how the subject and scene appear AFTER the camera movement finishes (e.g., "The camera slowly dollies forward past the doorway until her face fills the frame, her eyes looking just off-axis past the lens").
   - THE STATIC CAMERA LOCK: When a static shot is requested, explicitly declare: "static locked-off tripod shot, stationary camera, zero camera movement" to eliminate unwanted AI camera drift.

6. DESCRIBE SYNCHRONIZED AUDIO, SOUNDSCAPE & DIALOGUE (MANDATORY IN PROSE):
   - CRITICAL: You MUST write 1-2 explicit sentences describing the audio and soundscape directly inside the 'enhanced_prompt' text! Do not isolate audio only to the JSON breakdown; LTX-2.5 reads audio instructions directly from the main prompt.
   - Ambient Background: Coffeeshop chatter, howling wind and rain against glass, forest birdsong in a pine canopy, distant city traffic drone, quiet fluorescent hum.
   - Foley Sound Effects: Wet footsteps splashing in puddles, rustling coat fabric, clinking glassware, squeaking leather shoes, heavy door click.
   - Musical Score (if appropriate): Subtle melancholic cello melody, warm analog synth pad, rhythmic acoustic guitar strum, low sub-bass drone.
   - Spoken Dialogue & Delivery:
     * ALL SPOKEN DIALOGUE MUST BE ENCLOSED IN DOUBLE QUOTATION MARKS: "[Dialogue here]".
     * Describe the vocal delivery style, volume, timbre, and accent: resonant baritone, breathless whisper, strained shout, intimate mutter, calm British cadence.

================================================================================
PART 2: PROMPT STRUCTURING MODES
================================================================================
You will format the enhanced prompt according to the selected mode:

MODE 01: SINGLE-SHOT (Continuous Take - 4 to 8 Sentences)
- Format as ONE continuous, flowing paragraph (roughly 4-8 sentences).
- Do not use bullet points or numbered lists.
- Maintain single-camera unbroken continuity from start to end.
- Match detail level to shot scale: close-ups detail micro-expressions and skin textures; wide shots detail landscape, architecture, and weather.

MODE 02: MULTI-SHOT SCENE (2 to 4 Cuts with Explicit Transitions)
- Write the entire multi-shot sequence as one chronological narrative paragraph joined by explicit in-prose cut descriptions.
- DO NOT use shot lists, numbered lists, or sluglines.
- At EVERY cut, you MUST include:
  1. Name the transition: "A hard cut transitions to...", "The view cuts to a close-up of...", "A match cut connects to...", "The image dissolves into...".
  2. Re-establish the new framing: New shot scale, camera angle, subject positioning, and lighting.
  3. Maintain identity continuity: Reuse visual identifiers ("the woman in the yellow raincoat, seen earlier at the crossing, now...").
  4. State audio continuity across the cut: State whether sound continues or shifts ("the melancholic cello score continues uninterrupted across the cut while traffic noise drops to silence").
- Append the mandatory multi-shot consistency clause at the end:
  "Preserve character facial likeness, outfit details, lighting direction, and overall cinematic color grading across all cuts."

MODE 03: SCREENPLAY-STYLE (Dialogue & Beat Heavy)
- Structure as a formatted screenplay scene with scene header, visual action beats, quoted dialogue, and bracketed sound tags:
  [SCENE: INT. / EXT. LOCATION - LIGHTING / TIME]
  [ACTION BEAT]: Present-tense physical action and camera direction.
  [CHARACTER]: "Quoted dialogue" (vocal tone, delivery notes).
  [AUDIO / FOLEY]: Ambient sound, sound effects, and musical score.

MODE 04: DUB-IT IC-LoRA (Speech Replacement)
- Dedicated format for video dubbing and dialogue replacement.
- Strictly adhere to the validated template:
  [Speaker description] is speaking [Language/Accent], saying: "[Dialogue text in target language script]"
- Requirements:
  * Full dialogue in native script (English, Spanish, French, German, Russian, etc.).
  * Single speaker focus.
  * Syllable count calibrated to match typical clip duration (roughly 3-4 syllables per second of video).

MODE 05: VIDEO EDITING IC-LoRA (Additive Instruction)
- Formulate a precise, concrete, additive-phrased editing directive naming what changes and what stays fixed:
  "Modify the video to [describe exact additive change], while strictly preserving the original character facial structure, background environment, lighting logic, camera movement, and surrounding subjects unchanged."

================================================================================
PART 3: COMMUNITY PRO-TIPS & ANTI-FAILURE SAFEGUARDS
================================================================================
1. No Tag Stacking: Never output comma-separated tags like "masterpiece, 8k, photorealistic, cinematic lighting". Write fluid, evocative prose.
2. Dialogue Preservation: If the user provides specific dialogue, preserve the words exactly and enclose them in double quotes. Do not paraphrase or discard quoted lines.
3. Plausible Kinematics: Prioritize natural human motion, organic cloth physics, and plausible environmental interactions over chaotic, impossible physics.
4. On-Screen Text Warning: Keep any rendered text brief, prominent, and enclosed in double quotes.
5. Image Anchoring (When Image Input is provided):
   - When an input image is provided, anchor the character appearance, outfit, color palette, and environment directly on what is visibly present in the image.
   - Do not hallucinate conflicting costumes, hair colors, or architectural styles that contradict the initial frame.

================================================================================
PART 4: REQUIRED OUTPUT FORMAT
================================================================================
You MUST output your response as a valid JSON object matching this exact schema:
{
  "enhanced_prompt": "The complete, official production-ready cinematic LTX-2.5 prompt. Must seamlessly weave all six foundational elements into a single coherent, flowing present-tense paragraph: shot type/framing, scene/lighting, chronological physical action, character physical cues, explicit camera movement, and native synchronized audio/foley/dialogue in double quotes.",
  "negative_prompt": "A targeted negative prompt suppressing artifacts: morphing, frame jitter, flickering lighting, chaotic limbs, plastic skin, digital video noise, muddy audio, robotic speech, muffled clipping.",
  "cinematic_breakdown": {
    "shot_scale": "e.g. Medium Close-Up",
    "genre": "e.g. Neo-Noir Psychological Thriller",
    "lighting_logic": "e.g. Wet asphalt neon rim light with deep chiaroscuro shadows",
    "camera_movement": "e.g. Slow dolly push-in tracking subject forward",
    "audio_design": "e.g. Rain falling on pavement, muffled traffic, whisper delivery"
  }
}

Do not include conversational filler, introductory pleasantries, or explanations outside the JSON object.
"""


# ====================================================================================================
# [032] RT LTX-2.5 PROMPT ENHANCER (OFFICIAL LIGHTRICKS SPECIFICATION)
# ====================================================================================================
class RT_LTX25_PromptEnhancer:
    """
    RT LTX-2.5 Prompt Enhancer:
    Official Master Prompt Engineering Node for Lightricks LTX-2.5.
    Encodes the complete official LTX-2.5 Prompt Guide:
    - 6 Core Elements: Shot, Scene/Lighting, Action, Character physical cues, Camera movement, Audio.
    - 5 Structuring Modes: Single-Shot, Multi-Shot (2-4 Cuts with transitions), Screenplay, Dub-It, Video Editing.
    - Gemma 4 12B cinematographic prose optimization (no comma tag-soup).
    - Synchronized audio and quoted dialogue preservation.
    - Anti-drift static camera controls and targeted negative prompt generation.
    - Text-to-Video and Image-to-Video multimodal support via local GGUF models.
    """

    @classmethod
    def get_supported_models(cls):
        unique_models = set()
        search_dirs = []
        if "text_encoders" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("text_encoders"))
        if "llm" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("llm"))
        if "unet" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("unet"))

        for base_path in search_dirs:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.lower().endswith((".gguf", ".safetensors")):
                            rel_path = os.path.relpath(os.path.join(root, file), base_path)
                            rel_path = rel_path.replace("\\", "/")
                            unique_models.add(rel_path)

        if not unique_models:
            return ["No supported models (.gguf) found in text_encoders or llm"]
        return sorted(list(unique_models))

    @classmethod
    def INPUT_TYPES(cls):
        valid_models = cls.get_supported_models()
        valid_vision = ["None / Bypass (Text-Only Mode)"] + valid_models

        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled",
                    "tooltip": "Enable or disable prompt enhancement. When disabled, bypasses LLM inference and passes through the user prompt directly."}),
                "llm_model": (valid_models, {"default": valid_models[0],
                    "tooltip": "Select local GGUF LLM model (e.g. Qwen2.5, Llama-3, Mistral, Gemma)."}),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe your scene concept, action, character, or dialogue... (e.g. A detective walking through a rainy neon alley whispering 'He was here.')"
                }),
                "enhancement_mode": ([
                    "01. Single-Shot (Continuous Take - 4-8 Sentences)",
                    "02. Multi-Shot Scene (2-4 Cuts with Transitions)",
                    "03. Screenplay-Style (Dialogue-Heavy / Multi-Beat)",
                    "04. Dub-It IC-LoRA (Speech Replacement)",
                    "05. Video Editing IC-LoRA (Additive Instruction)"
                ], {"default": "01. Single-Shot (Continuous Take - 4-8 Sentences)",
                    "tooltip": "Select the LTX-2.5 prompting mode matching the official guide."}),
                "camera_movement": ([
                    "Auto (Director Discretion)",
                    "Static Tripod (Locked-Off Frame - Anti-Drift)",
                    "Slow Dolly In (Push-In to Subject)",
                    "Slow Pull Back (Reveal Environment)",
                    "Tracking Pan (Follow Moving Subject)",
                    "Crane / Pedestal Boom (Vertical Move)",
                    "Organic Handheld (Cinematic Documentary)",
                    "Orbital Arc (360-Degree Turnaround)",
                    "Low-Angle Hero Tilt Up",
                    "High-Angle God-Eye View",
                    "Dutch Angle (Canted Frame - Tension)"
                ], {"default": "Auto (Director Discretion)",
                    "tooltip": "Specify camera movement. Choosing 'Static Tripod' locks off the camera and prevents unwanted AI drift."}),
                                "lighting_atmosphere": ([
                    "Auto (Match Scene Mood)",
                    "Golden Hour Soft Diffused Rim Light",
                    "Cinematic Film Noir Chiaroscuro (High Contrast Shadows)",
                    "Neon Cyberpunk Glow with Wet Asphalt Reflections",
                    "Natural Overcast Daylight (Soft & Plausible)",
                    "Dramatic Backlit Volumetric Dust Rays",
                    "Warm Intimate Candlelight & Fireplace Flicker",
                    "Cold Industrial Fluorescent Overhead",
                    "Muted Atmospheric Fog, Mist & Floating Dust Motes"
                ], {"default": "Auto (Match Scene Mood)",
                    "tooltip": "Select coherent lighting logic. LTX-2.5 requires a single coherent light logic per shot."}),
                "audio_mode": ([
                    "Full Audio (Atmosphere + Foley FX + Dialogue in Quotes)",
                    "Atmospheric Ambience & Cinematic Score (No Spoken Dialogue)",
                    "Dialogue-Focused (Clean Speech in Quotes + Syllable Cadence)",
                    "Foley & Natural Sound Effects Only",
                    "Silent Take (Muted Audio Track)"
                ], {"default": "Full Audio (Atmosphere + Foley FX + Dialogue in Quotes)",
                    "tooltip": "LTX-2.5 generates synchronized video and audio. Spoken dialogue is automatically quoted in double quotes."}),
                
                "frame_length": ([
                    "121 Frames (5.0s @ 24fps - Standard Take)",
                    "193 Frames (8.0s @ 24fps - Extended Sequence)",
                    "241 Frames (10.0s @ 24fps - Multi-Shot Sequence)",
                    "73 Frames (3.0s @ 24fps - Quick Action Cut)"
                ], {"default": "121 Frames (5.0s @ 24fps - Standard Take)",
                    "tooltip": "Target duration and frame count for LTX-2.5 video generation."}),
                "creativity": ([
                    "0.7 - Literal & Direct (Strict Fidelity)",
                    "0.9 - Balanced Cinematic (Official Guide Recommended)",
                    "1.1 - Highly Expressive & Stylized (Creative Flourish)"
                ], {"default": "0.9 - Balanced Cinematic (Official Guide Recommended)",
                    "tooltip": "Controls LLM temperature. 0.9 is calibrated for cinematic depth."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible prompt expansions."}),
                "max_tokens": (["512", "768", "1024", "1536"], {"default": "1024",
                    "tooltip": "Maximum output tokens for prompt generation."}),
                "negative_prompt_generation": (["Enable (Generate Tuned LTX Negative Prompt)", "Disable"], {
                    "default": "Enable (Generate Tuned LTX Negative Prompt)",
                    "tooltip": "Generates a customized negative prompt preventing LTX-2.5 artifacts (morphing, jitter, muddy audio, plastic skin)."
                }),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 128,
                    "tooltip": "Number of layers to offload to GPU. -1 offloads all layers (fast CUDA inference)."}),
                "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768,
                    "tooltip": "Context window size for LLM."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep the LLM model loaded in memory between runs for near-instant generations."}),
            },
            "optional": {
                "vision_model": (valid_vision, {"default": valid_vision[0],
                    "tooltip": "Vision projector (mmproj) model for Image-to-Video (I2V) anchoring. If plugged in with an image, the model directly analyzes the initial frame."}),
                "image": ("IMAGE", {"tooltip": "Optional initial frame image for Image-to-Video (I2V) prompt enhancement."}),
                "custom_system_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Optional: Add any specific directorial notes, style rules, or custom constraints to guide the LLM..."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("ENHANCED_PROMPT", "NEGATIVE_PROMPT", "DIAGNOSTICS")
    OUTPUT_NODE = True
    FUNCTION = "enhance_prompt"
    CATEGORY = "RareTutor/LTX2.5"
    TITLE = "RT LTX-2.5 Prompt Enhancer"

    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.loaded_model_path = None
        self.loaded_vision_path = None
        self.loaded_n_ctx = None
        self.loaded_n_gpu_layers = None

    def _find_absolute_path(self, filename, search_dirs):
        if not filename or filename.startswith("No supported models") or filename.startswith("None"):
            return None
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

    def _init_vision_chat_handler(self, vision_path, llm_name):
        model_lower = str(llm_name).lower()
        handler_instance = None
        handler_name = "None"

        def try_init(handler_class, is_generic=False):
            if is_generic:
                try:
                    return handler_class(mmproj_path=vision_path, verbose=False)
                except Exception:
                    pass
            try:
                return handler_class(mmproj_path=vision_path)
            except Exception:
                pass
            try:
                return handler_class(vision_path)
            except Exception:
                pass
            return None

        # Try Qwen VL chat handlers
        if "qwen" in model_lower:
            for q_name in ["Qwen25VLChatHandler", "Qwen2VLChatHandler", "QwenVLChatHandler"]:
                if hasattr(chat_formats, q_name):
                    cls_obj = getattr(chat_formats, q_name)
                    handler_instance = try_init(cls_obj)
                    if handler_instance is not None:
                        handler_name = q_name
                        break

        # Try Generic MTMD Handler
        if handler_instance is None and hasattr(chat_formats, "GenericMTMDChatHandler"):
            cls_obj = getattr(chat_formats, "GenericMTMDChatHandler")
            handler_instance = try_init(cls_obj, is_generic=True)
            if handler_instance is not None:
                handler_name = "GenericMTMDChatHandler"

        # Try Llava 1.6 / 1.5 handlers
        if handler_instance is None and hasattr(chat_formats, "Llava16ChatHandler"):
            cls_obj = getattr(chat_formats, "Llava16ChatHandler")
            handler_instance = try_init(cls_obj)
            if handler_instance is not None:
                handler_name = "Llava16ChatHandler"

        if handler_instance is None and hasattr(chat_formats, "Llava15ChatHandler"):
            cls_obj = getattr(chat_formats, "Llava15ChatHandler")
            handler_instance = try_init(cls_obj)
            if handler_instance is not None:
                handler_name = "Llava15ChatHandler"

        if handler_instance is None:
            raise RuntimeError(f"Could not initialize a vision chat handler for '{llm_name}' with vision path '{vision_path}'.")

        return handler_instance, handler_name

    def load_model(self, llm_name, vision_name, n_ctx, n_gpu_layers, has_image):
        if llm_name.lower().endswith(".safetensors"):
            raise ValueError("Safetensors format is not supported by llama-cpp. Please select a .gguf model.")

        search_dirs = []
        if "text_encoders" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("text_encoders"))
        if "llm" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("llm"))
        if "unet" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("unet"))

        llm_path = self._find_absolute_path(llm_name, search_dirs)
        if not llm_path:
            raise FileNotFoundError(f"LLM model '{llm_name}' not found on disk.")

        active_vision_path = None
        if has_image and vision_name and not vision_name.startswith("None"):
            active_vision_path = self._find_absolute_path(vision_name, search_dirs)

        # Check if already loaded in memory
        if (self.llm is not None and
            self.loaded_model_path == llm_path and
            self.loaded_vision_path == active_vision_path and
            self.loaded_n_ctx == n_ctx and
            self.loaded_n_gpu_layers == n_gpu_layers):
            return

        self.unload_model()

        model_lower = llm_name.lower()
        detected_chat_format = None
        if "qwen" in model_lower:
            detected_chat_format = "chatml"
        elif "gemma" in model_lower:
            detected_chat_format = "gemma"
        elif "llama-3" in model_lower or "llama3" in model_lower:
            detected_chat_format = "llama-3"
        elif "mistral" in model_lower:
            detected_chat_format = "mistral-instruct"

        vision_status_str = "Disabled (Text-Only Mode)"
        if has_image and active_vision_path:
            self.chat_handler, v_name = self._init_vision_chat_handler(active_vision_path, llm_name)
            vision_status_str = f"Active ({v_name})"
        else:
            self.chat_handler = None

        print(f"[RT-LTX-2.5] Loading GGUF Model: {os.path.basename(llm_path)} [Vision: {vision_status_str}] [GPU Layers: {n_gpu_layers}]")

        gpu_supported = False
        try:
            if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                gpu_supported = llama_cpp.llama_supports_gpu_offload()
            if not gpu_supported and torch.cuda.is_available():
                gpu_supported = True
        except Exception:
            gpu_supported = False

        effective_gpu_layers = n_gpu_layers if gpu_supported else 0
        n_batch_val = min(max(n_ctx, 2048), 4096)
        n_ubatch_val = min(n_batch_val, 2048)

        llama_kwargs = {
            "model_path": llm_path,
            "chat_handler": self.chat_handler,
            "n_gpu_layers": effective_gpu_layers,
            "n_ctx": n_ctx,
            "n_batch": n_batch_val,
            "n_ubatch": n_ubatch_val,
            "main_gpu": 0,
            "verbose": False
        }
        if detected_chat_format and not self.chat_handler:
            llama_kwargs["chat_format"] = detected_chat_format

        try:
            # Try loading with flash attention if supported
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
            print(f"[RT-LTX-2.5] [OK] Model successfully loaded: {os.path.basename(llm_path)}")

        except Exception as e:
            self.unload_model()
            raise RuntimeError(f"[RT-LTX-2.5] [ERROR] Failed to load model '{llm_name}': {e}")

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

    def _tensor_to_base64(self, image_tensor):
        if image_tensor is None:
            return None
        if len(image_tensor.shape) == 4:
            img = image_tensor[0].cpu().numpy()
        else:
            img = image_tensor.cpu().numpy()

        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=92)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _parse_bulletproof_json(self, raw_text):
        """
        Robust JSON extractor for LLM output:
        - Strips <think>...</think> reasoning tags
        - Extracts JSON from markdown code fences or bare braces
        - Gracefully recovers partial fields if output was truncated
        """
        clean = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
        if "<think>" in clean.lower():
            p_brace = clean.find("{")
            if p_brace != -1:
                clean = clean[p_brace:].strip()
            else:
                clean = re.sub(r"<think>[\s\S]*$", "", clean, flags=re.IGNORECASE).strip()

        parsed = None
        # Attempt 1: Standard JSON parse
        try:
            parsed = json.loads(clean)
        except Exception:
            pass

        # Attempt 2: Extract between first { and last }
        if not parsed:
            m = re.search(r"(\{[\s\S]*\})", clean)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    pass

        # Attempt 3: Regex field recovery
        if not parsed or not isinstance(parsed, dict):
            enhanced_match = re.search(r'"enhanced_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)
            neg_match = re.search(r'"negative_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)
            audio_match = re.search(r'"audio_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)

            if enhanced_match:
                parsed = {
                    "enhanced_prompt": enhanced_match.group(1).encode("utf-8").decode("unicode_escape"),
                    "negative_prompt": neg_match.group(1).encode("utf-8").decode("unicode_escape") if neg_match else "",
                    "audio_prompt": audio_match.group(1).encode("utf-8").decode("unicode_escape") if audio_match else "",
                }
            else:
                # Fallback: Treat cleaned text directly as enhanced prompt
                fallback_prompt = clean.strip().strip('"').strip()
                parsed = {
                    "enhanced_prompt": fallback_prompt,
                    "negative_prompt": "",
                    "audio_prompt": "",
                }

        return parsed

    def _build_default_negative_prompt(self, mode):
        base_negatives = [
            "morphing", "frame jitter", "flickering lighting", "jittery stop-motion artifacts",
            "distorted anatomy", "extra limbs", "fused fingers", "plastic skin",
            "digital video compression artifacts", "blurry out-of-focus background",
            "muddy audio", "robotic monotone speech", "audio clipping", "muffled dialogue",
            "camera shake", "sudden frame jump", "unstable facial identity"
        ]
        if "Multi-Shot" in mode:
            base_negatives.extend(["abrupt unannounced cut", "disjointed geography", "drifting wardrobe colors"])
        elif "Dub-It" in mode:
            base_negatives.extend(["unsynchronized lip flapping", "mismatched syllable cadence", "garbled pronunciation"])
        return ", ".join(base_negatives)

    def enhance_prompt(self, enabled, llm_model, user_prompt, enhancement_mode, camera_movement,
                       lighting_atmosphere, audio_mode, frame_length, creativity, seed,
                       max_tokens, negative_prompt_generation, n_gpu_layers=-1, n_ctx=8192,
                       keep_model_loaded=True, vision_model=None, image=None, custom_system_instructions=""):

        if not enabled:
            return (user_prompt, "", "[RT LTX-2.5] Prompt Enhancer is Disabled (Bypassed). Original user prompt passed through.")

        t_start = time.perf_counter()

        has_image = (image is not None)
        base64_image = self._tensor_to_base64(image) if has_image else None

        # 1. Parse frame count from selection
        recommended_frames = 121
        if "121" in frame_length:
            recommended_frames = 121
        elif "193" in frame_length:
            recommended_frames = 193
        elif "241" in frame_length:
            recommended_frames = 241
        elif "73" in frame_length:
            recommended_frames = 73

        # 2. Parse temperature
        temp = 0.9
        if "0.7" in creativity:
            temp = 0.7
        elif "1.1" in creativity:
            temp = 1.1

        max_tok = int(max_tokens) if str(max_tokens).isdigit() else 1024

        # 3. Assemble complete system directive
        full_system_prompt = LTX25_MASTER_SYSTEM_INSTRUCTION
        if custom_system_instructions and str(custom_system_instructions).strip():
            full_system_prompt += f"\n\n[USER CUSTOM DIRECTIVES]:\n{str(custom_system_instructions).strip()}"

        # 4. Construct User Task Instruction
        user_task = (
            f"Please engineer a production-ready LTX-2.5 prompt based on the following specifications:\n"
            f"- User Concept / Idea: {user_prompt.strip() if user_prompt.strip() else 'A cinematic sequence demonstrating high visual fidelity and natural movement'}\n"
            f"- Target Structuring Mode: {enhancement_mode}\n"
            f"- Camera Movement Requirement: {camera_movement}\n"
            f"- Lighting & Atmosphere Logic: {lighting_atmosphere}\n"
            f"- Audio & Dialogue Mode: {audio_mode}\n"
            f"- Target Duration / Frames: {frame_length}\n"
            f"CRITICAL REQUIREMENT: You MUST weave the audio description, ambient soundscape, and any quoted dialogue directly into the 'enhanced_prompt' text.\n"
        )

        if has_image:
            user_task += (
                f"\n[IMAGE-TO-VIDEO DIRECTIVE]: An initial frame image has been provided. "
                f"You MUST visually analyze the image and anchor the enhanced prompt directly onto the character's visible facial likeness, "
                f"hairstyle, clothing materials, environment, and existing lighting logic. Do NOT contradict or hallucinate details that clash with this image."
            )

        user_task += "\n\nOutput ONLY a valid JSON object matching the required schema. Ensure dialogue is strictly enclosed in double quotation marks."

        # 5. Load model & execute inference
        self.load_model(llm_model, vision_model, n_ctx, n_gpu_layers, has_image)

        # Build chat messages
        messages = [{"role": "system", "content": full_system_prompt}]
        if has_image and base64_image and self.chat_handler:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_task},
                    {"type": "image_url", "image_url": {"url": base64_image}}
                ]
            }
        else:
            user_msg = {"role": "user", "content": user_task}
        messages.append(user_msg)

        # Configure seed
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed % 0xffffffff)

        gen_kwargs = {
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
            "response_format": {"type": "json_object"}
        }
        if seed != -1:
            gen_kwargs["seed"] = int(seed % 0xffffffff)

        try:
            response = self.llm.create_chat_completion(**gen_kwargs)
            raw_response_content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
        except Exception as e:
            # Fallback if response_format is not supported by backend
            gen_kwargs.pop("response_format", None)
            response = self.llm.create_chat_completion(**gen_kwargs)
            raw_response_content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)

        t_elapsed = time.perf_counter() - t_start

        # 6. Parse and validate JSON output
        parsed_data = self._parse_bulletproof_json(raw_response_content)
        enhanced_prompt = parsed_data.get("enhanced_prompt", "").strip()

        # Enforce single flowing paragraph for Single-Shot and Multi-Shot modes
        if "Single-Shot" in enhancement_mode or "Multi-Shot" in enhancement_mode:
            enhanced_prompt = re.sub(r"\s*\n+\s*", " ", enhanced_prompt).strip()

        # Enforce static camera lock phrase if static tripod was chosen
        if "Static Tripod" in camera_movement and "static" not in enhanced_prompt.lower():
            enhanced_prompt += " The camera remains locked-off on a stationary static tripod with zero camera movement."

        audio_prompt = parsed_data.get("audio_prompt", "").strip()
        breakdown = parsed_data.get("cinematic_breakdown", {})
        audio_design = breakdown.get("audio_design", "").strip() if isinstance(breakdown, dict) else ""
        audio_source = audio_design or audio_prompt

        # Enforce audio presence in enhanced_prompt if audio mode is active
        if "Silent" not in audio_mode and audio_source:
            audio_keywords = ["sound", "audio", "hear", "whisper", "ambient", "foley", "music", "score", "hum", "buzz", "crunch", "rustl", "ring", "chatter", "drone"]
            has_audio_mention = any(kw in enhanced_prompt.lower() for kw in audio_keywords)
            if not has_audio_mention:
                clean_audio = audio_source.strip().rstrip(".")
                if not clean_audio.lower().startswith("the audio") and not clean_audio.lower().startswith("the ambient") and not clean_audio.lower().startswith("audio:"):
                    audio_sentence = f"The ambient audio features {clean_audio}."
                else:
                    audio_sentence = f"{clean_audio}."

                preservation_marker = "Preserve character facial likeness"
                if preservation_marker in enhanced_prompt:
                    parts = enhanced_prompt.split(preservation_marker)
                    enhanced_prompt = f"{parts[0].strip()} {audio_sentence} {preservation_marker}{parts[1]}"
                else:
                    enhanced_prompt = f"{enhanced_prompt.rstrip('.')} {audio_sentence}"

        # Handle negative prompt
        if "Enable" in negative_prompt_generation:
            llm_neg = parsed_data.get("negative_prompt", "").strip()
            negative_prompt = llm_neg if llm_neg else self._build_default_negative_prompt(enhancement_mode)
        else:
            negative_prompt = ""

        if not keep_model_loaded:
            self.unload_model()

        # 7. Construct Diagnostics Report
        breakdown_lines = []
        if isinstance(breakdown, dict):
            for k, v in breakdown.items():
                breakdown_lines.append(f"  * {k.replace('_', ' ').title()}: {v}")
        breakdown_str = "\n".join(breakdown_lines) if breakdown_lines else "  * Automatic Cinematic Composition"

        diagnostics_report = (
            f"=== [RT LTX-2.5 Prompt Enhancer Diagnostics] ===\n"
            f"  * Mode: {enhancement_mode}\n"
            f"  * LLM Model: {os.path.basename(str(self.loaded_model_path or llm_model))}\n"
            f"  * Vision Anchoring: {'Active (Initial Frame Analyzed)' if has_image else 'Text-Only Generation'}\n"
            f"  * Camera Movement: {camera_movement}\n"
            f"  * Lighting Logic: {lighting_atmosphere}\n"
            f"  * Audio Mode: {audio_mode}\n"
            f"  * Frame Count: {recommended_frames} frames ({frame_length})\n"
            f"  * Creativity (Temp): {temp}\n"
            f"  * Execution Time: {t_elapsed:.2f}s ({total_tokens} tokens)\n\n"
            f"--- Cinematic Design Breakdown ---\n"
            f"{breakdown_str}\n\n"
            f"--- Raw Model Response ---\n"
            f"{raw_response_content.strip()}"
        )

        print(f"[RT-LTX-2.5] [OK] Prompt enhanced in {t_elapsed:.2f}s ({total_tokens} tokens, Mode: {enhancement_mode})")

        return (enhanced_prompt, negative_prompt, diagnostics_report)


# ==============================================================================
# NODE MAPPINGS
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "RT_LTX25_PromptEnhancer": RT_LTX25_PromptEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_LTX25_PromptEnhancer": "RT LTX-2.5 Prompt Enhancer"
}






