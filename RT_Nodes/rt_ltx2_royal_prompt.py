import os
import gc
import base64
import io
import re
import numpy as np
import torch
import json
from PIL import Image
import folder_paths

# ── Dependency Check ─────────────────────────────────────────────────────────
try:
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    print("\n\033[91m[RT-LTX-2] ERROR: 'llama-cpp-python' is missing.\033[0m")
    print("Please run: pip install llama-cpp-python\n")
    raise


# ====================================================================================================
# SECTION 1: EVERYTHING RELATED TO THE NODE CONFIGURATION & HELPERS
# ====================================================================================================
class RT_LTX2_RoyalPrompt:
    
    @staticmethod
    def get_supported_models():
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
            return ["No supported models (.gguf or .safetensors) found in text_encoders/llm"]
        return sorted(list(unique_models))

    @classmethod
    def INPUT_TYPES(s):
        valid_models = s.get_supported_models()
        return {
            "required": {
                "llm_model": (valid_models, {"default": valid_models[0]}), 
                "vision_model": (valid_models, {"default": valid_models[0]}),
                "user_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe action... (Keep it very simple for LTX-2)"
                }),
                "enhancement": (["01. LTX Prompt Enhancer", "02. Prompt Relay", "03. HiDream Prompt Enhancer", "04. SCOPE Prompt Enhancer", "05. Flux Prompt Enhancer"], {"default": "01. Prompt Enhancer"}),
                "max_tokens": (["256", "512", "800", "1024", "2048"], {"default": "1024"}),
                "creativity": (["0.7 - Literal", "0.9 - Balanced", "1.1 - Artistic"], {"default": "0.9 - Balanced"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "debug_console": ("BOOLEAN", {"default": True}), 
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768}),
                "frame_count": ("INT", {"default": 120, "min": 24, "max": 960}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("PROMPT 1", "PROMPT 2", "FRAMES")
    FUNCTION = "generate"
    CATEGORY = "RareTutor"

    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.loaded_model_path = None
        self.loaded_vision_path = None
        self.banned_tokens = {}

    def _tensor_to_base64(self, image_tensor):
        if image_tensor is None:
            return None
        if len(image_tensor.shape) == 4:
            img = image_tensor[0].cpu().numpy()
        else:
            img = image_tensor.cpu().numpy()
            
        img = (img * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

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

    def load_model(self, llm_name, vision_name, n_ctx, has_image):
        if llm_name.lower().endswith(".safetensors") or vision_name.lower().endswith(".safetensors"):
            raise ValueError("SAFETENSORS_ERROR")

        search_dirs = []
        if "text_encoders" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("text_encoders"))
        if "llm" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("llm"))
        if "unet" in folder_paths.folder_names_and_paths:
            search_dirs.extend(folder_paths.get_folder_paths("unet"))

        llm_path = self._find_absolute_path(llm_name, search_dirs)
        vision_path = self._find_absolute_path(vision_name, search_dirs)
        
        if not llm_path: raise FileNotFoundError(f"LLM '{llm_name}' not found.")
        if has_image and not vision_path: raise FileNotFoundError(f"Vision '{vision_name}' not found.")
        
        active_vision_path = vision_path if has_image else None
        if (self.llm is not None and 
            self.loaded_model_path == llm_path and 
            self.loaded_vision_path == active_vision_path):
            return

        self.unload_model()
        
        # ── AUTO MODEL DETECTION FOR CHAT FORMATTING ──
        model_lower = llm_name.lower()
        detected_chat_format = None
        
        if "qwen" in model_lower:
            detected_chat_format = "chatml"
            model_family = "Qwen (ChatML)"
        elif "gemma-4" in model_lower or "gemma4" in model_lower:
            detected_chat_format = "gemma" 
            model_family = "Gemma 4"
        elif "gemma" in model_lower:
            detected_chat_format = "gemma"
            model_family = "Gemma"
        else:
            model_family = "Auto-Detect"

        vision_status = "Enabled" if has_image else "Disabled (Text Only)"
        print(f"\n[RT-LTX-2] Loading Models into VRAM... [Vision: {vision_status}] [Format: {model_family}]")
        
        try:
            if has_image:
                self.chat_handler = Llava15ChatHandler(clip_model_path=vision_path)
            else:
                self.chat_handler = None

            llama_kwargs = {
                "model_path": llm_path,
                "chat_handler": self.chat_handler,
                "n_gpu_layers": -1,
                "n_ctx": n_ctx,
                "logits_all": True,
                "verbose": False
            }
            
            # Inject correct chat format if text-only mode to prevent formatting bleed
            if detected_chat_format and not has_image:
                llama_kwargs["chat_format"] = detected_chat_format

            self.llm = Llama(**llama_kwargs)
            self.loaded_model_path = llm_path
            self.loaded_vision_path = active_vision_path
            
            banned_strings = ["ASSISTANT:", "Assistant:", "USER:", "User:", "Here is the", "Okay,", "Sure,"]
            self.banned_tokens = {}
            for bad_str in banned_strings:
                tokens = self.llm.tokenize(bad_str.encode('utf-8'), add_bos=False)
                if len(tokens) > 0:
                    self.banned_tokens[tokens[0]] = -100.0
                    
        except Exception as e:
            print(f"[RT-LTX-2] CRITICAL ERROR: {e}")
            self.unload_model()
            raise RuntimeError(f"Load failed: {e}")

    def unload_model(self):
        if self.llm: del self.llm
        if self.chat_handler: del self.chat_handler
        self.llm = None
        self.chat_handler = None
        gc.collect()
        torch.cuda.empty_cache()

    def _clean_output(self, text):
        text = re.sub(r"^(Sure|Okay|Here is|Here's).*?:\n+", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"^\s*\**Here is a cinematic.*?\**\s*\n+", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"^\x60\x60\x60(?:text)?\n(.*?)\n\x60\x60\x60$", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"\x60\x60\x60[a-z]*\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\x60\x60\x60", "", text)
        text = re.sub(r"\*\*(\[.*?\])\*\*", r"\1", text)
        
        # Scrub any rogue format tags that bleed into the response
        text = re.sub(r"<\|im_end\|>|<\|im_start\|>|<turn\|>|<end_of_turn>|<start_of_turn>", "", text)
        
        ambient_matches = list(re.finditer(r"\[AMBIENT:.*?\]", text, re.IGNORECASE))
        if ambient_matches:
            last_match = ambient_matches[-1]
            text = text[:last_match.end()]
        else:
            text = re.sub(r"(ASSISTANT|USER REQUEST|USER:|\[SCENE START\]).*", "", text, flags=re.IGNORECASE | re.DOTALL)
            
        text = text.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
            
        text = text.strip()

        if "[Style" not in text and "[style" not in text:
            text = "[Style : highly detailed, accurate to visual reference]\n" + text
        else:
            text = re.sub(r"^.*?(\[Style\s*:)", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
            
        return text.strip()

    # ── THE ROUTER: Sends data to Options 1, 2, or 3 ──
    def generate(self, llm_model, vision_model, user_input, enhancement, max_tokens, creativity, seed, debug_console, keep_model_loaded, n_ctx, frame_count, image=None):
        has_image = image is not None
        try:
            self.load_model(llm_model, vision_model, n_ctx, has_image)
        except ValueError as e:
            if str(e) == "SAFETENSORS_ERROR":
                error_msg = ("[ERROR] You selected a .safetensors model. Please select a `.gguf` file.")
                return (error_msg, error_msg, frame_count)
            else:
                raise e
            
        base64_image = self._tensor_to_base64(image) if has_image else None
        token_val = int(max_tokens.split(" - ")[0]) if " - " in max_tokens else int(max_tokens)
        temp = 0.6 if "Literal" in creativity else float(creativity.split(" - ")[0])
        
        # Route based on dropdown selection
        if enhancement == "01. LTX Prompt Enhancer":
            return self._run_option_01(llm_model, user_input, seed, keep_model_loaded, frame_count, base64_image, token_val, temp)
        elif enhancement == "02. Prompt Relay":
            return self._run_option_02(llm_model, user_input, seed, keep_model_loaded, frame_count, base64_image, token_val, temp)
        elif enhancement == "03. HiDream Prompt Enhancer":
            return self._run_option_03(llm_model, user_input, seed, keep_model_loaded, frame_count, base64_image, token_val, temp)
        elif enhancement == "04. SCOPE Prompt Enhancer":
            return self._run_option_04(llm_model, user_input, seed, keep_model_loaded, frame_count, base64_image, token_val, temp)
        elif enhancement == "05. Flux Prompt Enhancer":
            return self._run_option_05(llm_model, user_input, seed, keep_model_loaded, frame_count, base64_image, token_val, temp)


# ====================================================================================================
# SECTION 2: OPTION 01 (STRICTLY UNCHANGED ORIGINAL LOGIC + NO HALLUCINATED DIALOGUE)
# ====================================================================================================
    def _run_option_01(self, llm_model, user_input, seed, keep_model_loaded, frame_count, base64_image, token_val, temp):

        # ── Pre-extract narrator and dialogue blocks that MUST be preserved verbatim ──
        # Matches: [Narrator's voice in background says : "text"]
        #          [Narrator's voice : "text"]
        #          [Narrator : "text"]  and close variations
        narrator_pattern = re.compile(
            r'\[Narrator(?:\'s)?\s+(?:voice\s+)?(?:in\s+background\s+)?(?:says?\s*)?[:\-]\s*"[^"]*"\]',
            re.IGNORECASE
        )
        narrator_blocks = narrator_pattern.findall(user_input)

        # Build an explicit preservation checklist to embed in the user message
        if narrator_blocks:
            preservation_list = "\n".join(f"  • {b}" for b in narrator_blocks)
            preservation_notice = (
                f"\n\nPRESERVATION MANDATE — embed these VERBATIM at their correct position "
                f"(exact brackets, exact words, exact punctuation):\n{preservation_list}"
            )
        else:
            preservation_notice = ""

        SYSTEM_PROMPT = """You are an elite video prompt engineer for LTX-2.3. Your task is to ENHANCE the user's request into a rich, cinematic, highly detailed video prompt — while making the MINIMUM necessary changes and preserving every piece of dialogue and narrator annotation EXACTLY as written.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE PRESERVATION RULES (NEVER VIOLATE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NARRATOR ANNOTATIONS: Any text in the format [Narrator's voice in background says : "text"] or [Narrator's voice : "text"] or any [Narrator...] variation MUST be copied into the output VERBATIM — same brackets, same wording, same punctuation — placed at the same logical point in the scene. Do NOT paraphrase, reword, or move them.
2. CHARACTER DIALOGUE: Any spoken dialogue attributed to a character (e.g. He says "...", She whispers "...", or quoted speech "...") MUST appear in the final prompt word-for-word, unchanged. Do NOT paraphrase or drop any dialogue.
3. NO INVENTION: If the user provided NO dialogue and NO narrator text, do not add any. Do not invent characters, events, or lines that were not in the original request.
4. MINIMAL RESTRUCTURING: Enhance AROUND the user's content — layer in visual, cinematic, and sensory detail. Do NOT reorder narrative events or replace the user's chosen words with different ones.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL FORMATTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. STYLE TAG FIRST: The absolute first line MUST be: [Style : <3D Animation OR Live-Action OR 2D Anime>, <texture>, <lighting>].
2. NO CHATTY FILLER: NEVER start with "Here is the prompt". Start instantly with the [Style : ...] tag.
3. AMBIENT TAG: The final line MUST be the [AMBIENT: ...] audio tag.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ENHANCE FOR LTX-2 (ADD these layers; do NOT replace the user's content)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- TEXTURES & MATERIALS: Fabric weave, surface reflectivity, skin details, ground texture, material aging.
- LIGHTING DETAIL: Light source direction, shadow softness, color temperature, rim lighting, how light catches edges and surfaces.
- MICRO-MOVEMENTS: Eye blinks, breath rhythm, individual hair strands shifting, fabric draping under gravity, hands trembling.
- SPATIAL CLARITY: Exact camera framing (close-up / medium / wide), subject distance, depth of field, foreground and background layers.
- ATMOSPHERE: Ambient haze, dust particles in light, color grading, weather nuances, environmental sounds implied by visuals.
- WRITE LONG AND VIVID: Each enhancement adds depth; the output should be substantially more detailed than the input while keeping the same story and same words.

=== PERFECT OUTPUT EXAMPLE (narrator annotation preserved) ===
[Style : Live-Action, photorealistic skin and worn fabric texture, warm late-afternoon volumetric lighting]
A medium close-up shot of an elderly man seated at a weathered oak desk, late-afternoon light slanting through dusty venetian blinds and casting amber horizontal bars across his deeply lined face. His worn charcoal wool cardigan catches the warm glow at each fold, individual threads visible at the fraying elbows. His thick-veined hands rest flat on a yellowed letter, trembling almost imperceptibly. [Narrator's voice in background says : "He never sent that letter."] The man's eyes, clouded with age and rimmed with moisture, drift slowly toward the window. A long, deliberate breath makes his chest rise and fall — the only movement in a room that holds its silence like a held breath. Pale dust motes drift through the amber beam of light above him.
[AMBIENT: faint ticking wall clock, quiet room tone, distant muffled traffic, soft creak of the chair]
============================================================="""

        duration_secs = max(1, frame_count // 24)
        if duration_secs <= 6:
            pacing_rule = (
                f"CRITICAL PACING: The video is short ({duration_secs} seconds). "
                f"Describe ONLY ONE continuous camera shot. Do NOT use cuts. "
                f"Instead of fast action, write a LONG, highly descriptive paragraph focusing deeply on MICRO-DETAILS."
            )
        else:
            pacing_rule = (
                f"CRITICAL PACING: The target video is {duration_secs} seconds long. "
                f"Write a LONG, highly detailed sequence."
            )

        clean_user_input = user_input.strip()

        blueprint = (
            f"{pacing_rule}\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "Line 1: [Style : <MUST STATE IF 3D ANIMATION, PHOTOREALISTIC, OR 2D ANIME>, <texture, lighting>]\n"
            "Line 2+: <Your long, vivid, highly detailed scene — ALL narrator annotations and dialogue preserved verbatim>\n"
            "Final Line: [AMBIENT: <soundscape>]\n\n"
            "START IMMEDIATELY with the '[Style : ' tag."
        )

        final_prompt = (
            f"USER REQUEST:\n'{clean_user_input}'\n\n"
            f"TASK: Enhance this into a highly detailed cinematic LTX-2 prompt. "
            f"Preserve every narrator annotation and character dialogue VERBATIM — add rich visual, lighting, "
            f"texture, and micro-movement details around them, but do NOT change or remove them."
            f"{preservation_notice}\n\n"
            f"{blueprint}"
        )

        if base64_image is not None:
            user_content_block = [{"type": "text", "text": final_prompt}, {"type": "image_url", "image_url": {"url": base64_image}}]
        else:
            user_content_block = final_prompt

        stop_tokens = ["<end_of_turn>", "<eos>", "<|eot_id|>", "User:", "ASSISTANT:", "Assistant:", "REAL TASK:", "USER REQUEST:"]
        model_lower = llm_model.lower()
        if "qwen" in model_lower: stop_tokens.extend(["<|im_end|>", "<|im_start|>"])
        elif "gemma" in model_lower: stop_tokens.extend(["<turn|>", "<start_of_turn>"])

        # ── Seed fix: llama.cpp expects uint32 (0 – 4294967295); clamp 64-bit values ──
        safe_seed = (int(seed) % (2 ** 32)) if seed != -1 else None

        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content_block}],
                max_tokens=token_val, temperature=temp, stop=stop_tokens, logit_bias=self.banned_tokens,
                seed=safe_seed
            )
            raw_result = response['choices'][0]['message']['content'].strip()
            final_result = self._clean_output(raw_result)

            # ── Safety net: re-inject any narrator block the LLM silently dropped ──
            for block in narrator_blocks:
                if block not in final_result:
                    ambient_match = re.search(r'\[AMBIENT:', final_result, re.IGNORECASE)
                    if ambient_match:
                        insert_pos = ambient_match.start()
                        final_result = (
                            final_result[:insert_pos].rstrip() + "\n" + block + "\n" + final_result[insert_pos:]
                        )
                    else:
                        final_result = final_result.rstrip() + "\n" + block

        except Exception as e:
            final_result = f"Error: {e}"

        if not keep_model_loaded: self.unload_model()
        return (final_result, "", frame_count)


# ====================================================================================================
# SECTION 3: OPTION 02 (PROMPT RELAY - UPGRADED WITH TEMPORAL ROUTING PRINCIPLES)
# ====================================================================================================
# Based on: GordonChen19/Prompt-Relay (arXiv:2604.10030)
# Core insight: PART1 = global prompt (anchors persistent identity/scene across all segments).
# PART2 segments = local prompts (each is a self-contained, temporally isolated event).
# Temporal entanglement is prevented by making each local prompt describe ONLY its own moment.
# Output format (PART1 / PART2 / pipe separator) is UNCHANGED.
# ─────────────────────────────────────────────────────────────────────────────
    def _run_option_02(self, llm_model, user_input, seed, keep_model_loaded, frame_count, base64_image, token_val, temp):
        analytical_temp = 0.3

        SYSTEM_PROMPT = """You are a strict video prompt routing assistant for multi-event video generation. You do not converse. You ONLY output text wrapped in XML tags. No text outside the tags — ever.

Your task is to split the user's scene description into two XML blocks: <PART1> and <PART2>.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — GLOBAL ANCHOR PROMPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART1 is the global prompt. It conditions the ENTIRE video and must lock in every persistent element so characters and objects stay consistent across all events. Include:
  - Style tag (animation style, render quality, lighting type)
  - Camera description (angle, movement, framing)
  - Persistent character identity: name/role, physical appearance (hair color, build, clothing) described with enough specificity that the model can maintain identity across segments
  - Persistent scene/environment: location, time of day, weather, background details
  - Overall tone and mood

DO NOT include any event or action in PART1. It is a stable foundation only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — LOCAL EVENT PROMPTS (pipe-separated)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART2 contains the ordered sequence of events, separated by the pipe character '|' on its own line.

Each segment (between pipes) is a LOCAL PROMPT for ONE temporal slice of the video. Each must:
  1. Be SELF-CONTAINED: briefly re-identify the subject (e.g. "The young man") so it is unambiguous even when read in isolation
  2. Describe ONE clear, specific event — what is happening RIGHT NOW in this segment only
  3. Include a DYNAMIC ACTION: always state what the subject is physically doing, including body part motion, direction, and any interaction with objects or environment. Never leave a subject static unless the event is explicitly a pause or freeze
  4. Be TEMPORALLY ISOLATED: do NOT describe what happened before or after; do not bleed semantics from adjacent segments. Each segment owns its own moment
  5. Keep transitions smooth: the action in each segment should naturally follow from the prior segment's endpoint

Number of segments: derive naturally from how many distinct events the user described. Do not merge or skip events.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Output ONLY <PART1>...</PART1> and <PART2>...</PART2> — zero text outside
2. Pipe '|' must be on its own line between segments
3. Never add section headers, numbers, or commentary inside the tags

=== PERFECT OUTPUT EXAMPLE ===
<PART1>
[3D Disney Pixar animation style, warm volumetric lighting, wide establishing shot]
The camera holds still at a medium distance, slightly low angle.
An old man with a white beard, wearing a brown vest and suspenders, sits in a wooden rocking chair on a sunlit porch.
A small red panda with rust-red fur and a striped tail rests nearby on the porch railing.
</PART1>
<PART2>
The old man leans forward slowly in his rocking chair, raises one eyebrow, and says "Well well..." as he lifts his hat off his head with his right hand and sets it on his knee
|
The red panda pushes off the railing with its hind legs, leaps across the gap, and lands softly on the old man's lap, curling into a ball as the old man's free hand reaches down to stroke its back
</PART2>
==============================="""

        final_prompt = f"USER REQUEST:\n'{user_input.strip()}'\n\nTASK: Split the above request EXACTLY into <PART1> (global anchor) and <PART2> (pipe-separated local event prompts)."
        
        if base64_image is not None:
            user_content_block = [{"type": "text", "text": final_prompt}, {"type": "image_url", "image_url": {"url": base64_image}}]
        else:
            user_content_block = final_prompt

        stop_tokens = ["<end_of_turn>", "<eos>", "<|eot_id|>", "User:", "ASSISTANT:"]
        model_lower = llm_model.lower()
        if "qwen" in model_lower: stop_tokens.extend(["<|im_end|>", "<|im_start|>"])
        elif "gemma" in model_lower: stop_tokens.extend(["<turn|>", "<start_of_turn>"])

        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content_block}],
                max_tokens=token_val, temperature=analytical_temp, stop=stop_tokens,
                seed=(int(seed) % (2 ** 32)) if seed != -1 else None
            )
            raw_result = response['choices'][0]['message']['content'].strip()
            part1_match = re.search(r"<PART1>\s*(.*?)\s*</PART1>", raw_result, re.DOTALL | re.IGNORECASE)
            part2_match = re.search(r"<PART2>\s*(.*?)\s*</PART2>", raw_result, re.DOTALL | re.IGNORECASE)
            
            final_part1 = part1_match.group(1).strip() if part1_match else raw_result
            if part2_match:
                final_part2 = re.sub(r'\s*\|\s*', '\n|\n', part2_match.group(1).strip())
            else:
                final_part2 = "Error: The AI failed to use <PART2> tags."
        except Exception as e:
            final_part1, final_part2 = f"Error: {e}", ""

        if not keep_model_loaded: self.unload_model()
        return (final_part1, final_part2, frame_count)


# ====================================================================================================
# SECTION 4: OPTION 03 (HIDREAM PROMPT ENHANCER V2 - FULL SCALIST AGENT)
# ====================================================================================================
    @staticmethod
    def _has_tail_repetition(text: str, n: int = 5, threshold: int = 11) -> bool:
        """Detect degenerate looping output (ported from HiDream prompt_agent_v2.py)."""
        words = text.lower().split()
        if len(words) < n + 1:
            return True
        grams: dict = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i: i + n])
            grams[gram] = grams.get(gram, 0) + 1
            if grams[gram] > threshold:
                return True
        return False

    @staticmethod
    def _is_structured_multichar(text: str) -> bool:
        """Detect structured multi-character prompt format (bbox + refs + color anchors)."""
        has_bbox   = bool(re.search(r"Bounding Box\s*:", text, re.IGNORECASE))
        has_ref    = bool(re.search(r"\[ref_\d+", text, re.IGNORECASE))
        has_anchor = bool(re.search(r"Color anchor\s*:", text, re.IGNORECASE))
        return has_bbox and (has_ref or has_anchor)

    @staticmethod
    def _scrub_markdown(text: str) -> str:
        """Remove markdown formatting injected by LLMs into generation prompts."""
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)          # **bold** → plain
        text = re.sub(r"\*(.+?)\*",     r"\1", text)           # *italic* → plain
        text = re.sub(r"^#{1,6}\s+",    "",    text, flags=re.MULTILINE)  # ### headers
        text = re.sub(r"^-{3,}\s*$",    "",    text, flags=re.MULTILINE)  # --- dividers
        text = re.sub(r"`(.+?)`",        r"\1", text)           # `code` → plain
        return text.strip()

    def _run_option_03(self, llm_model, user_input, seed, keep_model_loaded, frame_count, base64_image, token_val, temp):

        is_structured = self._is_structured_multichar(user_input)

        # ══════════════════════════════════════════════════════════════════════
        # MODE A — STRUCTURED MULTI-CHARACTER PROMPT
        # Input has Bounding Boxes, [ref_N] tags, Color anchors, per-character
        # sections. The LLM must PRESERVE all structural data and AMPLIFY only
        # the descriptions/actions within each section.
        # ══════════════════════════════════════════════════════════════════════
        if is_structured:
            SYSTEM_PROMPT = """You are an expert image generation prompt enhancer specializing in structured multi-character scene specifications. Your task is to enhance the prompt IN-PLACE — improving the quality and specificity of descriptions WITHOUT altering or removing any structural data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE PRESERVATION RULES — NEVER VIOLATE THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. BOUNDING BOXES: Every `Bounding Box: [x, y, w, h]` value must appear in the output VERBATIM. Do NOT remove, round, or alter bbox coordinates under any circumstances.
2. REFERENCE TAGS: Every `[ref_N]` tag must be preserved exactly as written. Never expand or remove them.
3. COLOR ANCHORS: Every `Color anchor: #XXXXXX` must be preserved verbatim AND referenced in the character description as the dominant color identifier.
4. SECTION STRUCTURE: Preserve ALL section headers exactly: `---`, `SCENE:`, `CAMERA:`, `COMPOSITION:`, `CHAR_N`, character names, `VISUAL STYLE:`, `COLOR PALETTE:`.
5. `[BG_N]` TAGS: Preserve all background reference tags verbatim.
6. NO MARKDOWN: Never use `**bold**`, `*italic*`, `### headers`, or any markdown. Output plain text only.
7. CHAR `[]` BLOCKS: Preserve the `[]` wrapper on character appearance lines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU MUST ENHANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOR EACH CHARACTER — enhance the action line. The action line is the line AFTER the `[]` appearance block that describes what the character is doing. You MUST:
  - Expand it with full biomechanical specificity:
    * Which leg/foot is moving and in which direction
    * Upper body lean angle (e.g. torso pitched 15 degrees back)
    * Arm position (e.g. left arm raised instinctively, right hand grabbing nearest surface)
    * Shoulder/hip alignment during the movement
    * Weight shift and balance point
    * Facial micro-expressions: brow position, eye dilation, jaw, specific muscle tension
    * Clothing/hair/fabric reaction to the motion (e.g. vest flapping open, hoodie hem rising)
    * If the character is reacting to something — describe the direction of the reaction and what body language conveys the emotion

FOR EACH CHARACTER — enhance the `[]` appearance block:
  - Add material texture details to clothing (e.g. "worn canvas", "heavy-weave cotton")
  - Add how the scene lighting hits their specific skin tone and clothing color
  - Reference the Color anchor hex in the description

FOR THE SCENE block:
  - Enrich environmental texture descriptions (material surfaces, steam behavior, light quality)
  - Add how the lighting interacts with each character's position

FOR VISUAL STYLE and COLOR PALETTE:
  - Expand with precise cinematography terms and rendering quality descriptors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Wrap the full enhanced prompt in <PROMPT> tags.
Wrap a short enhancement log in <ANALYSIS> tags (list what you amplified per character).
Output no text outside these two tags."""

            user_message = (
                f"STRUCTURED PROMPT TO ENHANCE:\n\n{user_input.strip()}\n\n"
                "Enhance in-place: preserve ALL bbox coordinates, ref tags, color anchors, and section structure exactly. "
                "Amplify each character's action with full biomechanical detail. "
                "Wrap the enhanced prompt in <PROMPT> tags and the log in <ANALYSIS> tags."
            )

        # ══════════════════════════════════════════════════════════════════════
        # MODE B — FREE-FORM PROMPT
        # Apply full HiDream v2 SCALIST expansion.
        # Source: HiDream-ai/HiDream-O1-Image · prompt_agent_v2.py (dev branch)
        # ══════════════════════════════════════════════════════════════════════
        else:
            SYSTEM_PROMPT = """You are a professional AI Image Generation Prompt Engineering Engine and Creative Director with encyclopedic knowledge and visual directing ability. Your task is to analyze the user's raw image/scene request, resolve all implicit knowledge and find the best visual approach, then rewrite it into a clear, detailed, self-contained English prompt ready for direct use in an image or video generation model.

Image generation models can only execute direct visual descriptions. They cannot fill in background knowledge, logical relationships, or text content on their own. You must complete all knowledge resolution, spatial planning, and visual direction BEFORE writing the prompt, then embed all results explicitly. Never use markdown formatting (no **bold**, no headers).

SCALIST Framework — expand every scene across ALL seven dimensions:

Subject: The subject's identity, appearance, colors, materials, textures, ACTION, expression, and clothing.
Composition: Shot type (close-up, medium, wide, aerial), camera angle, subject placement, foreground/midground/background layers, negative space, visual focus.
Action: CRITICAL — What is the subject actively DOING? Describe the specific movement, direction, body posture, weight distribution, limb positions, and any object/environment interactions. NEVER write a static standing pose unless explicitly requested. If the context implies a role (warrior, dancer, chef, athlete), derive a physically specific dynamic action appropriate to that role and write it in full biomechanical detail.
Location: Scene setting, indoors/outdoors, era, weather, time of day, environmental textures, atmosphere.
Image Style: photorealistic, cinematic, oil painting, watercolor, anime, 3D render — matched with appropriate lighting and color atmosphere.
Specs: Photography/rendering parameters — lens focal length, shot angle, depth of field, lighting quality, texture sharpness, film grain.
Text Rendering: If text is required, preserve it verbatim in double quotes with font style, color, material, and exact position.

Five Mandatory Rules:
1. Knowledge Resolution: Resolve any cultural reference, landmark, historical figure, formula, or named artwork into its specific visible characteristics. Never leave abstract labels in the prompt.
2. Spatial Anchoring: Replace all vague spatial language with precise positional descriptions (top-left corner, centered foreground, 30% from left edge).
3. Text Precision: Preserve any required text verbatim in double quotes with full rendering spec.
4. Real-World Accuracy: Use internal knowledge to fill in factually accurate visual details when the scene requires it.
5. Abstract Concretization: Translate emotional or abstract words (freedom, loneliness, tension) into concrete visible scene elements, symbols, lighting, and color.

Output Format:
Wrap SCALIST reasoning in <ANALYSIS> tags.
Wrap the final prompt in <PROMPT> tags.
The <PROMPT> content must be a single fluent paragraph in plain text — no markdown, no bullet points, no headers."""

            user_message = (
                f"USER REQUEST:\n'{user_input.strip()}'\n\n"
                "Apply the full HiDream SCALIST v2 framework. "
                "Wrap dimensional analysis in <ANALYSIS> tags and the final prompt in <PROMPT> tags. "
                "Plain text only — no markdown."
            )

        # ── Shared execution ─────────────────────────────────────────────────
        if base64_image is not None:
            user_content_block = [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]
        else:
            user_content_block = user_message

        stop_tokens = ["<end_of_turn>", "<eos>", "<|eot_id|>", "User:", "ASSISTANT:"]
        model_lower = llm_model.lower()
        if "qwen" in model_lower:
            stop_tokens.extend(["<|im_end|>", "<|im_start|>"])
        elif "gemma" in model_lower:
            stop_tokens.extend(["<turn|>", "<start_of_turn>"])

        mode_label = "STRUCTURED" if is_structured else "FREEFORM"
        MAX_ATTEMPTS = 3
        final_p1, final_p2 = f"Error: all {MAX_ATTEMPTS} attempts produced degenerate output.", ""

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                attempt_seed = ((int(seed) + attempt - 1) % (2 ** 32)) if seed != -1 else None
                response = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content_block}
                    ],
                    max_tokens=token_val,
                    temperature=temp,
                    stop=stop_tokens,
                    seed=attempt_seed
                )
                raw_result = response['choices'][0]['message']['content'].strip()

                if self._has_tail_repetition(raw_result):
                    print(f"[RT-LTX-2] HiDream ({mode_label}): attempt {attempt}/{MAX_ATTEMPTS} degenerate, retrying...")
                    continue

                analysis_match = re.search(r"<ANALYSIS>\s*(.*?)\s*</ANALYSIS>", raw_result, re.DOTALL | re.IGNORECASE)
                prompt_match   = re.search(r"<PROMPT>\s*(.*?)\s*</PROMPT>",     raw_result, re.DOTALL | re.IGNORECASE)

                if prompt_match:
                    raw_p1 = prompt_match.group(1).strip()
                    raw_p2 = analysis_match.group(1).strip() if analysis_match else "Analysis tags missing."
                else:
                    sections = re.split(r"(?i)\*\*(?:Generated|Final|Enhanced) Prompt[:\*]*\*\*|### (?:2\.|Final|Enhanced)", raw_result)
                    raw_p1 = sections[1].strip() if len(sections) > 1 else raw_result
                    raw_p2 = sections[0].strip() if len(sections) > 1 else "Tag parsing failed."

                # Always scrub markdown from the generation prompt output
                final_p1 = self._scrub_markdown(raw_p1)
                final_p2 = raw_p2
                print(f"[RT-LTX-2] HiDream ({mode_label}): OK (attempt {attempt})")
                break

            except Exception as e:
                final_p1, final_p2 = f"Error (attempt {attempt}): {e}", ""
                print(f"[RT-LTX-2] HiDream ({mode_label}): attempt {attempt}/{MAX_ATTEMPTS} error: {e}")

        if not keep_model_loaded:
            self.unload_model()
        return (final_p1, final_p2, frame_count)


# ====================================================================================================
# SECTION 5: OPTION 04 (SCOPE PROMPT ENHANCER - FULL TWO-PASS PIPELINE)
# ====================================================================================================
# Source: nopnor/SCOPE (github.com/nopnor/SCOPE)
# Full workflow: Decompose → Reason (unknowns) → Synthesize → Verify → Repair
# Adapted to a two-LLM-call pipeline for local ComfyUI execution.
# ────────────────────────────────────────────────────────────────────────────

    # ── Helper: extract text between XML tags ──────────────────────────────
    @staticmethod
    def _extract_tag(text: str, tag: str) -> str | None:
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    def _run_option_04(self, llm_model, user_input, seed, keep_model_loaded, frame_count, base64_image, token_val, temp):

        model_lower = llm_model.lower()
        stop_tokens = ["<end_of_turn>", "<eos>", "<|eot_id|>", "User:", "ASSISTANT:"]
        if "qwen" in model_lower:  stop_tokens.extend(["<|im_end|>", "<|im_start|>"])
        elif "gemma" in model_lower: stop_tokens.extend(["<turn|>", "<start_of_turn>"])

        MAX_RETRY = 3

        # ══════════════════════════════════════════════════════════════════════
        # PASS 1 — DECOMPOSE + REASON
        # Stage contract mirrors nopnor/SCOPE: entities (E), constraints (C),
        # unknowns (U). Unknowns are resolved inline:
        #   - external_reference: cultural refs, landmarks, facts, formulas
        #   - semantic_reasoning: abstract words → concrete visible details
        # Temperature is forced LOW (0.2) for analytical accuracy.
        # ══════════════════════════════════════════════════════════════════════
        DECOMPOSE_SYSTEM = """You are the SCOPE Decomposer and Reasoner for image/video generation.

Your job is to parse a raw user prompt into a structured semantic specification z = (E, C, U) and then immediately resolve all unknowns into explicit visual descriptions. Work through ALL stages below in ONE response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1 — DECOMPOSE: Build z = (E, C, U)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

E (Entities): List every visual object, character, creature, or element that MUST appear in the scene. For each entity, assign:
  - id: short label (e.g. E1, E2)
  - name: what it is
  - priority: critical | primary | secondary

C (Constraints): For each entity, list ALL verifiable visual commitments. Sort by these three sub-types:
  - ATTRIBUTE constraints: appearance, color, material, texture, clothing, identity, expression, age, size
  - RELATION constraints: interactions, contact, relative positions BETWEEN entities
  - LAYOUT constraints: absolute placement in frame (top-left, centered, foreground), background elements, depth, negative space

  For each constraint: assign id (C1, C2…), type (attribute/relation/layout), priority (critical/primary/secondary), and a clear one-line commitment statement.

U (Unknowns): Identify every gap, vague term, abstract concept, or missing visual detail. Classify each as:
  - external_reference: needs factual/cultural knowledge (e.g. "Mona Lisa", "E=mc²", "Dunkirk", historical figures)
  - semantic_reasoning: needs abstract→concrete translation (e.g. "loneliness", "futuristic", "tension", "healing")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2 — REASON: Resolve All Unknowns
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For every U item:
  - external_reference → use internal encyclopedic knowledge to describe the actual visible characteristics (what does it look like? colors, shapes, key features, era-accurate details, accurate text/formulas verbatim)
  - semantic_reasoning → translate to concrete visible scene elements, symbols, colors, and atmosphere

Replace every vague or abstract phrase with specific, model-executable visual language.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — use EXACTLY these XML tags
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<ENTITIES>
E1 [critical]: <name and resolved visual description>
E2 [primary]: <name and resolved visual description>
...
</ENTITIES>

<ATTRIBUTE_CONSTRAINTS>
C1 [critical, E1]: <specific verifiable commitment about appearance, material, clothing, etc.>
C2 [primary, E1]: ...
</ATTRIBUTE_CONSTRAINTS>

<RELATION_CONSTRAINTS>
C3 [critical, E1↔E2]: <specific verifiable commitment about interaction or relative position>
...
</RELATION_CONSTRAINTS>

<LAYOUT_CONSTRAINTS>
C4 [critical, E1]: <exact placement in frame, depth, background>
...
</LAYOUT_CONSTRAINTS>

<RESOLVED_UNKNOWNS>
U1 [external_reference]: original="<original vague text>" → resolved="<explicit visual description>"
U2 [semantic_reasoning]: original="<abstract word>" → resolved="<concrete visible scene details>"
...
</RESOLVED_UNKNOWNS>"""

        decompose_user = (
            f"USER REQUEST:\n'{user_input.strip()}'\n\n"
            "Run SCOPE STAGE 1 (Decompose) and STAGE 2 (Reason). "
            "Output ALL five XML sections: <ENTITIES>, <ATTRIBUTE_CONSTRAINTS>, "
            "<RELATION_CONSTRAINTS>, <LAYOUT_CONSTRAINTS>, <RESOLVED_UNKNOWNS>."
        )

        if base64_image is not None:
            decompose_content = [
                {"type": "text", "text": decompose_user},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]
        else:
            decompose_content = decompose_user

        decomposition_text = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                resp1 = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": DECOMPOSE_SYSTEM},
                        {"role": "user",   "content": decompose_content}
                    ],
                    max_tokens=min(token_val, 1024),  # decomposition doesn't need max tokens
                    temperature=0.2,                   # low temp for analytical accuracy
                    stop=stop_tokens,
                    seed=((int(seed) + attempt - 1) % (2 ** 32)) if seed != -1 else None
                )
                raw1 = resp1['choices'][0]['message']['content'].strip()
                if not self._has_tail_repetition(raw1):
                    decomposition_text = raw1
                    print(f"[RT-LTX-2] SCOPE Pass 1 (Decompose+Reason): OK (attempt {attempt})")
                    break
                print(f"[RT-LTX-2] SCOPE Pass 1: attempt {attempt} degenerate, retrying...")
            except Exception as e:
                print(f"[RT-LTX-2] SCOPE Pass 1 error (attempt {attempt}): {e}")
                decomposition_text = None

        if decomposition_text is None:
            if not keep_model_loaded: self.unload_model()
            return ("Error: SCOPE Pass 1 (Decompose) failed after all retries.", "", frame_count)

        # ══════════════════════════════════════════════════════════════════════
        # PASS 2 — SYNTHESIZE → VERIFY → REPAIR
        # Takes the resolved specification from Pass 1 and synthesizes a prompt.
        # Then self-verifies each entity and each critical/primary constraint.
        # If any commitment is violated, repairs inline before producing the
        # final output — mirroring SCOPE's verify→repair loop.
        # ══════════════════════════════════════════════════════════════════════
        SYNTHESIZE_SYSTEM = """You are the SCOPE Synthesizer, Verifier, and Repairer for image/video generation.

You receive a fully resolved SCOPE specification (entities, attribute/relation/layout constraints, and resolved unknowns). Your task is three stages in one response:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3 — SYNTHESIZE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a single, fluent, detailed English paragraph that:
  - Includes EVERY entity from E, using its resolved visual description
  - Honors ALL constraints from C (attribute, relation, layout)
  - Uses the RESOLVED values from U — never the original vague/abstract terms
  - Reads like a Creative Director's brief: complete sentences, rich precise adjectives, photographic/cinematic terminology
  - Is fully self-contained — someone should generate the correct image from this prompt alone

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 4 — VERIFY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Re-read your synthesized prompt and check it against the specification:
  - For each entity: is it clearly present and visually described?
  - For each CRITICAL constraint: is the commitment explicitly honored?
  - For each PRIMARY constraint: is the commitment clearly present?
  - Are all resolved unknowns used (not the original vague words)?

Mark each check as PASS or FAIL with a brief note.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 5 — REPAIR (only if needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If ANY critical or primary constraint FAILED verification, rewrite the prompt to fix every failing point.
If all checks passed, copy the prompt unchanged.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — use EXACTLY these XML tags
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<DRAFT_PROMPT>
[Your first synthesized paragraph here]
</DRAFT_PROMPT>

<VERIFICATION>
Entity checks:
  - E1 (<name>): PASS/FAIL — <note>
  - E2 (<name>): PASS/FAIL — <note>
Constraint checks:
  - C1 [critical]: PASS/FAIL — <note>
  - C2 [critical]: PASS/FAIL — <note>
  ...
Repair needed: YES / NO
</VERIFICATION>

<PROMPT>
[Final repaired (or unchanged) single-paragraph prompt ready for image/video generation]
</PROMPT>"""

        synthesize_user = (
            f"ORIGINAL USER REQUEST:\n'{user_input.strip()}'\n\n"
            f"RESOLVED SCOPE SPECIFICATION FROM PASS 1:\n{decomposition_text}\n\n"
            "Run SCOPE STAGE 3 (Synthesize), STAGE 4 (Verify), and STAGE 5 (Repair if needed). "
            "Output: <DRAFT_PROMPT>, <VERIFICATION>, and <PROMPT> tags."
        )

        final_p1 = f"Error: SCOPE Pass 2 (Synthesize+Verify) failed after {MAX_RETRY} retries."
        final_p2 = decomposition_text  # Always return decomposition in PROMPT 2 even on failure

        for attempt in range(1, MAX_RETRY + 1):
            try:
                resp2 = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYNTHESIZE_SYSTEM},
                        {"role": "user",   "content": synthesize_user}
                    ],
                    max_tokens=token_val,
                    temperature=temp,
                    stop=stop_tokens,
                    seed=((int(seed) + 100 + attempt - 1) % (2 ** 32)) if seed != -1 else None  # different seed range from pass 1
                )
                raw2 = resp2['choices'][0]['message']['content'].strip()

                if self._has_tail_repetition(raw2):
                    print(f"[RT-LTX-2] SCOPE Pass 2: attempt {attempt} degenerate, retrying...")
                    continue

                # Extract final <PROMPT> (post-repair)
                prompt_text   = self._extract_tag(raw2, "PROMPT")
                verify_text   = self._extract_tag(raw2, "VERIFICATION")
                draft_text    = self._extract_tag(raw2, "DRAFT_PROMPT")

                if prompt_text:
                    final_p1 = prompt_text
                elif draft_text:
                    final_p1 = draft_text  # fallback: use draft if repair tag missing
                else:
                    final_p1 = raw2  # last resort

                # Build PROMPT 2: full SCOPE trace for the user to inspect
                trace_parts = ["═══ SCOPE DECOMPOSITION & REASONING (Pass 1) ═══\n", decomposition_text]
                if verify_text:
                    trace_parts += ["\n\n═══ SCOPE VERIFICATION & REPAIR LOG (Pass 2) ═══\n", verify_text]
                final_p2 = "".join(trace_parts)

                print(f"[RT-LTX-2] SCOPE Pass 2 (Synthesize+Verify+Repair): OK (attempt {attempt})")
                break

            except Exception as e:
                print(f"[RT-LTX-2] SCOPE Pass 2 error (attempt {attempt}): {e}")

        if not keep_model_loaded:
            self.unload_model()

        # PROMPT 1: final verified+repaired generation prompt
        # PROMPT 2: full SCOPE trace (decomposition + verification log)
        return (final_p1, final_p2, frame_count)

# ====================================================================================================
# SECTION 6: OPTION 05 (FLUX PROMPT ENHANCER - NOVELIST PROSE)
# ====================================================================================================
    def _run_option_05(self, llm_model, user_input, seed, keep_model_loaded, frame_count, base64_image, token_val, temp):
        SYSTEM_PROMPT = """You are a master prompt enhancement assistant for the FLUX.2 [klein] image generation model. Your role is to transform brief user requests into extremely detailed, long-form, novelist-style cinematic prose descriptions that maximize image quality and narrative depth.

CORE PRINCIPLES:
- Write Like a Master Novelist: Convert keywords into expansive, flowing, highly descriptive prose. Absolutely NO comma-separated lists, bullet points, or search-engine style keywords.
- Maximum Granularity: FLUX.2 [klein] does NOT auto-enhance. You must explicitly describe every micro-detail: fabric textures, precise colors, character postures, facial features, weather, and environment architecture.
- Length: Output should be long-form and comprehensive (typically 200-400 words).

PROMPT STRUCTURE FRAMEWORK (Flow these seamlessly together into continuous paragraphs):
1. Camera & Framing: Start with the shot type (e.g., medium-wide, close-up) and camera angle.
2. Subject(s) & Action: Front-load the main subjects. If multiple characters exist, describe each one's physical traits, specific clothing, posture, and expression in dedicated, flowing sentences. Ensure they occupy distinct spaces.
3. Setting & Environment: Describe the exact location, ground textures, background architecture, and weather/atmosphere.
4. Lighting (THE MOST CRITICAL ELEMENT): Describe the lighting with technical and atmospheric precision. Include the primary light source, direction, quality (soft/harsh), shadows, and any secondary or ambient light. Describe how the light interacts with textures, rain, or characters.
5. Footer: Append exact style and mood descriptors at the end formatted strictly as: "Style: [aesthetic]. Mood: [tone]."

CRITICAL RULES:
- Write in complete, flowing, cinematic paragraphs.
- Describe lighting in extreme detail EVERY time.
- Use highly sensory, evocative, and specific language.
- DO NOT use bullet points, XML tags, or checklists in the final output.
- DO NOT include meta-text like "Here is the prompt" or "I have enhanced it." Output ONLY the enhanced prose.

OUTPUT:
Provide ONLY the final enhanced prose paragraph(s), followed by the "Style: [...] Mood: [...]" footer."""

        final_prompt = f"USER REQUEST:\n'{user_input.strip()}'\n\nEnhance this into an expansive, highly detailed, long-form cinematic prose description (250+ words) for FLUX.2. Ensure the shot type, subjects, extreme character details, environment, and complex lighting are woven into a flowing narrative."
        
        if base64_image is not None:
            user_content_block = [{"type": "text", "text": final_prompt}, {"type": "image_url", "image_url": {"url": base64_image}}]
        else:
            user_content_block = final_prompt

        stop_tokens = ["<end_of_turn>", "<eos>", "<|eot_id|>", "User:", "ASSISTANT:"]
        model_lower = llm_model.lower()
        if "qwen" in model_lower: stop_tokens.extend(["<|im_end|>", "<|im_start|>"])
        elif "gemma" in model_lower: stop_tokens.extend(["<turn|>", "<start_of_turn>"])

        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content_block}],
                max_tokens=token_val, temperature=temp, stop=stop_tokens, seed=(int(seed) % (2 ** 32)) if seed != -1 else None
            )
            raw_result = response['choices'][0]['message']['content'].strip()
            
            # Clean up any conversational prefixes the LLM might try to inject
            final_result = re.sub(r"^(Enhanced Output|Enhanced Prompt|Output|Prompt):\s*", "", raw_result, flags=re.IGNORECASE).strip()
            final_result = final_result.replace("**", "") # Remove bolding if the LLM tries to style the output
        except Exception as e:
            final_result = f"Error: {e}"

        if not keep_model_loaded: self.unload_model()
        
        # Flux option outputs the prose to PROMPT 1, leaving PROMPT 2 empty
        return (final_result, "", frame_count)




# ── NODE MAPPINGS ──
NODE_CLASS_MAPPINGS = { "RT_LTX2_RoyalPrompt": RT_LTX2_RoyalPrompt }
NODE_DISPLAY_NAME_MAPPINGS = { "RT_LTX2_RoyalPrompt": "RT-LTX-2 Royal Prompt by RareTutor" }