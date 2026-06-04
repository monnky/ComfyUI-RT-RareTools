# rt_ltx2_basic_utils.py
import torch
import torch.nn.functional as F
import re
import numpy as np
import comfy.utils
from nodes import node_helpers

# This is the line that fixes your error!
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

# 001: RT-Text Input Node ######################################################
class RTLTX2TextInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "incoming_text": ("STRING", {"forceInput": True}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "RareTutor/Utils"
    OUTPUT_NODE = True # Crucial: Forces the node to send UI updates

    def process(self, text, incoming_text=None):
        if incoming_text is not None and incoming_text.strip() != "":
            final_text = incoming_text
        else:
            final_text = text
            
        # Returns BOTH the UI update signal AND the string data for the next node
        return {"ui": {"text": [final_text]}, "result": (final_text,)}


# 002: RT-Text Concatenate Node ######################################################
class RTLTX2TextConcatenate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_a": ("STRING", {"multiline": True, "default": ""}),
                "text_b": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "combine"
    CATEGORY = "RareTutor/Utils"

    def combine(self, text_a, text_b):
        # Combines both strings with a simple addition
        combined_text = f"{text_a} {text_b}"
        return (combined_text,)

# 003: RT-Workflow Sticky Note ######################################################
class RTLTX2StickyNote:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "note_text": ("STRING", {"multiline": True, "default": "Write workflow instructions here..."}),
                "color_theme": (["Standard", "Urgent (Red)", "Success (Green)", "Info (Blue)"], {"default": "Standard"}),
            },
            "optional": {
                # Creates the permanent input dot on the left
                "incoming_text": ("STRING", {"forceInput": True}), 
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "process"
    CATEGORY = "RareTutor/Utils"
    OUTPUT_NODE = True # Crucial: Tells ComfyUI this is a valid end-point for a workflow!

    def process(self, note_text, color_theme, incoming_text=None):
        # Determine which text to use (prioritize incoming wire data if connected)
        if incoming_text is not None and incoming_text.strip() != "":
            final_text = incoming_text
        else:
            final_text = note_text
            
        # By returning it as a UI element, ComfyUI knows data successfully reached the end of the line
        return {"ui": {"text": [final_text]}}
    
# 004: RT-HTML/Rich Text Preview ######################################################
class RTLTX2HtmlPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "html_text": ("STRING", {"multiline": True, "default": "<h1>Welcome</h1>\n<b>Bold Text</b> and 🚀 Smileys"}),
            },
            "optional": {
                # Creates the permanent input dot on the left
                "incoming_html": ("STRING", {"forceInput": True}), 
            }
        }

    RETURN_TYPES = () 
    FUNCTION = "process"
    CATEGORY = "RareTutor/Utils"
    OUTPUT_NODE = True # Crucial: Allows this node to finish a workflow

    def process(self, html_text, incoming_html=None):
        # Prioritize incoming wire data if connected
        if incoming_html is not None and incoming_html.strip() != "":
            final_html = incoming_html
        else:
            final_html = html_text
            
        # Send the final string to the UI 
        return {"ui": {"text": [final_html]}}

# 005: RT-Image Scale to Megapixels ######################################################
class RTLTX2ImageScaleToMegapixels:
    def __init__(self):
        # Initializes the native ComfyUI upscaler function
        self.upscale_model_node = ImageUpscaleWithModel()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.05}),
            },
            "optional": {
                "upscale_model_opt": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "RareTutor/Image"
    FUNCTION = "scale_to_megapixels"

    def _resize_image(self, images, target_width, target_height, upscale_method="bicubic"):
        # Helper function to handle standard tensor resizing
        s = images.movedim(-1, 1)
        s = comfy.utils.common_upscale(s, target_width, target_height, upscale_method, "disabled")
        s = s.movedim(1, -1)
        return s

    def scale_to_megapixels(self, images, megapixels, upscale_model_opt=None):
        # 1. Get current dimensions
        width = images.shape[2]
        height = images.shape[1]
        
        # 2. Calculate the target scale factor based on desired megapixels
        scale_by = np.sqrt((megapixels * 1024 * 1024) / (width * height))

        target_width = round(width * scale_by)
        target_height = round(height * scale_by)

        # 3. Scaling down OR minor upscale without a model
        if scale_by <= 1.0 or scale_by < 1.2 or upscale_model_opt is None:
            return (self._resize_image(images, target_width, target_height),)
        
        # 4. Heavy upscale (Uses AI model first, then scales down exactly to target)
        else:
            upscaled_images = self.upscale_model_node.execute(upscale_model_opt, images)[0]
            return (self._resize_image(upscaled_images, target_width, target_height, "center"),)

# 006: RT-Number Counter ######################################################
class RTLTX2NumberCounter:
    def __init__(self):
        # This keeps track of the number between workflow runs
        self.count_value = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_type": (["integer", "float"],),
                "mode": (["increment", "decrement", "increment_to_stop", "decrement_to_stop"],),
                "start": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "stop": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "step": ("FLOAT", {"default": 1.0, "step": 0.01}),
            },
            "optional": {
                "reset_bool": ("BOOLEAN", {"forceInput": True}),
            }
        }

    # We output a generic FLOAT for 'number' to maximize compatibility with other nodes
    RETURN_TYPES = ("FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("number", "float", "int")
    FUNCTION = "count"
    CATEGORY = "RareTutor/Utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # THE MAGIC TRICK: 
        # ComfyUI is lazy and caches outputs if inputs haven't changed.
        # Returning NaN forces ComfyUI to execute this node EVERY time you click Queue.
        return float("NaN")

    def count(self, number_type, mode, start, stop, step, reset_bool=False):
        # 1. Check if we need to initialize or reset
        if self.count_value is None or reset_bool:
            self.count_value = start
        else:
            # 2. Apply counting logic
            if mode == "increment":
                self.count_value += step
            elif mode == "decrement":
                self.count_value -= step
            elif mode == "increment_to_stop":
                self.count_value += step
                if self.count_value >= stop:
                    self.count_value = stop
            elif mode == "decrement_to_stop":
                self.count_value -= step
                if self.count_value <= stop:
                    self.count_value = stop

        # 3. Format the final output based on user selection
        if number_type == "integer":
            result = int(self.count_value)
        else:
            result = float(self.count_value)

        # Returns the number in three formats: generic float, strict float, and strict int
        return (float(result), float(result), int(result))


# 007: RT Flux Prompt Pro 1 (Text Concatenate Pro) ###################################
class RTLTX2TextConcatenatePro:
    @classmethod
    def INPUT_TYPES(s):
        prefix_options = [
            "none",
            # "Professional character design sheet, split view, full face close-up focusing on face on left, full-body and covering only face and hair completely with pure flat white color, standing view on right, white background,",
            "Split view, full face close-up focusing on face on left, full-body standing view on right, white background."
        ]

        dropdown_options = [
            "none",
            
            "--- CLOSE-UP ---",  # Visually separates the list
            "01. front view eye-level shot close-up",
            "02. front-right view eye-level shot close-up",
            "03. right view eye-level shot close-up",
            "04. back-right view eye-level shot close-up",
            "05. back view eye-level shot close-up",
            "06. back-left view eye-level shot close-up",
            "07. left view eye-level shot close-up",
            "08. front-left view eye-level shot close-up",
            "09. front view high-angle shot close-up",
            "10. front-right view high-angle shot close-up",
            "11. right view high-angle shot close-up",
            "12. back-right view high-angle shot close-up",
            "13. back view high-angle shot close-up",
            "14. back-left view high-angle shot close-up",
            "15. left view high-angle shot close-up",
            "16. front-left view high-angle shot close-up",
            "17. front view overhead shot close-up",
            "18. front-right view overhead shot close-up",
            "19. right view overhead shot close-up",
            "20. back-right view overhead shot close-up",
            "21. back view overhead shot close-up",
            "22. back-left view overhead shot close-up",
            "23. left view overhead shot close-up",
            "24. front-left view overhead shot close-up",
            
            "--- MEDIUM SHOT ---", # Visually separates the list
            "25. front view eye-level shot medium shot",
            "26. front-right view eye-level shot medium shot",
            "27. right view eye-level shot medium shot",
            "28. back-right view eye-level shot medium shot",
            "29. back view eye-level shot medium shot",
            "30. back-left view eye-level shot medium shot",
            "31. left view eye-level shot medium shot",
            "32. front-left view eye-level shot medium shot",
            "33. front view high-angle shot medium shot",
            "34. front-right view high-angle shot medium shot",
            "35. right view high-angle shot medium shot",
            "36. back-right view high-angle shot medium shot",
            "37. back view high-angle shot medium shot",
            "38. back-left view high-angle shot medium shot",
            "39. left view high-angle shot medium shot",
            "40. front-left view high-angle shot medium shot",
            "41. front view overhead shot medium shot",
            "42. front-right view overhead shot medium shot",
            "43. right view overhead shot medium shot",
            "44. back-right view overhead shot medium shot",
            "45. back view overhead shot medium shot",
            "46. back-left view overhead shot medium shot",
            "47. left view overhead shot medium shot",
            "48. front-left view overhead shot medium shot",
            
            "--- FULL SHOT ---", # Visually separates the list
            "49. front view eye-level shot full shot",
            "50. front-right view eye-level shot full shot",
            "51. right view eye-level shot full shot",
            "52. back-right view eye-level shot full shot",
            "53. back view eye-level shot full shot",
            "54. back-left view eye-level shot full shot",
            "55. left view eye-level shot full shot",
            "56. front-left view eye-level shot full shot",
            "57. front view high-angle shot full shot",
            "58. front-right view high-angle shot full shot",
            "59. right view high-angle shot full shot",
            "60. back-right view high-angle shot full shot",
            "61. back view high-angle shot full shot",
            "62. back-left view high-angle shot full shot",
            "63. left view high-angle shot full shot",
            "64. front-left view high-angle shot full shot",
            "65. front view overhead shot full shot",
            "66. front-right view overhead shot full shot",
            "67. right view overhead shot full shot",
            "68. back-right view overhead shot full shot",
            "69. back view overhead shot full shot",
            "70. back-left view overhead shot full shot",
            "71. left view overhead shot full shot",
            "72. front-left view overhead shot full shot"
        ]
        
        return {
            "required": {
                # Dummy headings to create visual separation outside the dropdowns
                "___CHARACTER_PRESET___": (["⬇️ Select Below ⬇️"],),
                "Character_Preset": (prefix_options, {"default": "none"}),
                
                "___CAMERA_ANGLE___": (["⬇️ Select Below ⬇️"],),
                "Camera_Angle": (dropdown_options, {"default": "none"}),
                
                "Text_Input_1": ("STRING", {"multiline": True, "default": ""}),
                "Text_Input_2": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "combine"
    CATEGORY = "RareTutor/Utils"

    # We must include the dummy headings in the function arguments even though we ignore them
    def combine(self, ___CHARACTER_PRESET___, Character_Preset, ___CAMERA_ANGLE___, Camera_Angle, Text_Input_1, Text_Input_2):
        parts = []

        # 1. Add the character sheet preset if selected
        if Character_Preset != "none":
            parts.append(Character_Preset)

        # 2. Add the camera angle, stripping out the numbers
        if Camera_Angle != "none" and not Camera_Angle.startswith("---"):
            clean_text_a = re.sub(r'^\d+\.\s*', '', Camera_Angle)
            parts.append(clean_text_a)

        # 3. Add the manual string inputs
        parts.extend([Text_Input_1, Text_Input_2])
        
        # Clean up empty strings to avoid stray commas
        valid_parts = [p.strip() for p in parts if p and p.strip() != ""]
        
        # JOIN MAGIC: Adds a comma and a new line between each valid block of text!
        combined_text = ",\n".join(valid_parts)
        
        return (combined_text,)



# 008: RT Auto Audio Trimmer #################################################
class RTLTX2AudioTrimmer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                # Now correctly accepts the green STRING wire from RT_Storyboard_Preview
                "scene_name": ("STRING", {"forceInput": True}),
                "scene_duration": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "trim_audio"
    CATEGORY = "RareTutor/Audio"

    def trim_audio(self, audio, scene_name, scene_duration):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # --- THE FIX: Extract the number from the string ---
        # This takes "Scene_05" and magically turns it into the integer 5
        try:
            scene_number = int(re.findall(r'\d+', scene_name)[0])
        except IndexError:
            print(f"[⚠️ RT-Audio Notice] Could not find a number in '{scene_name}'. Defaulting to Scene 1.")
            scene_number = 1
        
        # 1. Math: Calculate start time based on standard scene length
        start_time_seconds = (scene_number - 1) * scene_duration
        
        start_sample = int(start_time_seconds * sample_rate)
        duration_samples = int(scene_duration * sample_rate)
        end_sample = start_sample + duration_samples
        
        total_samples = waveform.shape[-1]
        
        # 2. FAILSAFE 1: Audio track ended before this scene even started
        if start_sample >= total_samples:
            print(f"\n[⚠️ RT-Audio Notice] Scene {scene_number}: Source audio ended before this scene. Outputting {scene_duration}s of pure silence.\n")
            trimmed_waveform = torch.zeros((1, waveform.shape[1], duration_samples))
            return ({"waveform": trimmed_waveform, "sample_rate": sample_rate},)
            
        # 3. Slice the audio
        actual_end = min(end_sample, total_samples)
        trimmed_waveform = waveform[:, :, start_sample:actual_end]
        
        # 4. SMART PAD LOGIC: Check if we are short on audio
        if trimmed_waveform.shape[-1] < duration_samples:
            
            missing_samples = duration_samples - trimmed_waveform.shape[-1]
            missing_seconds = missing_samples / sample_rate
            actual_seconds = trimmed_waveform.shape[-1] / sample_rate
            
            print(f"\n---------------------------------------------------------")
            print(f"[⚠️ RT-AUDIO TRIMMER NOTICE] - Timeline Auto-Correction")
            print(f"Scene: {scene_number} ({scene_name})")
            print(f"Requested Duration: {scene_duration}.0s")
            print(f"Remaining Audio: {actual_seconds:.1f}s")
            print(f"Action Taken: Source audio is shorter than the requested duration.")
            print(f"Automatically padded {missing_seconds:.1f}s of silence to maintain perfect video sync.")
            print(f"---------------------------------------------------------\n")
            
            padding = torch.zeros((1, waveform.shape[1], missing_samples))
            trimmed_waveform = torch.cat([trimmed_waveform, padding], dim=-1)
            
        return ({"waveform": trimmed_waveform, "sample_rate": sample_rate},)
    
    
    
# 009: RT Flux2 Encode #################################################



class Flux2MultiRefConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "conditioning": ("CONDITIONING",),
                "max_images_allowed": ("INT", {
                    "default": 4, 
                    "min": 0, 
                    "max": 4,
                    "tooltip": "Maximum number of images to process."
                }),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }
        
        # Adding exactly 4 image inputs as requested
        for i in range(1, 5):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
            
        return inputs

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "add_references"
    CATEGORY = "RT_Flux2/Conditioning"

    def add_references(self, conditioning, max_images_allowed=4, vae=None, **kwargs):
        ref_latents = []
        
        # Collect connected images
        images = []
        for i in range(1, 5):
            img = kwargs.get(f"image_{i}")
            images.append(img)
            
        # Only process up to max_images_allowed
        for i, image in enumerate(images[:max_images_allowed]):
            if image is not None and vae is not None:
                samples = image.movedim(-1, 1)
                
                # Encode via VAE and append to the latents list
                ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))

        # Append the encoded latents to the existing conditioning stream
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, 
                {"reference_latents": ref_latents}, 
                append=True
            )

        return (conditioning,)


# 010: RT Frame Replacer ######################################################
class RTLTX2FrameReplacer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Takes in a batch of images (a video sequence)
                "images": ("IMAGE",),
                
                # The manual box to choose which frame to replace (1-based counting)
                "frame_to_replace": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "replace_frame"
    CATEGORY = "RareTutor/Video"

    def replace_frame(self, images, frame_to_replace):
        batch_size = images.shape[0]

        # Failsafe 1: If it's just a single picture, we can't do anything
        if batch_size < 2:
            print("[⚠️ RT Frame Replacer] Notice: Only 1 image provided. Cannot perform replacement.")
            return (images,)

        # Convert human counting (starts at 1) to Python counting (starts at 0)
        target_index = frame_to_replace - 1
        next_index = target_index + 1

        # Failsafe 2: If the user asks to replace a frame that doesn't exist, or asks 
        # to replace the very last frame (which has no "next" frame to copy from).
        if target_index >= batch_size or next_index >= batch_size:
            print(f"[⚠️ RT Frame Replacer] Notice: Frame {frame_to_replace} is out of bounds or is the last frame. Skipping replacement.")
            return (images,)

        # Clone the tensor to keep other nodes safe
        modified_images = images.clone()

        # THE MAGIC: Copy the next frame over the target frame
        modified_images[target_index] = modified_images[next_index]

        print(f"[RT Frame Replacer] Successfully replaced Frame {frame_to_replace} with Frame {frame_to_replace + 1}.")

        return (modified_images,)

# --- Mapping Registration ---

NODE_CLASS_MAPPINGS = {
    "RTLTX2TextInput": RTLTX2TextInput,
    "RTLTX2TextConcatenate": RTLTX2TextConcatenate,
    "RTLTX2StickyNote": RTLTX2StickyNote,
    "RTLTX2HtmlPreview": RTLTX2HtmlPreview,
    "RTLTX2ImageScaleToMegapixels": RTLTX2ImageScaleToMegapixels,
    "RTLTX2NumberCounter": RTLTX2NumberCounter,
    "RTLTX2TextConcatenatePro": RTLTX2TextConcatenatePro,
    "RTLTX2AudioTrimmer": RTLTX2AudioTrimmer,
    "RT_Flux2MultiRefConditioning": Flux2MultiRefConditioning,
    "RTLTX2FrameReplacer": RTLTX2FrameReplacer # <-- Updated Name!
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RTLTX2TextInput": "RT Text Input",
    "RTLTX2TextConcatenate": "RT Text Concatenate",
    "RTLTX2StickyNote": "RT Sticky Note",
    "RTLTX2HtmlPreview": "RT Note++",
    "RTLTX2ImageScaleToMegapixels": "RT Image Scale to Megapixels",
    "RTLTX2NumberCounter": "RT Number Counter",
    "RTLTX2TextConcatenatePro": "RT Flux Prompt Pro 1",
    "RTLTX2AudioTrimmer": "RT Auto Audio Trimmer",
    "RT_Flux2MultiRefConditioning": "RT FLUX2 Multi-Ref Encode",
    "RTLTX2FrameReplacer": "RT Target Frame Replacer" # <-- Updated Name!
    
}