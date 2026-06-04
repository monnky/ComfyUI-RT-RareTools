from .RT_Nodes.rt_ltx2_self_refining_patch import NODE_CLASS_MAPPINGS as PATCH_NODES, NODE_DISPLAY_NAME_MAPPINGS as PATCH_NAMES
from .RT_Nodes.rt_ltx2_tiled_vae_encoder import NODE_CLASS_MAPPINGS as ENCODE_NODES, NODE_DISPLAY_NAME_MAPPINGS as ENCODE_NAMES
from .RT_Nodes.rt_ltx2_streaming_decoder import NODE_CLASS_MAPPINGS as DECODE_NODES, NODE_DISPLAY_NAME_MAPPINGS as DECODE_NAMES
from .RT_Nodes.rt_ltx2_tiled_vae_decoder import NODE_CLASS_MAPPINGS as TILED_DECODE_NODES, NODE_DISPLAY_NAME_MAPPINGS as TILED_DECODE_NAMES
from .RT_Nodes.rt_ltx2_stg_guider import NODE_CLASS_MAPPINGS as STG_NODES, NODE_DISPLAY_NAME_MAPPINGS as STG_NAMES
from .RT_Nodes.rt_ltx2_royal_prompt import NODE_CLASS_MAPPINGS as PROMPT_NODES, NODE_DISPLAY_NAME_MAPPINGS as PROMPT_NAMES
from .RT_Nodes.rt_ltx2_longvideo import NODE_CLASS_MAPPINGS as LONG_VIDEO_NODES, NODE_DISPLAY_NAME_MAPPINGS as LONG_VIDEO_NAMES
from .RT_Nodes.rt_ltx2_basic_utils import NODE_CLASS_MAPPINGS as BASIC_NODES, NODE_DISPLAY_NAME_MAPPINGS as BASIC_NAMES

# Import for the Spectrum Acceleration Node
from .RT_Nodes.rt_ltx2_acceleration1 import NODE_CLASS_MAPPINGS as ACCEL_NODES, NODE_DISPLAY_NAME_MAPPINGS as ACCEL_NAMES

# New Import for the Video-Only LoRA Node
from .RT_Nodes.rt_ltx2_video_lora import NODE_CLASS_MAPPINGS as VIDEO_LORA_NODES, NODE_DISPLAY_NAME_MAPPINGS as VIDEO_LORA_NAMES

# Import for the 6-Character Identity Reference Node
# from .RT_Nodes.rt_ltx2_batch_latent_attention import NODE_CLASS_MAPPINGS as BATCH_ATTN_NODES, NODE_DISPLAY_NAME_MAPPINGS as BATCH_ATTN_NAMES

NODE_CLASS_MAPPINGS = {
    **PATCH_NODES, 
    **ENCODE_NODES, 
    **DECODE_NODES,
    **TILED_DECODE_NODES,
    **STG_NODES,
    **PROMPT_NODES,
    **LONG_VIDEO_NODES,
    **ACCEL_NODES,
    **VIDEO_LORA_NODES, # Added the new Video-Only LoRA node
    # **BATCH_ATTN_NODES,  # Keeps existing nodes and adds the multi-char node
    **BASIC_NODES, # Add this line
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **PATCH_NAMES, 
    **ENCODE_NAMES, 
    **DECODE_NAMES,
    **TILED_DECODE_NAMES,
    **STG_NAMES,
    **PROMPT_NAMES,
    **LONG_VIDEO_NAMES,
    **ACCEL_NAMES,
    **VIDEO_LORA_NAMES, # Added the new Video-Only LoRA node names
    # **BATCH_ATTN_NAMES,
    **BASIC_NAMES, # Add this line
}

# Tell ComfyUI to load the custom javascript files from the 'web' folder
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']