from .RT_Nodes.rt_ltx2_self_refining_patch import NODE_CLASS_MAPPINGS as PATCH_NODES, NODE_DISPLAY_NAME_MAPPINGS as PATCH_NAMES
from .RT_Nodes.rt_ltx2_stg_guider import NODE_CLASS_MAPPINGS as STG_NODES, NODE_DISPLAY_NAME_MAPPINGS as STG_NAMES
from .RT_Nodes.rt_ltx2_royal_prompt import NODE_CLASS_MAPPINGS as PROMPT_NODES, NODE_DISPLAY_NAME_MAPPINGS as PROMPT_NAMES
from .RT_Nodes.rt_ltx2_longvideo import NODE_CLASS_MAPPINGS as LONG_VIDEO_NODES, NODE_DISPLAY_NAME_MAPPINGS as LONG_VIDEO_NAMES
from .RT_Nodes.rt_ltx2_video_lora import NODE_CLASS_MAPPINGS as VIDEO_LORA_NODES, NODE_DISPLAY_NAME_MAPPINGS as VIDEO_LORA_NAMES
from .RT_Nodes.rt_ltx2_basic_utils import NODE_CLASS_MAPPINGS as BASIC_NODES, NODE_DISPLAY_NAME_MAPPINGS as BASIC_NAMES

# Import for Qwen Image 2.1 Prompt Enhancer
try:
    from .RT_Nodes.rt_qwen_image_prompt_enhancer import NODE_CLASS_MAPPINGS as QWEN_PE_NODES, NODE_DISPLAY_NAME_MAPPINGS as QWEN_PE_NAMES
except ImportError:
    QWEN_PE_NODES, QWEN_PE_NAMES = {}, {}

# Import for RT LTX-2.5 Prompt Enhancer
try:
    from .RT_Nodes.rt_ltx_2_5_prompt_enhancer import NODE_CLASS_MAPPINGS as LTX25_PE_NODES, NODE_DISPLAY_NAME_MAPPINGS as LTX25_PE_NAMES
except ImportError:
    LTX25_PE_NODES, LTX25_PE_NAMES = {}, {}
    
# Import for Qwen Image 2.1 Normalized Attention Guidance (NAG)
try:
    from .RT_Nodes.rt_qi21_nag import NODE_CLASS_MAPPINGS as NAG_NODES, NODE_DISPLAY_NAME_MAPPINGS as NAG_NAMES
except ImportError:
    NAG_NODES, NAG_NAMES = {}, {}
    
# Import for RT Image Compare Suite
try:
    from .RT_Nodes.rt_Image_compare import NODE_CLASS_MAPPINGS as COMPARE_NODES, NODE_DISPLAY_NAME_MAPPINGS as COMPARE_NAMES
except ImportError:
    COMPARE_NODES, COMPARE_NAMES = {}, {}
    
# Import for RT Video Compare Suite
try:
    from .RT_Nodes.rt_video_compare import NODE_CLASS_MAPPINGS as VID_COMPARE_NODES, NODE_DISPLAY_NAME_MAPPINGS as VID_COMPARE_NAMES
except ImportError:
    VID_COMPARE_NODES, VID_COMPARE_NAMES = {}, {}

    

NODE_CLASS_MAPPINGS = {
    **PATCH_NODES, 
    **STG_NODES,
    **PROMPT_NODES,
    **LONG_VIDEO_NODES, # Added the new Long Video node
    **VIDEO_LORA_NODES, # Added the new Video-Only LoRA node
    **BASIC_NODES, # Add this line
    **QWEN_PE_NODES,
    **LTX25_PE_NODES,
    **NAG_NODES,
    **COMPARE_NODES,
    **VID_COMPARE_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **PATCH_NAMES, 
    **STG_NAMES,
    **PROMPT_NAMES,
    **LONG_VIDEO_NAMES, # Added the new Long Video node names
    **VIDEO_LORA_NAMES, # Added the new Video-Only LoRA node names
    **BASIC_NAMES, # Add this line
    **QWEN_PE_NAMES,
    **LTX25_PE_NAMES,
    **NAG_NAMES,
    **COMPARE_NAMES,
    **VID_COMPARE_NAMES,
}

# load the custom javascript files from the 'web' folder
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
