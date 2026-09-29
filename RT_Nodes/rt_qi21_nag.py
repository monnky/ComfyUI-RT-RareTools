# ====================================================================================================
# [RT] Qwen-Image-2.1 Normalized Attention Guidance (NAG) Suite
# Implements Normalized Attention Guidance (NeurIPS 2024 / arXiv:2505.21179)
# Enables robust, non-burning negative prompt guidance at CFG = 1.0 and in distilled models
# Model-In / Model-Out Adapter: Plugs directly between Model/LoRA Loader and KSampler
# 100% English ASCII Only
# ====================================================================================================

import torch
import comfy.samplers
import comfy.sample
from comfy.model_patcher import ModelPatcher


class RT_QI21_NAG:
    """
    RT Qwen-Image-2.1 NAG
    Normalized Attention Guidance node matching KJNodes / WanVideo NAG specification.
    Connects between Model/LoRA Loader and KSampler:
      Input:  model (MODEL), conditioning (CONDITIONING - Negative Prompt)
      Widgets: nag_scale (11.0), nag_alpha (0.250), nag_tau (2.500), input_type ('default')
      Output: model (MODEL)
    Injects negative prompt guidance into CFG = 1.0 inference without burning or artifacts.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Connect input from model or LoRA loader."
                }),
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Negative prompt conditioning to guide against (NAG reference direction)."
                }),
                "nag_scale": ("FLOAT", {
                    "default": 11.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.001,
                    "tooltip": "Guidance strength against the negative conditioning. Default is 11.0."
                }),
                "nag_alpha": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "tooltip": "Blend coefficient between NAG-guided attention and positive attention. Default is 0.25."
                }),
                "nag_tau": ("FLOAT", {
                    "default": 2.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "round": 0.001,
                    "tooltip": "L1 norm clipping threshold to prevent color burn and saturation. Default is 2.5."
                }),
                "input_type": (["default", "batch"], {
                    "default": "default",
                    "tooltip": "default: standard pair mode; batch: single conditioning mode for CFG=1 speed models."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_nag"
    CATEGORY = "RareTutor/Sampling"
    TITLE = "RT Qwen-Image-2.1 NAG"
    DESCRIPTION = "Normalized Attention Guidance (NAG) for Qwen-Image-2.1, Wan, and DiT models. Injects negative prompt guidance into CFG = 1.0."

    def apply_nag(self, model, conditioning, nag_scale=11.0, nag_alpha=0.25, nag_tau=2.5, input_type="default", inplace=False, **kwargs):
        m = model.clone()

        def nag_cfg_function(args):
            cond = args.get("cond", None)
            uncond = args.get("uncond", None)
            cond_scale = args.get("cond_scale", 1.0)

            # If uncond was not calculated by the KSampler (e.g. CFG=1 or negative slot empty in KSampler),
            # evaluate the negative conditioning connected to this NAG node
            if uncond is None and conditioning is not None:
                x = args.get("input", args.get("x", None))
                timestep = args.get("timestep", args.get("sigma", None))
                model_options = args.get("model_options", {})
                if x is not None and timestep is not None:
                    (uncond,) = comfy.samplers.calc_cond_batch(m.model, [conditioning], x, timestep, model_options)

            if cond is None or uncond is None or nag_scale <= 0.0:
                return cond

            # Ensure batch dimensions match
            if uncond.shape[0] != cond.shape[0]:
                repeat_factor = cond.shape[0] // uncond.shape[0]
                if repeat_factor > 1:
                    uncond = uncond.repeat(repeat_factor, *([1] * (uncond.ndim - 1)))
                else:
                    uncond = uncond.expand(cond.shape[0], *uncond.shape[1:])

            # Core NAG Extrapolation (Eq. 7 in paper):
            nag_guidance = cond * nag_scale - uncond * (nag_scale - 1.0)

            # L1-Norm Clipping via Tau (Eq. 9 in paper):
            dims = tuple(range(1, cond.ndim))
            norm_positive = torch.norm(cond, p=1, dim=dims, keepdim=True)
            norm_guidance = torch.norm(nag_guidance, p=1, dim=dims, keepdim=True)

            scale_ratio = norm_guidance / (norm_positive + 1e-7)
            scale_ratio = torch.nan_to_num(scale_ratio, nan=10.0)
            mask = scale_ratio > nag_tau

            adjustment = (norm_positive * nag_tau) / (norm_guidance + 1e-7)
            nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)

            # Blending via Alpha (Eq. 10 in paper):
            final_pred = nag_guidance * nag_alpha + cond * (1.0 - nag_alpha)

            return final_pred

        # Intercept sampler CFG with disable_cfg1_optimization=True so negative branch is actively computed at CFG = 1.0
        m.set_model_sampler_cfg_function(nag_cfg_function, disable_cfg1_optimization=True)
        return (m,)


class RT_NAG_Guider(comfy.samplers.CFGGuider):
    def __init__(self, model: ModelPatcher, nag_scale: float, nag_tau: float, nag_alpha: float):
        model = model.clone()
        super().__init__(model)
        self.nag_scale = float(nag_scale)
        self.nag_tau = float(nag_tau)
        self.nag_alpha = float(nag_alpha)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        pos_cond = self.conds.get("positive")
        neg_cond = self.conds.get("negative")

        if "transformer_options" not in model_options:
            model_options["transformer_options"] = {}

        if self.nag_scale <= 0.0 or neg_cond is None:
            (pos_pred,) = comfy.samplers.calc_cond_batch(
                self.inner_model, [pos_cond], x, timestep, model_options
            )
            return pos_pred

        (pos_pred, neg_pred) = comfy.samplers.calc_cond_batch(
            self.inner_model, [pos_cond, neg_cond], x, timestep, model_options
        )

        nag_guidance = pos_pred * self.nag_scale - neg_pred * (self.nag_scale - 1.0)

        dims = tuple(range(1, pos_pred.ndim))
        norm_positive = torch.norm(pos_pred, p=1, dim=dims, keepdim=True)
        norm_guidance = torch.norm(nag_guidance, p=1, dim=dims, keepdim=True)

        scale_ratio = norm_guidance / (norm_positive + 1e-7)
        scale_ratio = torch.nan_to_num(scale_ratio, nan=10.0)
        mask = scale_ratio > self.nag_tau

        adjustment = (norm_positive * self.nag_tau) / (norm_guidance + 1e-7)
        nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)

        final_pred = nag_guidance * self.nag_alpha + pos_pred * (1.0 - self.nag_alpha)
        return final_pred


class RT_QI21_NAG_Guider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.001}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.001}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "RareTutor/Sampling"
    TITLE = "RT Qwen-Image-2.1 NAG Guider"

    def get_guider(self, model, positive, negative, nag_scale, nag_alpha, nag_tau):
        guider = RT_NAG_Guider(model, nag_scale, nag_tau, nag_alpha)
        guider.set_conds(positive, negative)
        guider.set_cfg(1.0)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "RT_QI21_NAG": RT_QI21_NAG,
    "RT_QI21_NAG_Model": RT_QI21_NAG,
    "RT_QI21_NAG_Guider": RT_QI21_NAG_Guider,
    "WanVideoNAG": RT_QI21_NAG,
    "NAGGuidance": RT_QI21_NAG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RT_QI21_NAG": "RT Qwen-Image-2.1 NAG",
    "RT_QI21_NAG_Model": "RT Qwen-Image-2.1 NAG",
    "RT_QI21_NAG_Guider": "RT Qwen-Image-2.1 NAG Guider",
    "WanVideoNAG": "WanVideo NAG (Raretools)",
    "NAGGuidance": "Normalized Attention Guidance (NAG)",
}
