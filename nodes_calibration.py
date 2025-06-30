import math
import torch
import comfy.ldm.common_dit
import comfy.model_management as mm
import numpy as np

from torch import Tensor
from einops import repeat
from typing import Optional
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding, apply_mod
from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from comfy.ldm.wan.model import sinusoidal_embedding_1d


def magcache_flux_calibration_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        total_infer_steps = transformer_options.get("total_infer_steps")
    
        if not hasattr(self, 'calibration_data'):
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = None
        
        
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        txt = self.txt_in(txt)

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        blocks_replace = patches_replace.get("dit", {})

        ori_img = img.clone()
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                txt=args["txt"],
                                                vec=args["vec"],
                                                pe=args["pe"],
                                                attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                        "txt": txt,
                                                        "vec": vec,
                                                        "pe": pe,
                                                        "attn_mask": attn_mask},
                                                        {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                txt=txt,
                                vec=vec,
                                pe=pe,
                                attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

            # PuLID attention
            if getattr(self, "pulid_data", {}):
                if i % self.pulid_double_interval == 0:
                    # Will calculate influence of all pulid nodes at once
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps)
                                    & (timesteps >= node_data['sigma_end'])):
                            img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                    ca_idx += 1

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                    vec=args["vec"],
                                    pe=args["pe"],
                                    attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                        "vec": vec,
                                                        "pe": pe,
                                                        "attn_mask": attn_mask}, 
                                                        {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add

            # PuLID attention
            if getattr(self, "pulid_data", {}):
                real_img, txt = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
                if i % self.pulid_single_interval == 0:
                    # Will calculate influence of all nodes at once
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps)
                                    & (timesteps >= node_data['sigma_end'])):
                            real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], real_img)
                    ca_idx += 1
                img = torch.cat((txt, real_img), 1)

        img = img[:, txt.shape[1] :, ...]
        cur_residual = img - ori_img
        if self.calibration_data['step_count'] >= 1:
            # Calculate calibration metrics
            norm_ratio = (cur_residual.norm(dim=-1) / self.previous_residual.norm(dim=-1)).mean().item()
            norm_std = (cur_residual.norm(dim=-1) / self.previous_residual.norm(dim=-1)).std().item()
            cos_dist = (1 - torch.nn.functional.cosine_similarity(cur_residual, self.previous_residual, dim=-1, eps=1e-8)).mean().item()
            
            # Store metrics
            self.calibration_data['norm_ratios'].append(round(norm_ratio, 5))
            self.calibration_data['norm_stds'].append(round(norm_std, 5))
            self.calibration_data['cos_dists'].append(round(cos_dist, 5))
            if self.calibration_data['step_count'] >= (total_infer_steps-1):
                print("mag_ratios")
                print(self.calibration_data['norm_ratios'])
                print("mag_ratio_std")
                print(self.calibration_data['norm_stds'])
                print("mag_cos_dist")
                print(self.calibration_data['cos_dists'])
        self.previous_residual = cur_residual
        self.calibration_data['step_count'] += 1  
        if total_infer_steps == self.calibration_data['step_count']: # del cache when calibration multiple times
            del self.calibration_data
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = None
            
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        
        return img

def magcache_chroma_calibration_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        total_infer_steps = transformer_options.get("total_infer_steps")
    
        if not hasattr(self, 'calibration_data'):
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = [None, None]
        if total_infer_steps*2 == self.calibration_data['step_count']: # del cache when calibration multiple times
            del self.calibration_data
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = [None, None]
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
            
        # running on sequences img
        img = self.img_in(img)
        
        # Chroma-specific modulation vectors setup
        mod_index_length = 344
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)
        modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)
        mod_vectors = self.distilled_guidance_layer(input_vec)
        
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        blocks_replace = patches_replace.get("dit", {})
        
            
        ori_img = img.clone()
        for i, block in enumerate(self.double_blocks):
            if i not in self.skip_mmdit:
                double_mod = (
                    self.get_modulations(mod_vectors, "double_img", idx=i),
                    self.get_modulations(mod_vectors, "double_txt", idx=i),
                )
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"],
                                                        txt=args["txt"],
                                                        vec=args["vec"],
                                                        pe=args["pe"],
                                                        attn_mask=args.get("attn_mask"))
                        return out
                    out = blocks_replace[("double_block", i)]({"img": img,
                                                                "txt": txt,
                                                                "vec": double_mod,
                                                                "pe": pe,
                                                                "attn_mask": attn_mask},
                                                                {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img,
                                    txt=txt,
                                    vec=double_mod,
                                    pe=pe,
                                    attn_mask=attn_mask)
                if control is not None:  # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add
                            
        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            if i not in self.skip_dit:
                single_mod = self.get_modulations(mod_vectors, "single", idx=i)
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"],
                                            vec=args["vec"],
                                            pe=args["pe"],
                                            attn_mask=args.get("attn_mask"))
                        return out
                    out = blocks_replace[("single_block", i)]({"img": img,
                                                                "vec": single_mod,
                                                                "pe": pe,
                                                                "attn_mask": attn_mask},
                                                            {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=single_mod, pe=pe, attn_mask=attn_mask)
                if control is not None:  # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1]:, ...] += add
                            
        img = img[:, txt.shape[1]:, ...]
       
        cur_residual = img - ori_img
        if self.calibration_data['step_count'] >= 2:
            # Calculate calibration metrics
            norm_ratio = (cur_residual.norm(dim=-1) / self.previous_residual[self.calibration_data["step_count"]%2].norm(dim=-1)).mean().item()
            norm_std = (cur_residual.norm(dim=-1) / self.previous_residual[self.calibration_data["step_count"]%2].norm(dim=-1)).std().item()
            cos_dist = (1 - torch.nn.functional.cosine_similarity(cur_residual, self.previous_residual[self.calibration_data["step_count"]%2], dim=-1, eps=1e-8)).mean().item()
            
            # Store metrics
            self.calibration_data['norm_ratios'].append(round(norm_ratio, 5))
            self.calibration_data['norm_stds'].append(round(norm_std, 5))
            self.calibration_data['cos_dists'].append(round(cos_dist, 5))
            if self.calibration_data['step_count'] >= (total_infer_steps*2-1):
                print("mag_ratios")
                print(self.calibration_data['norm_ratios'])
                print("mag_ratio_std")
                print(self.calibration_data['norm_stds'])
                print("mag_cos_dist")
                print(self.calibration_data['cos_dists'])
        self.previous_residual[self.calibration_data["step_count"]%2] = cur_residual.detach()
        self.calibration_data['step_count'] += 1    
        # print(self.calibration_data['step_count'])
        final_mod = self.get_modulations(mod_vectors, "final")
        img = self.final_layer(img, vec=final_mod)
        
        return img

class MagCacheCalibration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the MagCache will be applied to."}),
                "model_type": (["chroma_calibration", "flux_calibration", "flux_kontext_calibration"], {"default": "chroma_calibration", "tooltip": "Supported diffusion model."}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_magcache"
    CATEGORY = "MagCacheCalibration"
    TITLE = "MagCache Calibration"
    
    def apply_magcache(self, model, model_type: str):
        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
        diffusion_model = new_model.get_model_object("diffusion_model")

        if "chroma_calibration" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_chroma_calibration_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "flux_calibration" in model_type or "flux_kontext_calibration" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_flux_calibration_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        else:
            raise ValueError(f"Unknown type {model_type}")
        
        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            cond_or_uncond = kwargs["cond_or_uncond"]
            # referenced from https://github.com/kijai/ComfyUI-KJNodes/blob/d126b62cebee81ea14ec06ea7cd7526999cb0554/nodes/model_optimization_nodes.py#L868
            sigmas = c["transformer_options"]["sample_sigmas"]
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                current_step_index = 0
                for i in range(len(sigmas) - 1):
                    # walk from beginning of steps until crossing the timestep
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
            
            if current_step_index == 0:
                if is_cfg:
                    # uncond first
                    if (1 in cond_or_uncond) and hasattr(diffusion_model, 'magcache_state_state'):
                        delattr(diffusion_model, 'magcache_state_state')
                else:
                    if hasattr(diffusion_model, 'accumulated_rel_l1_distance'):
                        delattr(diffusion_model, 'accumulated_rel_l1_distance')
            total_infer_steps = len(sigmas)-1

            c["transformer_options"]["total_infer_steps"] = total_infer_steps
            with context:
                return model_function(input, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)
    
def patch_optimized_module():
    try:
        from torch._dynamo.eval_frame import OptimizedModule
    except ImportError:
        return

    if getattr(OptimizedModule, "_patched", False):
        return

    def __getattribute__(self, name):
        if name == "_orig_mod":
            return object.__getattribute__(self, "_modules")[name]
        if name in (
            "__class__",
            "_modules",
            "state_dict",
            "load_state_dict",
            "parameters",
            "named_parameters",
            "buffers",
            "named_buffers",
            "children",
            "named_children",
            "modules",
            "named_modules",
        ):
            return getattr(object.__getattribute__(self, "_orig_mod"), name)
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        return delattr(self._orig_mod, name)

    @classmethod
    def __instancecheck__(cls, instance):
        return isinstance(instance, OptimizedModule) or issubclass(
            object.__getattribute__(instance, "__class__"), cls
        )

    OptimizedModule.__getattribute__ = __getattribute__
    OptimizedModule.__delattr__ = __delattr__
    OptimizedModule.__instancecheck__ = __instancecheck__
    OptimizedModule._patched = True

def patch_same_meta():
    try:
        from torch._inductor.fx_passes import post_grad
    except ImportError:
        return

    same_meta = getattr(post_grad, "same_meta", None)
    if same_meta is None:
        return

    if getattr(same_meta, "_patched", False):
        return

    def new_same_meta(a, b):
        try:
            return same_meta(a, b)
        except Exception:
            return False

    post_grad.same_meta = new_same_meta
    new_same_meta._patched = True


NODE_CLASS_MAPPINGS = {
    "MagCacheCalibration": MagCacheCalibration,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
