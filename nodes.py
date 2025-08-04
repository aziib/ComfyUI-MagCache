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


SUPPORTED_MODELS_MAG_RATIOS = {
    "flux": np.array([1.0]+[1.21094, 1.11719, 1.07812, 1.0625, 1.03906, 1.03125, 1.03906, 1.02344, 1.03125, 1.02344, 0.98047, 1.01562, 1.00781, 1.0, 1.00781, 1.0, 1.00781, 1.0, 1.0, 0.99609, 0.99609, 0.98047, 0.98828, 0.96484, 0.95703, 0.93359, 0.89062]),
    "flux_kontext": np.array([1.0]+[1.21875, 1.0625, 1.03125, 1.03125, 1.0, 1.00781, 1.03906, 0.98047, 1.03125, 0.96875, 1.02344, 1.0, 0.99219, 1.02344, 0.98047, 0.95703, 0.98828, 0.98047, 0.88672]),
    "chroma": np.array([1.0]*2+[1.00781, 1.01562, 1.03906, 1.03906, 1.05469, 1.05469, 1.07031, 1.07031, 1.04688, 1.04688, 1.03906, 1.03906, 1.03906, 1.03906, 1.01562, 1.01562, 1.05469, 1.05469, 0.99609, 0.99609, 1.02344, 1.02344, 1.01562, 1.01562, 0.99609, 0.99609, 1.0, 1.0, 0.99219, 0.99219, 1.00781, 1.00781, 1.00781, 1.00781, 0.97656, 0.97656, 0.98828, 0.98828, 0.97266, 0.97266, 1.0, 1.0, 0.93359, 0.93359, 0.94922, 0.94922, 0.92578, 0.92578, 1.0625, 1.0625]),
    "hunyuan_video": np.array([1.0]+[1.0754, 1.27807, 1.11596, 1.09504, 1.05188, 1.00844, 1.05779, 1.00657, 1.04142, 1.03101, 1.00679, 1.02556, 1.00908, 1.06949, 1.05438, 1.02214, 1.02321, 1.03019, 1.00779, 1.03381, 1.01886, 1.01161, 1.02968, 1.00544, 1.02822, 1.00689, 1.02119, 1.0105, 1.01044, 1.01572, 1.02972, 1.0094, 1.02368, 1.0226, 0.98965, 1.01588, 1.02146, 1.0018, 1.01687, 0.99436, 1.00283, 1.01139, 0.97122, 0.98251, 0.94513, 0.97656, 0.90943, 0.85703, 0.75456]),
    "wan2.1_t2v_1.3B": np.array([1.0]*2+[1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762, 0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575, 0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549, 0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514, 0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416, 0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304, 0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135, 0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849, 0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359, 0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411, 0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089, 0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868, 0.81838, 0.81939]),
    "wan2.1_t2v_14B": np.array([1.0]*2+[1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189]),
    "wan2.1_i2v_480p_14B": np.array([1.0]*2+[0.98783, 0.98993, 0.97559, 0.97593, 0.98311, 0.98319, 0.98202, 0.98225, 0.9888, 0.98878, 0.98762, 0.98759, 0.98957, 0.98971, 0.99052, 0.99043, 0.99383, 0.99384, 0.98857, 0.9886, 0.99065, 0.99068, 0.98845, 0.98847, 0.99057, 0.99057, 0.98957, 0.98961, 0.98601, 0.9861, 0.98823, 0.98823, 0.98756, 0.98759, 0.98808, 0.98814, 0.98721, 0.98724, 0.98571, 0.98572, 0.98543, 0.98544, 0.98157, 0.98165, 0.98411, 0.98413, 0.97952, 0.97953, 0.98149, 0.9815, 0.9774, 0.97742, 0.97825, 0.97826, 0.97355, 0.97361, 0.97085, 0.97087, 0.97056, 0.97055, 0.96588, 0.96587, 0.96113, 0.96124, 0.9567, 0.95681, 0.94961, 0.94969, 0.93973, 0.93988, 0.93217, 0.93224, 0.91878, 0.91896, 0.90955, 0.90954, 0.92617, 0.92616]),
    "wan2.1_i2v_720p_14B": np.array([1.0]*2+[0.99428, 0.99498, 0.98588, 0.98621, 0.98273, 0.98281, 0.99018, 0.99023, 0.98911, 0.98917, 0.98646, 0.98652, 0.99454, 0.99456, 0.9891, 0.98909, 0.99124, 0.99127, 0.99102, 0.99103, 0.99215, 0.99212, 0.99515, 0.99515, 0.99576, 0.99572, 0.99068, 0.99072, 0.99097, 0.99097, 0.99166, 0.99169, 0.99041, 0.99042, 0.99201, 0.99198, 0.99101, 0.99101, 0.98599, 0.98603, 0.98845, 0.98844, 0.98848, 0.98851, 0.98862, 0.98857, 0.98718, 0.98719, 0.98497, 0.98497, 0.98264, 0.98263, 0.98389, 0.98393, 0.97938, 0.9794, 0.97535, 0.97536, 0.97498, 0.97499, 0.973, 0.97301, 0.96827, 0.96828, 0.96261, 0.96263, 0.95335, 0.9534, 0.94649, 0.94655, 0.93397, 0.93414, 0.91636, 0.9165, 0.89088, 0.89109, 0.8679, 0.86768]),
    "wan2.1_vace_1.3B": np.array([1.0]*2+[1.00129, 1.0019, 1.00056, 1.00053, 0.99776, 0.99746, 0.99726, 0.99789, 0.99725, 0.99785, 0.9958, 0.99625, 0.99703, 0.99728, 0.99863, 0.9988, 0.99735, 0.99731, 0.99714, 0.99707, 0.99697, 0.99687, 0.9969, 0.99683, 0.99695, 0.99702, 0.99697, 0.99701, 0.99608, 0.99617, 0.99721, 0.9973, 0.99649, 0.99657, 0.99659, 0.99667, 0.99727, 0.99731, 0.99603, 0.99612, 0.99652, 0.99659, 0.99635, 0.9964, 0.9958, 0.99585, 0.99581, 0.99585, 0.99573, 0.99579, 0.99531, 0.99534, 0.99505, 0.99508, 0.99481, 0.99484, 0.99426, 0.99433, 0.99403, 0.99406, 0.99357, 0.9936, 0.99302, 0.99305, 0.99243, 0.99247, 0.9916, 0.99164, 0.99085, 0.99087, 0.98985, 0.9899, 0.98857, 0.98859, 0.98717, 0.98721, 0.98551, 0.98556, 0.98301, 0.98305, 0.9805, 0.98055, 0.97635, 0.97641, 0.97183, 0.97187, 0.96496, 0.965, 0.95526, 0.95533, 0.94102, 0.94104, 0.91809, 0.91815, 0.87871, 0.87879, 0.80141, 0.80164]),
    "wan2.1_vace_14B": np.array([1.0]*2+[1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189]),
}


def magcache_flux_forward(
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
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        mag_ratios = transformer_options.get("mag_ratios")
        enable_magcache = transformer_options.get("enable_magcache", False)
        cur_step = transformer_options.get("current_step")
        
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

        # enable magcache
        # enable magcache
        if not hasattr(self, 'accumulated_err'):
            self.accumulated_err = 0
            self.accumulated_ratio = 1
            self.accumulated_steps = 0
        skip_forward = False
        if enable_magcache and cur_step not in [11]:
            cur_mag_ratio = mag_ratios[cur_step]
            self.accumulated_ratio = self.accumulated_ratio*cur_mag_ratio # magnitude ratio between current step and the cached step
            self.accumulated_steps += 1 # skip steps plus 1
            cur_skip_err = np.abs(1-self.accumulated_ratio) # skip error of current steps
            self.accumulated_err += cur_skip_err # accumulated error of multiple steps
            if self.accumulated_err<magcache_thresh and self.accumulated_steps<=magcache_K:
                skip_forward = True
                residual_x = self.residual_cache
            else:
                self.accumulated_err = 0
                self.accumulated_steps = 0
                self.accumulated_ratio = 1.0


        if skip_forward:
            img += self.residual_cache.to(img.device)
        else:
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
            self.residual_cache = (img - ori_img).to(mm.unet_offload_device())

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        
        return img

def magcache_hunyuanvideo_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        guiding_frame_index=None,
        ref_latent=None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        mag_ratios = transformer_options.get("mag_ratios")
        enable_magcache = transformer_options.get("enable_magcache", False)
        cur_step = transformer_options.get("current_step")
        
        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        if ref_latent is not None:
            ref_latent_ids = self.img_ids(ref_latent)
            ref_latent = self.img_in(ref_latent)
            img = torch.cat([ref_latent, img], dim=-2)
            ref_latent_ids[..., 0] = -1
            ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
            img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
            vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        pe = self.pe_embedder(ids)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        blocks_replace = patches_replace.get("dit", {})

        # enable magcache
        if not hasattr(self, 'accumulated_err'):
            self.accumulated_err = 0
            self.accumulated_ratio = 1
            self.accumulated_steps = 0
        skip_forward = False
        if enable_magcache:
            cur_mag_ratio = mag_ratios[cur_step]
            self.accumulated_ratio = self.accumulated_ratio*cur_mag_ratio # magnitude ratio between current step and the cached step
            self.accumulated_steps += 1 # skip steps plus 1
            cur_skip_err = np.abs(1-self.accumulated_ratio) # skip error of current steps
            self.accumulated_err += cur_skip_err # accumulated error of multiple steps
            if self.accumulated_err<magcache_thresh and self.accumulated_steps<=magcache_K:
                skip_forward = True
                residual_x = self.residual_cache
            else:
                self.accumulated_err = 0
                self.accumulated_steps = 0
                self.accumulated_ratio = 1.0

        if skip_forward:
            img += self.residual_cache.to(img.device)
        else:
            ori_img = img.clone()
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims_img=args["modulation_dims_img"], modulation_dims_txt=args["modulation_dims_txt"])
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims_img': modulation_dims, 'modulation_dims_txt': modulation_dims_txt}, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims, modulation_dims_txt=modulation_dims_txt)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            img = torch.cat((img, txt), 1)

            for i, block in enumerate(self.single_blocks):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims=args["modulation_dims"])
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims': modulation_dims}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, : img_len] += add

            img = img[:, : img_len]
            self.residual_cache = (img - ori_img).to(mm.unet_offload_device())

        if ref_latent is not None:
            img = img[:, ref_latent.shape[1]:]
        
        img = self.final_layer(img, vec, modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        return img

def magcache_wanmodel_forward(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        patches_replace = transformer_options.get("patches_replace", {})
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        # ret_ratio = transformer_options.get("magcache_ret_ratio")
        enable_magcache = transformer_options.get("enable_magcache", True)
        cur_step = transformer_options.get("current_step")
        mag_ratios = transformer_options.get("mag_ratios")
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        blocks_replace = patches_replace.get("dit", {})

        # [FIX] Check for latent dimension mismatch and reset cache if necessary
        if hasattr(self, 'magcache_state') and self.magcache_state[0]['residual_cache'] is not None:
            # x is flattened, so we check the number of tokens (patches) in dim 1.
            cached_tokens = self.magcache_state[0]['residual_cache'].shape[1]
            current_tokens = x.shape[1]
            if cached_tokens != current_tokens:
                # The input latent shape has changed. Invalidate the entire cache.
                delattr(self, 'magcache_state')

        # enable magcache
        if not hasattr(self, 'magcache_state'):
            self.magcache_state = {
                0: {'skip_forward': False, 'accumulated_ratio': 1, 'accumulated_err': 0, 'accumulated_steps': 0, 'residual_cache': None}, # condition
                1: {'skip_forward': False, 'accumulated_ratio': 1, 'accumulated_err': 0, 'accumulated_steps': 0, 'residual_cache': None} # uncondition
            }

        def update_cache_state(cache, cur_step):
            if enable_magcache:
                cur_scale = mag_ratios[cur_step]
                cache['accumulated_ratio'] = cache['accumulated_ratio']*cur_scale
                cache['accumulated_steps'] = cache['accumulated_steps'] + 1
                cache['accumulated_err'] += np.abs(1-cache['accumulated_ratio'])
                if cache['accumulated_err']<=magcache_thresh and cache['accumulated_steps']<=magcache_K:
                    cache['skip_forward'] = True
                else:
                    cache['skip_forward'] = False
                    cache['accumulated_ratio'] = 1.0
                    cache['accumulated_steps'] = 0
                    cache['accumulated_err'] = 0
            
        b = int(len(x) / len(cond_or_uncond))

        for i, k in enumerate(cond_or_uncond):
            update_cache_state(self.magcache_state[k], cur_step*2+i)

        # [FIX] Corrected skip_forward logic
        skip_forward = enable_magcache
        if skip_forward:
            for k in cond_or_uncond:
                # To skip the unified forward pass, all components (cond/uncond) must be ready.
                # A component is ready if its skip flag is set AND its cache has been previously populated.
                if not (self.magcache_state[k]['skip_forward'] and self.magcache_state[k]['residual_cache'] is not None):
                    skip_forward = False
                    break

        if skip_forward:
            for i, k in enumerate(cond_or_uncond):
                x[i*b:(i+1)*b] += self.magcache_state[k]['residual_cache'].to(x.device)
        else:
            ori_x = x.clone()
            for i, block in enumerate(self.blocks): # note: perform conditional and uncondition forward together, which can be improved by seperate into two single process.
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap, "transformer_options": transformer_options})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
            for i, k in enumerate(cond_or_uncond):
                self.magcache_state[k]['residual_cache'] = (x - ori_x)[i*b:(i+1)*b].to(mm.unet_offload_device())

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

def magcache_wan_vace_forward(
        self,
        x,
        t,
        context,
        vace_context,
        vace_strength,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        enable_magcache = transformer_options.get("enable_magcache", True)
        cur_step = transformer_options.get("current_step")
        mag_ratios = transformer_options.get("mag_ratios")
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        orig_shape = list(vace_context.shape)
        vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
        c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
        c = c.flatten(2).transpose(1, 2)
        c = list(c.split(orig_shape[0], dim=0))

        # arguments
        x_orig = x

        # [FIX] Check for latent dimension mismatch and reset cache if necessary
        if hasattr(self, 'magcache_state') and self.magcache_state[0]['residual_cache'] is not None:
            # x is flattened, so we check the number of tokens (patches) in dim 1.
            cached_tokens = self.magcache_state[0]['residual_cache'].shape[1]
            current_tokens = x.shape[1]
            if cached_tokens != current_tokens:
                # The input latent shape has changed. Invalidate the entire cache.
                delattr(self, 'magcache_state')

        # enable magcache
        if not hasattr(self, 'magcache_state'):
            self.magcache_state = {
                0: {'skip_forward': False, 'accumulated_ratio': 1, 'accumulated_err': 0, 'accumulated_steps': 0, 'residual_cache': None}, # condition
                1: {'skip_forward': False, 'accumulated_ratio': 1, 'accumulated_err': 0, 'accumulated_steps': 0, 'residual_cache': None} # uncondition
            }
        b = int(len(x) / len(cond_or_uncond))
        def update_cache_state(cache, cur_step):
            if enable_magcache:
                cur_scale = mag_ratios[cur_step]
                cache['accumulated_ratio'] = cache['accumulated_ratio']*cur_scale
                cache['accumulated_steps'] = cache['accumulated_steps'] + 1
                cache['accumulated_err'] += np.abs(1-cache['accumulated_ratio'])
                if cache['accumulated_err']<=magcache_thresh and cache['accumulated_steps']<=magcache_K:
                    cache['skip_forward'] = True
                else:
                    cache['skip_forward'] = False
                    cache['accumulated_ratio'] = 1.0
                    cache['accumulated_steps'] = 0
                    cache['accumulated_err'] = 0
        
        for i, k in enumerate(cond_or_uncond):
            update_cache_state(self.magcache_state[k], cur_step*2+i)
        
        # [FIX] Corrected skip_forward logic
        skip_forward = enable_magcache
        if skip_forward:
            for k in cond_or_uncond:
                # To skip the unified forward pass, all components (cond/uncond) must be ready.
                # A component is ready if its skip flag is set AND its cache has been previously populated.
                if not (self.magcache_state[k]['skip_forward'] and self.magcache_state[k]['residual_cache'] is not None):
                    skip_forward = False
                    break

        if skip_forward:
            for i, k in enumerate(cond_or_uncond):
                x[i*b:(i+1)*b] += self.magcache_state[k]['residual_cache'].to(x.device)
        else:
            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

                ii = self.vace_layers_mapping.get(i, None)
                if ii is not None:
                    for iii in range(len(c)):
                        c_skip, c[iii] = self.vace_blocks[ii](c[iii], x=x_orig, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
                        x += c_skip * vace_strength[iii]
                    del c_skip
            for i, k in enumerate(cond_or_uncond):
                self.magcache_state[k]['residual_cache'] = (x - x_orig)[i*b:(i+1)*b].to(mm.unet_offload_device())
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

def magcache_chroma_forward(
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
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        mag_ratios = transformer_options.get("mag_ratios")
        enable_magcache = transformer_options.get("enable_magcache", False)
        total_infer_steps = transformer_options.get("total_infer_steps")

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
        
        # MagCache initialization
        if not hasattr(self, 'accumulated_err'):
            # forward conditional and unconditional seperately.
            self.accumulated_err = [0.0, 0.0]
            self.accumulated_ratio = [1.0, 1.0]
            self.accumulated_steps = [0, 0]
            self.residual_cache = [None, None]
            self.cnt = 0            
        skip_forward = False
        if enable_magcache:  # Skip certain steps if needed
            cur_mag_ratio = mag_ratios[self.cnt]
            self.accumulated_ratio[self.cnt%2] = self.accumulated_ratio[self.cnt%2] * cur_mag_ratio
            self.accumulated_steps[self.cnt%2] += 1
            cur_skip_err = np.abs(1 - self.accumulated_ratio[self.cnt%2])
            self.accumulated_err[self.cnt%2] += cur_skip_err
            if self.accumulated_err[self.cnt%2] < magcache_thresh and self.accumulated_steps[self.cnt%2] <= magcache_K:
                skip_forward = True
            else:
                self.accumulated_err[self.cnt%2] = 0
                self.accumulated_steps[self.cnt%2] = 0
                self.accumulated_ratio[self.cnt%2] = 1.0
                
        if skip_forward:
            img += self.residual_cache[self.cnt%2].to(img.device)
        else:
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
            self.residual_cache[self.cnt%2] = (img - ori_img).to(mm.unet_offload_device())
        self.cnt += 1
        final_mod = self.get_modulations(mod_vectors, "final")
        img = self.final_layer(img, vec=final_mod)
        return img

class MagCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the MagCache will be applied to."}),
                "model_type": (["flux", "flux_kontext", "chroma", "hunyuan_video", "wan2.1_t2v_1.3B", "wan2.1_t2v_14B", "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B", "wan2.1_vace_1.3B", "wan2.1_vace_14B"], {"default": "wan2.1_t2v_1.3B", "tooltip": "Supported diffusion model."}),
                "magcache_thresh": ("FLOAT", {"default": 0.24, "min": 0.0, "max": 0.3, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                "retention_ratio": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 0.3, "step": 0.01, "tooltip": "The start percentage of the steps that will apply MagCache."}),
                "magcache_K": ("INT", {"default": 4, "min": 0, "max": 6, "step": 1, "tooltip": "The maxium skip steps of MagCache."}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "The maxium skip steps of MagCache."}),
                "end_step": ("INT", {"default": -1, "min": -100, "max": 100, "step": 1, "tooltip": "The maxium skip steps of MagCache."}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_magcache"
    CATEGORY = "MagCache"
    TITLE = "MagCache"
    
    def apply_magcache(self, model, model_type: str, magcache_thresh: float, retention_ratio: float, magcache_K: int, start_step: int, end_step:int):
        if magcache_thresh == 0:
            return (model,)

        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
        new_model.model_options["transformer_options"]["magcache_thresh"] = magcache_thresh
        new_model.model_options["transformer_options"]["retention_ratio"] = retention_ratio
        mag_ratios = SUPPORTED_MODELS_MAG_RATIOS[model_type]
        mag_ratios_tensor = torch.from_numpy(mag_ratios).float()
        new_model.model_options["transformer_options"]["mag_ratios"] = mag_ratios_tensor
        new_model.model_options["transformer_options"]["magcache_K"] = magcache_K
        new_model.model_options["transformer_options"]["start_step"] = start_step
        new_model.model_options["transformer_options"]["end_step"] = end_step
        diffusion_model = new_model.get_model_object("diffusion_model")

        if "flux" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_flux_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "chroma" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_chroma_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "hunyuan_video" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_hunyuanvideo_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "wan2.1_vace" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_wan_vace_forward.__get__(diffusion_model, diffusion_model.__class__)
            ) 
        elif "wan2.1" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_wanmodel_forward.__get__(diffusion_model, diffusion_model.__class__)
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
                    if (1 in cond_or_uncond) and hasattr(diffusion_model, 'magcache_state'):
                        delattr(diffusion_model, 'magcache_state')
                else:
                    if hasattr(diffusion_model, 'accumulated_err'):
                        delattr(diffusion_model, 'accumulated_err')
            
            total_infer_steps = len(sigmas)-1
            start_step = c["transformer_options"]["start_step"]
            end_step = c["transformer_options"]["end_step"]
            if end_step<0:
                end_step = total_infer_steps + end_step
            if  current_step_index>=int(total_infer_steps*c["transformer_options"]["retention_ratio"]) and (start_step<=current_step_index<=end_step): # start index of magcache
                c["transformer_options"]["enable_magcache"] = True
            else:
                c["transformer_options"]["enable_magcache"] = False
            calibration_len = len(c["transformer_options"]["mag_ratios"])//2 if "wan2.1" in model_type else len(c["transformer_options"]["mag_ratios"])
            c["transformer_options"]["current_step"] = current_step_index if (total_infer_steps)==calibration_len else int((current_step_index*((calibration_len-1)/(len(sigmas)-2)))) #interpolate when the steps is not equal to pre-defined steps
            if "chroma" in model_type:
                predefined_steps = len(c["transformer_options"]["mag_ratios"])//2
                assert total_infer_steps==predefined_steps, f"The inference steps of chroma must be {predefined_steps}."
            
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

class CompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the torch.compile will be applied to."}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "backend": (["inductor","cudagraphs", "eager", "aot_eager"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_compile"
    CATEGORY = "MagCache"
    TITLE = "Compile Model"
    
    def apply_compile(self, model, mode: str, backend: str, fullgraph: bool, dynamic: bool):
        patch_optimized_module()
        patch_same_meta()
        torch._dynamo.config.suppress_errors = True
        
        new_model = model.clone()
        new_model.add_object_patch(
                                "diffusion_model",
                                torch.compile(
                                    new_model.get_model_object("diffusion_model"),
                                    mode=mode,
                                    backend=backend,
                                    fullgraph=fullgraph,
                                    dynamic=dynamic
                                )
                            )
        
        return (new_model,)
    
    
NODE_CLASS_MAPPINGS = {
    "MagCache": MagCache,
    "CompileModel": CompileModel
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
