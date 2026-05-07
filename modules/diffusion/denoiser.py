import gc
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def cads_add_noise(y, t, tau1, tau2, s):
    if 1 - t <= tau1:
        gamma = 1.0
    elif 1 - t >= tau2:
        gamma = 0.0
    else:
        gamma = (tau2 - 1 + t) / (tau2 - tau1)

    mu, sigma = y.mean(dim=-1, keepdim=True), y.std(dim=-1, keepdim=True)
    y_norm = (y - mu) / sigma

    y_cads = np.sqrt(gamma) * y_norm + s * np.sqrt(1 - gamma) * torch.randn_like(y)
    y_cads = y_cads / np.sqrt(gamma + (1 - gamma) * s**2)
    return y_cads * sigma + mu


def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


class Denoiser(nn.Module):
    def __init__(
        self,
        scaling,
        guider=None,
        enable_seg=False,
        seg_guidance=3.0,
        seg_sigma=9999.0,
        seg_start=12,
        seg_end=25,
        seg_tmin=-1.0,
        
        e_att_start=12,
        e_att_end=17,
        e_att=False,
        legacy_seg=False,
        uncond_gen=False,
        image_cfg=-1.0,
        flowception_setup=False,
    ):
        super().__init__()

        self.scaling = scaling
        self.guider = guider
        self.enable_seg = enable_seg
        self.seg_guidance = seg_guidance
        self.seg_sigma = seg_sigma
        self.seg_start = seg_start
        self.seg_end = seg_end
        self.seg_tmin = seg_tmin
        self.class_guider = None
        self.class_w = 0.0
        self.class_amp = True
        self.class_pow = 0.5
        self.noise_null=False
        
        self.use_e_att = e_att
        self.e_att_start = e_att_start
        self.e_att_end = e_att_end
        
        self.enable_cads = False
        self.cads_tau1 = 0.6
        self.cads_tau2 = 0.9
        self.cads_s = 0.25

        self.legacy_seg = legacy_seg
        
        self.uncond_gen = uncond_gen
        
        self.autoguidance_model = None
        
        self.use_sag=False
        
        self.sag_perc = 0.8
        
        self.inpaint_mask = None
        self.inpaint_latent = None
        self.latest_velocity = None
        self.image_cfg=image_cfg
        self.do_alg=False
        
        self.alg_blur_sigma=2.0
        self.alg_kappa = 0.1
        
        self.vit_cfg=False
        self.flowception_setup = flowception_setup

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: dict,
        context_frames: torch.Tensor,
        return_means: bool = False,
        return_velocity:bool = False,
        frame_mask=None,
        
    ) -> torch.Tensor:
        t0 = sigma
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        if not self.flowception_setup:
            if not return_velocity:
                if not return_means:
                    return network(input * c_in, timestep=c_noise, context_frames=context_frames, **cond) * c_out + input * c_skip
                else:
                    model_out, repa_out, means, means_y = network(input * c_in, timestep=c_noise, context_frames=context_frames, return_means=True, **cond)
                    return model_out * c_out + input * c_skip, repa_out, means, means_y
            else:
                if not return_means:
                    return network(input * c_in, timestep=c_noise, context_frames=context_frames, **cond)
                else:
                    model_out, repa_out, means, means_y = network(input * c_in, timestep=c_noise, context_frames=context_frames, return_means=True, **cond)
                    return model_out, repa_out, means, means_y
        else:
            if not return_velocity:
                if not return_means:
                    model_out, lambda_ins = network(input, timestep=sigma, context_frames=context_frames, frame_mask=frame_mask, **cond)
                    return model_out * c_out + input * c_skip, lambda_ins
                else:
                    model_out, lambda_ins, repa_out, means, means_y = network(input, timestep=sigma, context_frames=context_frames, frame_mask=frame_mask, return_means=True, **cond)
                    return model_out * c_out + input * c_skip, lambda_ins, repa_out, means, means_y
            else:
                if not return_means:
                    model_out, lambda_ins = network(input, timestep=t0, context_frames=context_frames, frame_mask=frame_mask, **cond)
                    return model_out, lambda_ins
                else:
                    model_out, lambda_ins, repa_out, means, means_y = network(input, timestep=t0, context_frames=context_frames, frame_mask=frame_mask, return_means=True, **cond)
                    return model_out, lambda_ins, repa_out, means, means_y
            
            
    def forward_drift(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: list[dict],
        null_cond: list[dict],
        guidance: float,
        context_frames: torch.Tensor,
    ) -> torch.Tensor:
        # sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        # sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(append_dims(sigma, input.ndim))
        # c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        c_noise = append_dims(sigma, input.ndim)  # sigma = t
        c_noise = c_noise.reshape(sigma_shape)
        t = c_noise[:, None, None, None]
        
        velocity0, grads0 = torch.randn(1), torch.randn(1)
        input1 = input.detach().clone().requires_grad_()
        
        
        null_cond2 = null_cond
        cond2 = deepcopy(cond) if not self.uncond_gen else deepcopy(null_cond)
        if self.uncond_gen:
            cond2 = null_cond2
        else:
            cond2 = deepcopy(cond)
        
        if self.enable_cads:
            if isinstance(cond2['class_labels'], torch.Tensor):
                cond2['class_labels'] = cads_add_noise(cond2['class_labels'], t=t.mean().cpu().item(), tau1=self.cads_tau1, tau2=self.cads_tau2, s=self.cads_s)
            elif isinstance(cond2['class_labels'], list):
                cond2['class_labels'][0] = cads_add_noise(cond2['class_labels'][0], t=t.mean().cpu().item(), tau1=self.cads_tau1, tau2=self.cads_tau2, s=self.cads_s)
                cond2['class_labels'][1] = cads_add_noise(cond2['class_labels'][1], t=t.mean().cpu().item(), tau1=self.cads_tau1, tau2=self.cads_tau2, s=self.cads_s)
            
        if self.legacy_seg:
            network.blur_sigma = self.seg_sigma
            xc = network(
                input1,
                timestep=c_noise,
                # **cond
                **cond2,
                )  # .float()
            
            xu = network(
                input1,
                timestep=c_noise,
                # **cond
                **null_cond2,
                )  # .float()
            
            xp = network(
                    input1,
                    timestep=c_noise,
                    seg_start=self.seg_start,
                    seg_end=self.seg_end,
                    blur_sigma=self.seg_sigma,
                    # **cond, # coco evals were with this
                    **null_cond2,
                )  # .float()
            
            # velocity = xp + self.seg_guidance * (xc - xp)
            network.blur_sigma = -1.0
            velocity = (1 - guidance + self.seg_guidance) * xu + guidance * xc - self.seg_guidance * xp
           
        if self.do_alg:
            blur_sigma=self.alg_blur_sigma
            kernel_size = math.ceil(6 * blur_sigma) + 1 - math.ceil(6 * blur_sigma) % 2
            if t.mean().cpu().item() < self.alg_kappa:
                blurred_context = gaussian_blur_2d(context_frames[:, :, 0], kernel_size=kernel_size, sigma=blur_sigma)[:, :, None]
            else:
                blurred_context=context_frames
                
            xc = network(
                input1,
                timestep=c_noise,
                context_frames=blurred_context,
                **cond2
            )  
            n = network
            for i in range(len(n.blocks)):
                n.blocks[i].attn.use_e_att = False
            if self.use_e_att and c_noise.mean() > self.seg_tmin:
                for i in range(self.e_att_start, self.e_att_end):
                    n.blocks[i].attn.use_e_att = True
                    
            cu = cond2 if self.use_e_att else null_cond2
            xu = n(
                input1,
                timestep=c_noise,
                context_frames=context_frames,
                **null_cond2
            ) 
            xb = n(
                input1,
                timestep=c_noise,
                context_frames=blurred_context,
                **null_cond2
            ) 
            for i in range(len(n.blocks)):
                n.blocks[i].attn.use_e_att = False
            # velocity = xu + guidance * (xc - xu)
            velocity = xu + guidance * (xc - xb)
            
        elif self.vit_cfg:
                
            xc = network(
                input1,
                timestep=c_noise,
                context_frames=context_frames,
                **cond2
            )  
            n = network
            for i in range(len(n.blocks)):
                n.blocks[i].attn.use_e_att = False
            if self.use_e_att and c_noise.mean() > self.seg_tmin:
                for i in range(self.e_att_start, self.e_att_end):
                    n.blocks[i].attn.use_e_att = True
                    
            cu = cond2 if self.use_e_att else null_cond2
            xu = n(
                input1,
                timestep=c_noise,
                context_frames=context_frames,
                **null_cond2
            ) 
            
            novit_cond = deepcopy(cond2)
            novit_cond['class_labels'][1] = torch.zeros_like(novit_cond['class_labels'][1])
            xb = n(
                input1,
                timestep=c_noise,
                context_frames=torch.zeros_like(context_frames),
                **cond2
            ) 
            
            for i in range(len(n.blocks)):
                n.blocks[i].attn.use_e_att = False
            # velocity = xu + guidance * (xc - xu)
            velocity = xu + guidance * (xc - xb)
            
        else:
            if self.guider is None or t.mean().detach().cpu().item() > 0.98:
                if self.enable_seg and self.seg_sigma > 0 and c_noise.mean() > self.seg_tmin:
                    network.blur_sigma = self.seg_sigma
                    n = network

                    input2 = input1
                    if self.use_sag and c_noise.mean() > self.seg_tmin:
                        n.blocks[self.seg_start].attn.use_sag = True
                        cache_t = []
                        def forward_hook_sag(module, input, output):
                            attn = input[0].abs().sum(1, keepdim=True)
                            cache_t.append(attn)
                        handle = n.blocks[self.seg_start].attn.sag_dummy.register_forward_hook(forward_hook_sag)

                    if self.image_cfg < 0.0:
                        xc = n(
                            input1,
                            timestep=c_noise,
                            context_frames=context_frames,
                            # **cond
                            **cond2,
                            )  # .float()
                        
                        if self.use_sag and c_noise.mean() > self.seg_tmin:
                            blur_sigma=3
                            kernel_size = math.ceil(6 * blur_sigma) + 1 - math.ceil(6 * blur_sigma) % 2
                            
                            x0_hat = xc * (1-t) + input1
                            x0_hat = gaussian_blur_2d(x0_hat, kernel_size=9, sigma=1.0)
                            xt_smooth = x0_hat - xc*(1-t)
                            
                            attn_map = F.interpolate(cache_t[0], size=(input1.shape[-2], input1.shape[-1]))
                            phi=torch.quantile(attn_map, self.sag_perc)
                            
                            mask = attn_map > phi
                            input2 = input1*(~mask) + (mask) * xt_smooth
                            
                            handle.remove()
                            for i in range(len(n.blocks)):
                                n.blocks[i].attn.sag_dummy._forward_hooks = OrderedDict()
                                n.blocks[self.seg_start].attn.use_sag = False
                            cache_t = []
                            
                        if self.autoguidance_model is not None:
                            n = self.autoguidance_model
                            
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.use_e_att = False
                        if self.use_e_att and c_noise.mean() > self.seg_tmin:
                            for i in range(self.e_att_start, self.e_att_end):
                                n.blocks[i].attn.use_e_att = True
                        
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.sag_dummy._forward_hooks = OrderedDict()


                        xp = n(
                            input2,
                            timestep=c_noise,
                            seg_start=self.seg_start,
                            seg_end=self.seg_end,
                            blur_sigma=self.seg_sigma,
                            context_frames=torch.zeros_like(context_frames),
                            # **cond, # coco evals were with this
                            # **null_cond2,
                            **cond2,
                        )  # .float()
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.use_e_att = False
                            
                        
                        network.blur_sigma = -1.0
                        velocity = xp + self.seg_guidance * (xc - xp)
                    
                    else:
                        xc = n(
                            input1,
                            timestep=c_noise,
                            context_frames=context_frames,
                            # **cond
                            **cond2,
                            )  # .float()
                        
                        xi = n(
                            input1,
                            timestep=c_noise,
                            context_frames=context_frames,
                            # **cond
                            **null_cond2,
                        )  # .float()
                        
                        if self.use_sag and c_noise.mean() > self.seg_tmin:
                            blur_sigma=3
                            kernel_size = math.ceil(6 * blur_sigma) + 1 - math.ceil(6 * blur_sigma) % 2
                            
                            x0_hat = xc * (1-t) + input1
                            x0_hat = gaussian_blur_2d(x0_hat, kernel_size=9, sigma=1.0)
                            xt_smooth = x0_hat - xc*(1-t)
                            
                            attn_map = F.interpolate(cache_t[0], size=(input1.shape[-2], input1.shape[-1]))
                            phi=torch.quantile(attn_map, self.sag_perc)
                            
                            mask = attn_map > phi
                            input2 = input1*(~mask) + (mask) * xt_smooth
                            
                            handle.remove()
                            for i in range(len(n.blocks)):
                                n.blocks[i].attn.sag_dummy._forward_hooks = OrderedDict()
                                n.blocks[self.seg_start].attn.use_sag = False
                            cache_t = []
                            
                        if self.autoguidance_model is not None:
                            n = self.autoguidance_model
                            
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.use_e_att = False
                        if self.use_e_att and c_noise.mean() > self.seg_tmin:
                            for i in range(self.e_att_start, self.e_att_end):
                                n.blocks[i].attn.use_e_att = True
                        
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.sag_dummy._forward_hooks = OrderedDict()

                        xi_tau = n(
                            input2,
                            timestep=c_noise,
                            seg_start=self.seg_start,
                            seg_end=self.seg_end,
                            blur_sigma=self.seg_sigma,
                            context_frames=context_frames,
                            # **cond, # coco evals were with this
                            # **null_cond2,
                            **null_cond2,
                        )  # .float()
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.use_e_att = False

                        network.blur_sigma = -1.0
                        # velocity = xp + self.seg_guidance * (xc - xp)
                        if t.mean().cpu().item() < 0.0:
                            velocity = xc
                        else:
                            velocity = xi_tau + guidance * (xc-xi) + self.image_cfg * (xi - xi_tau)

                            
                    #         scaling_t = (1 - t**self.class_pow) / ns

                else:
                    network.blur_sigma = -1.0
                    
                    if self.image_cfg < 0.0:
                        xc = network(
                            input1,
                            timestep=c_noise,
                            context_frames=context_frames,
                            # **cond
                            **cond2
                        )  # .float()
                        n = network
                        if self.autoguidance_model is not None:
                            n = self.autoguidance_model
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.use_e_att = False
                        if self.use_e_att and c_noise.mean() > self.seg_tmin:
                            for i in range(self.e_att_start, self.e_att_end):
                                n.blocks[i].attn.use_e_att = True
                                # network.blocks[i].attn.step_size=lr
                                # network.blocks[i].attn.num_steps=steps
                        cu = cond2 if self.use_e_att else null_cond2
                        xu = n(
                            input1,
                            timestep=c_noise,
                            # context_frames=torch.zeros_like(context_frames),
                            context_frames=context_frames,
                            **null_cond2
                            # **cond2,
                            # **cu,
                        )  # .float()
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.use_e_att = False
                        velocity = xu + guidance * (xc - xu)
                    else:
                        xc = network(
                            input1,
                            timestep=c_noise,
                            context_frames=context_frames,
                            **cond2
                        )  
                        n = network
                        if self.autoguidance_model is not None:
                            n = self.autoguidance_model
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.use_e_att = False
                        if self.use_e_att and c_noise.mean() > self.seg_tmin:
                            for i in range(self.e_att_start, self.e_att_end):
                                n.blocks[i].attn.use_e_att = True
                                
                        cu = cond2 if self.use_e_att else null_cond2
                        xu = n(
                            input1,
                            timestep=c_noise,
                            context_frames=torch.zeros_like(context_frames),
                            **null_cond2
                            # **cond2
                        )
                        xi = n(
                            input1,
                            timestep=c_noise,
                            context_frames=context_frames,
                            **null_cond2
                        )  # .float()
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.use_e_att = False
                        if t.mean().cpu().item() < 0.0:
                            velocity = xc
                        else:

                            velocity = xu + guidance*(xc-xi) + self.image_cfg*(xi - xu)
                            
                    # here add classifier guidance
                    if self.class_guider is not None:
                        ns = 1
                        for _ in range(ns):
                            x0_hat = velocity * (1 - t) + input1
                           
                            grads = self.class_guider(
                                x0_hat, input1
                            )  # +(c_noise[:, None, None, None]-1)*velocity)
                            velocity0 = x0_hat
                            grads0 = grads
                            grads = grads.clip(-0.01, 0.01)

                            scaling_t = (1 - t**self.class_pow) / ns

                            scaling = (
                                velocity.norm(2, dim=(-1, -2, -3), keepdim=True)
                                / grads.norm(2, dim=(-1, -2, -3), keepdim=True)
                                * scaling_t
                            )
                            
                            velocity = velocity + self.class_w * scaling * grads
            else:
                if self.enable_seg and self.seg_sigma > 0 and c_noise.mean() > self.seg_tmin:
                    network.blur_sigma = self.seg_sigma

                    xc = network(
                        input,
                        timestep=c_noise,
                        context_frames=context_frames,
                        **cond2
                    ) 

                    # network.blur_sigma = -1.0
                    n = network
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.use_e_att = False
                    if self.use_e_att and c_noise.mean() > self.seg_tmin:
                        for i in range(self.e_att_start, self.e_att_end):
                            n.blocks[i].attn.use_e_att = True
                    
                    xu = n(
                        input1,
                        timestep=c_noise,
                        seg_start=self.seg_start,
                        seg_end=self.seg_end,
                        blur_sigma=self.seg_sigma,
                        context_frames=torch.zeros_like(context_frames),
                        # context_frames=context_frames,
                        # **cond, # coco evals were with this
                        **null_cond2,
                    )  # .float()
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.use_e_att = False
                        
                    network.blur_sigma = -1.0
                    
                    if t.ndim != xc.ndim:
                        t = t[:, None]
                    imc = input + (1 - t) * xc
                    imu = input + (1 - t) * xu
                    im_cfg = self.guider(imc, imu, self.seg_guidance)
                    velocity = (im_cfg - input) / (1 - t).clip(1e-3, None)

                    if self.class_guider is not None:
                        x0_hat = torch.clone(velocity).requires_grad_() * c_out + input * c_skip
                        grads = self.class_guider(x0_hat)  # +(c_noise[:, None, None, None]-1)*velocity)
                        scaling_t = (1 - t) ** self.lass_pow
                        scaling = (
                            velocity.norm(2, dim=(-1, -2, -3), keepdim=True)
                            / grads.norm(2, dim=(-1, -2, -3), keepdim=True)
                            * scaling_t
                        )
                        velocity0 = velocity
                        grads0 = grads
                        velocity = velocity + self.class_w * scaling * grads
                else:
                    xc = network(
                        input,
                        timestep=c_noise, 
                        context_frames=context_frames,
                        **cond2
                        )  # .float()
                    n = network
                    for i in range(len(n.blocks)):
                        
                        n.blocks[i].attn.use_e_att = False
                    if self.use_e_att and c_noise.mean() > self.seg_tmin:
                        for i in range(self.e_att_start, self.e_att_end):
                            n.blocks[i].attn.use_e_att = True
                            # network.blocks[i].attn.step_size=lr
                            # network.blocks[i].attn.num_steps=steps
                    cu = cond2 if self.use_e_att else null_cond2
                    xu = n(
                        input1,
                        timestep=c_noise,
                        context_frames=torch.zeros_like(context_frames),
                        **null_cond2
                        # **cond2,
                        # **cu,
                    )  # .float()
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.use_e_att = False
                   
                    if t.ndim != xc.ndim:
                        t = t[:, None]
                    imc = input + (1 - t) * xc
                    imu = input + (1 - t) * xu
                    im_cfg = self.guider(imc, imu, guidance)
                    velocity = (im_cfg - input) / (1 - t).clip(1e-3, None)
                    # here add classifier guidance
                    if self.class_guider is not None:
                        x0_hat = torch.clone(velocity).requires_grad_() * c_out + input * c_skip
                        grads = self.class_guider(x0_hat)  # +(c_noise[:, None, None, None]-1)*velocity)
                        scaling_t = (
                            1 - t
                        ) ** self.class_pow  # ((1 - c_noise) / c_noise.clip(1e-2, None))[:, None, None, None]

                        scaling = (
                            velocity.norm(2, dim=(-1, -2, -3), keepdim=True)
                            / grads.norm(2, dim=(-1, -2, -3), keepdim=True)
                            * scaling_t
                        )

                        velocity0 = velocity
                        grads0 = grads
                        velocity = velocity + self.class_w * scaling * grads
                        
        if self.inpaint_mask is not None and self.inpaint_latent is not None and c_noise.mean() < 1.0:
            x0 = input1 + (1-t) * velocity
            eps = input - t * velocity
            
            assert self.inpaint_latent.shape == self.inpaint_mask.shape
            mask = self.inpaint_mask
            latent_y = self.inpaint_latent
            
            velocity = (latent_y - eps ) * (~mask) + velocity * mask
            
        torch.cuda.empty_cache()
        gc.collect()
        
        return velocity.detach(), velocity0.detach(), grads0.detach()


    def forward_drift_inpaint(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: list[dict],
        null_cond: list[dict],
        guidance: float,
        inpaint_mask: torch.Tensor,
        inpaint_latent: torch.Tensor
    ) -> torch.Tensor:
        
        sigma_shape = sigma.shape
        c_skip, c_out, c_in, c_noise = self.scaling(append_dims(sigma, input.ndim))

        c_noise = append_dims(sigma, input.ndim)  # sigma = t
        c_noise = c_noise.reshape(sigma_shape)
        t = c_noise[:, None, None, None]
        
        velocity0, grads0 = torch.randn(1), torch.randn(1)
        input1 = input.detach().clone().requires_grad_()
        
        null_cond2 = null_cond
        cond2 = deepcopy(cond) if not self.uncond_gen else deepcopy(null_cond)
        if self.uncond_gen:
            cond2 = null_cond2
        else:
            cond2 = deepcopy(cond)
        
        if self.enable_cads:
            if isinstance(cond2['class_labels'], torch.Tensor):
                cond2['class_labels'] = cads_add_noise(cond2['class_labels'], t=t.mean().cpu().item(), tau1=self.cads_tau1, tau2=self.cads_tau2, s=self.cads_s)
            elif isinstance(cond2['class_labels'], list):
                cond2['class_labels'][0] = cads_add_noise(cond2['class_labels'][0], t=t.mean().cpu().item(), tau1=self.cads_tau1, tau2=self.cads_tau2, s=self.cads_s)
                cond2['class_labels'][1] = cads_add_noise(cond2['class_labels'][1], t=t.mean().cpu().item(), tau1=self.cads_tau1, tau2=self.cads_tau2, s=self.cads_s)
            
        if self.legacy_seg:
            network.blur_sigma = self.seg_sigma
            xc = network(
                input1,
                timestep=c_noise,
                # **cond
                **cond2,
                )  # .float()
            
            xu = network(
                input1,
                timestep=c_noise,
                # **cond
                **null_cond2,
                )  # .float()
            
            xp = network(
                    input1,
                    timestep=c_noise,
                    seg_start=self.seg_start,
                    seg_end=self.seg_end,
                    blur_sigma=self.seg_sigma,
                    # **cond, # coco evals were with this
                    **null_cond2,
                )  # .float()
            
            # velocity = xp + self.seg_guidance * (xc - xp)
            network.blur_sigma = -1.0
            velocity = (1 - guidance + self.seg_guidance) * xu + guidance * xc - self.seg_guidance * xp
            
        else:
            if self.guider is None or t.mean().detach().cpu().item() > 0.98:
                if self.enable_seg and self.seg_sigma > 0 and c_noise.mean() > self.seg_tmin:
                # if self.enable_seg and self.seg_sigma > 0 and np.random.rand()>0.5:
                    network.blur_sigma = self.seg_sigma
                    n = network

                    input2 = input1
                    if self.use_sag and c_noise.mean() > self.seg_tmin:
                        n.blocks[self.seg_start].attn.use_sag = True
                        cache_t = []
                        def forward_hook_sag(module, input, output):
                            attn = input[0].abs().sum(1, keepdim=True)
                            cache_t.append(attn)
                        handle = n.blocks[self.seg_start].attn.sag_dummy.register_forward_hook(forward_hook_sag)
                        
                    # with torch.no_grad():
                    xc = n(
                        input1,
                        timestep=c_noise,
                        # **cond
                        **cond2,
                        )  # .float()
                    
                    if self.use_sag and c_noise.mean() > self.seg_tmin:
                        blur_sigma=3
                        kernel_size = math.ceil(6 * blur_sigma) + 1 - math.ceil(6 * blur_sigma) % 2
                        
                        # smooth_inp = gaussian_blur_2d(input1, kernel_size=kernel_size, sigma=blur_sigma)
                        # blur x0_hat
                        
                        x0_hat = xc * (1-t) + input1
                        # x0_hat = gaussian_blur_2d(x0_hat, kernel_size=kernel_size, sigma=blur_sigma)
                        x0_hat = gaussian_blur_2d(x0_hat, kernel_size=9, sigma=1.0)
                        xt_smooth = x0_hat - xc*(1-t)
                        
                        attn_map = F.interpolate(cache_t[0], size=(input1.shape[-2], input1.shape[-1]))
                        phi=torch.quantile(attn_map, self.sag_perc)
                        
                        mask = attn_map > phi
                        input2 = input1*(~mask) + (mask) * xt_smooth
                        
                        handle.remove()
                        for i in range(len(n.blocks)):
                            n.blocks[i].attn.sag_dummy._forward_hooks = OrderedDict()
                            n.blocks[self.seg_start].attn.use_sag = False
                        cache_t = []
                        
                        
                    if self.autoguidance_model is not None:
                        n = self.autoguidance_model
                        
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.use_e_att = False
                    if self.use_e_att and c_noise.mean() > self.seg_tmin:
                        for i in range(self.e_att_start, self.e_att_end):
                            n.blocks[i].attn.use_e_att = True
                    
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.sag_dummy._forward_hooks = OrderedDict()
                        
                        
                    xp = n(
                        input2,
                        timestep=c_noise,
                        seg_start=self.seg_start,
                        seg_end=self.seg_end,
                        blur_sigma=self.seg_sigma,
                        # **cond, # coco evals were with this
                        **null_cond2,
                    )  # .float()
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.use_e_att = False

                    network.blur_sigma = -1.0
                    velocity = xp + self.seg_guidance * (xc - xp)

                else:
                    network.blur_sigma = -1.0
                    
                    # with torch.no_grad():
                    xc = network(
                        input1,
                        timestep=c_noise,
                        # **cond
                        **cond2
                    )  # .float()
                    n = network
                    if self.autoguidance_model is not None:
                        n = self.autoguidance_model
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.use_e_att = False
                    if self.use_e_att and c_noise.mean() > self.seg_tmin:
                        for i in range(self.e_att_start, self.e_att_end):
                            n.blocks[i].attn.use_e_att = True
                            # network.blocks[i].attn.step_size=lr
                            # network.blocks[i].attn.num_steps=steps
                    cu = cond2 if self.use_e_att else null_cond2
                    xu = n(
                        input1,
                        timestep=c_noise,
                        **null_cond2
                        # **cond2,
                        # **cu,
                    )  # .float()
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.use_e_att = False
                    # with torch.cuda.amp.autocast(enabled=self.class_amp):
                    velocity = xu + guidance * (xc - xu)
                    

                    # here add classifier guidance
                    if self.class_guider is not None:
                        ns = 1
                        for _ in range(ns):
                            x0_hat = velocity * (1 - t) + input1
                            # grads = self.class_guider(x0_hat).detach()
                            # grads = self.class_guider(t*input.requires_grad_()).detach()


                            grads = self.class_guider(
                                x0_hat, input1
                            )  # +(c_noise[:, None, None, None]-1)*velocity)
                            velocity0 = x0_hat
                            grads0 = grads
                            
                            grads = grads.clip(-0.01, 0.01)
                            scaling_t = (1 - t**self.class_pow) / ns
                            
                            scaling = (
                                velocity.norm(2, dim=(-1, -2, -3), keepdim=True)
                                / grads.norm(2, dim=(-1, -2, -3), keepdim=True)
                                * scaling_t
                            )
                            velocity = velocity + self.class_w * scaling * grads

            else:
                if self.enable_seg and self.seg_sigma > 0 and c_noise.mean() > self.seg_tmin:
                    network.blur_sigma = self.seg_sigma

                    xc = network(input, timestep=c_noise, **cond2) 
                    
                    n = network
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.use_e_att = False
                    if self.use_e_att and c_noise.mean() > self.seg_tmin:
                        for i in range(self.e_att_start, self.e_att_end):
                            n.blocks[i].attn.use_e_att = True
                    
                    xu = n(
                        input1,
                        timestep=c_noise,
                        seg_start=self.seg_start,
                        seg_end=self.seg_end,
                        blur_sigma=self.seg_sigma,
                        # **cond, # coco evals were with this
                        **null_cond2,
                    )  # .float()
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.use_e_att = False
                        
                    network.blur_sigma = -1.0
                    
                    imc = input + (1 - t) * xc
                    imu = input + (1 - t) * xu
                    im_cfg = self.guider(imc, imu, self.seg_guidance)
                    velocity = (im_cfg - input) / (1 - t).clip(1e-3, None)

                    if self.class_guider is not None:
                        x0_hat = torch.clone(velocity).requires_grad_() * c_out + input * c_skip
                        grads = self.class_guider(x0_hat)  # +(c_noise[:, None, None, None]-1)*velocity)
                        scaling_t = (1 - t) ** self.lass_pow
                        scaling = (
                            velocity.norm(2, dim=(-1, -2, -3), keepdim=True)
                            / grads.norm(2, dim=(-1, -2, -3), keepdim=True)
                            * scaling_t
                        )
                        velocity0 = velocity
                        grads0 = grads
                        velocity = velocity + self.class_w * scaling * grads
                else:
                    xc = network(input, timestep=c_noise, **cond2)  # .float()
                    
                    n = network
                    for i in range(len(n.blocks)):
                        
                        n.blocks[i].attn.use_e_att = False
                    if self.use_e_att and c_noise.mean() > self.seg_tmin:
                        for i in range(self.e_att_start, self.e_att_end):
                            n.blocks[i].attn.use_e_att = True
                            # network.blocks[i].attn.step_size=lr
                            # network.blocks[i].attn.num_steps=steps
                    cu = cond2 if self.use_e_att else null_cond2
                    xu = n(
                        input1,
                        timestep=c_noise,
                        **null_cond2
                        # **cond2,
                        # **cu,
                    )  # .float()
                    for i in range(len(n.blocks)):
                        n.blocks[i].attn.use_e_att = False
                        
                    # Convert velocity to image estimate.
                    imc = input + (1 - t) * xc
                    imu = input + (1 - t) * xu
                    im_cfg = self.guider(imc, imu, guidance)
                    velocity = (im_cfg - input) / (1 - t).clip(1e-3, None)
                    # here add classifier guidance
                    if self.class_guider is not None:
                        x0_hat = torch.clone(velocity).requires_grad_() * c_out + input * c_skip
                        grads = self.class_guider(x0_hat)  # +(c_noise[:, None, None, None]-1)*velocity)
                        scaling_t = (
                            1 - t
                        ) ** self.class_pow  # ((1 - c_noise) / c_noise.clip(1e-2, None))[:, None, None, None]

                        scaling = (
                            velocity.norm(2, dim=(-1, -2, -3), keepdim=True)
                            / grads.norm(2, dim=(-1, -2, -3), keepdim=True)
                            * scaling_t
                        )

                        velocity0 = velocity
                        grads0 = grads
                        velocity = velocity + self.class_w * scaling * grads
                        
        if inpaint_mask is not None and inpaint_latent is not None and c_noise.mean() < 1.0:
            x0 = input1 + (1-t) * velocity
            eps = input - t * velocity
                
            assert inpaint_latent.shape == inpaint_mask.shape
            mask = inpaint_mask
            latent_y = inpaint_latent
            
            velocity = (latent_y - eps ) * (~mask) + velocity * mask
            # self.latest_velocity = velocity
            
        torch.cuda.empty_cache()
        

        gc.collect()
        
        return velocity.detach(), velocity0.detach(), grads0.detach()

    def forward_score(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: list[dict],
        null_cond: list[dict],
        guidance: float,
    ) -> torch.Tensor:
        # sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape

        c_noise = append_dims(sigma, input.ndim)  # sigma = t
        c_noise = c_noise.reshape(sigma_shape)
        t = c_noise[:, None, None, None]

        if self.guider is None or t.mean().detach().cpu().item() > 0.98:
            xc = network(input, timestep=c_noise, **cond)
            xu = network(input, timestep=c_noise, **null_cond)
            velocity = xu + (guidance - 1.0) * (xc - xu)
        else:
            xc = network(input, timestep=c_noise, **cond)
            xu = network(input, timestep=c_noise, **null_cond)

            # Convert velocity to image estimate.
            imc = input + (1 - t) * xc
            imu = input + (1 - t) * xu
            im_cfg = self.guider(imc, imu, guidance)
            velocity = (im_cfg - input) / (1 - t).clip(1e-3, None)

        score = (t * velocity - input) / (1 - t).clip(1e-3, None)

        return score


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        scaling,
        num_idx,
        discretization,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
        enable_seg=False,
        seg_guidance=3.0,
        seg_sigma=9999.0,
        seg_start=12,
        seg_end=25,
        uncond_gen=False,
    ):
        super().__init__(scaling, enable_seg=enable_seg, seg_guidance=seg_guidance, seg_sigma=seg_sigma)
        self.discretization = discretization
        sigmas = self.discretization(num_idx, do_append_zero=do_append_zero, flip=flip)

        # self.sigmas=  sigmas

        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: torch.Tensor | int) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise


def build_denoiser_wrapper(
    name,
    scaling,
    num_idx,
    discretization,
    enable_seg=False,
    seg_guidance=3.0,
    seg_sigma=9999.0,
    seg_start=12,
    seg_end=25,
    seg_tmin=-1.0,
    e_att_start=12,
    e_att_end=17,
    e_att=False,
    legacy_seg=False,
    uncond_gen=False,
    image_cfg=-1.0,
    flowception_setup=False,
):
    assert name.lower() in ["denoiser", "discrete_denoiser"]
    if name == "denoiser":
        return Denoiser(
            scaling,
            enable_seg=enable_seg,
            seg_guidance=seg_guidance,
            seg_sigma=seg_sigma,
            seg_start=seg_start,
            seg_end=seg_end,
            seg_tmin=seg_tmin,
            e_att_start=e_att_start,
            e_att_end=e_att_end,
            e_att=e_att,
            legacy_seg=legacy_seg,
            uncond_gen=uncond_gen,
            image_cfg=image_cfg,
            flowception_setup=flowception_setup,
        )
    elif name == "discrete_denoiser":
        return DiscreteDenoiser(
            scaling,
            num_idx,
            discretization=discretization,
            enable_seg=enable_seg,
            seg_guidance=seg_guidance,
            seg_sigma=seg_sigma,
            seg_start=seg_start,
            seg_end=seg_end,
            seg_tmin=seg_tmin,
            legacy_seg=legacy_seg,
            uncond_gen=uncond_gen
        )
