from typing import Dict, List, Optional, Tuple, Union

import wandb
import torch
import numpy as np
from math import ceil
import torch.nn as nn
import lightning as l
import torch.nn.functional as F
from hydra.utils import instantiate
from utils.utils import visualize_3d_slices


class SGPM(l.LightningModule):
    """Implementation of a Segmentation-Guided Diffusion Model for 3D tomogram segmentation."""

    def __init__(
        self,
        model: nn.Module,
        volume_shape: List[int],
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        loss_type: str = "l2",
        loss_alpha: float = 1.,
        sobel_alpha: float = 0.,
        scale: float = 1,
        step_size: int = 1,
        log_epochs: List[int] = -1
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.volume_shape = volume_shape
        self.real_volume =  None

        self.loss_alpha = loss_alpha
        self.sobel_alpha = sobel_alpha
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.scale = scale
        self.step_size = step_size
        self.log_epochs = log_epochs
        
        self._setup_diffusion_schedule(beta_schedule, beta_start, beta_end)
        self.segm_distribution = []

        kernel_z = torch.tensor([
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[-1, -2, -1],
             [-2, -4, -2],
             [-1, -2, -1]]
        ])
        kernel_x = kernel_z.permute(1, 2, 0)
        kernel_y = kernel_z.permute(1, 0, 2)
        sobel_kernels = torch.stack([kernel_x, kernel_y, kernel_z]).float()
        sobel_kernels = sobel_kernels.unsqueeze(1)
        self.sobel = nn.Conv3d(1, 3, kernel_size=3, padding=1, padding_mode="reflect", bias=False)
        self.sobel.weight.data = sobel_kernels
        self.sobel.weight.requires_grad_(False)
        
    def _setup_diffusion_schedule(
        self, beta_schedule: str, beta_start: float, beta_end: float
    ):
        """Set up the noise schedule for the diffusion process."""
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        
        if beta_schedule == "linear":
            self.register_buffer(
                "betas", 
                torch.linspace(beta_start, beta_end, self.timesteps+1)[1:]
            )
        elif beta_schedule == "cosine":
            x = torch.linspace(0, self.timesteps, self.timesteps+1)
            alphas_cumprod = torch.cos((x / self.timesteps) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.register_buffer("betas", torch.clip(betas, 0.0, 0.999))
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        alphas = 1. - self.betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        register_buffer('alphas_cumprod', alphas_cumprod)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        maybe_clipped_snr.clamp_(max=5)
        self.register_buffer('loss_weight', maybe_clipped_snr / snr)
    
    def forward(self, x: torch.Tensor, segm: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run the model forward."""
        # return self.model(torch.cat((x, segm), dim=1), t)
        return self.model(x, segm, t)
    
    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_t | x_0) - the forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1, 1)
        
        return (
            sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise,
            noise
        )
    
    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from the noise prediction."""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1, 1)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps
    
    def p_mean_variance(
        self, x: torch.Tensor, segm: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Get p(x_{t-1} | x_t) distribution parameters."""
        model_output = self.forward(x, segm, t)
        
        if self.prediction_type == "epsilon":
            pred_noise = model_output
            x_recon = self._predict_x0_from_eps(x, t, pred_noise)
            
            if clip_denoised:
                x_recon = torch.clamp(x_recon, -self.scale-2, self.scale+2)
        elif self.prediction_type == "x_start":
            x_recon = model_output
            
            if clip_denoised:
                x_recon = torch.clamp(x_recon, -self.scale-2, self.scale+2)
                
            pred_noise = (
                (x - self.sqrt_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1, 1) * x_recon)
                / self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1, 1)
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        posterior_mean = (
            self.posterior_mean_coef1.gather(-1, t).reshape(-1, 1, 1, 1, 1) * x_recon
            + self.posterior_mean_coef2.gather(-1, t).reshape(-1, 1, 1, 1, 1) * x
        )
        
        posterior_variance = self.posterior_variance.gather(-1, t).reshape(-1, 1, 1, 1, 1)
        
        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "pred_xstart": x_recon,
            "pred_noise": pred_noise,
        }
    
    @torch.no_grad()
    def p_sample(
        self, x: torch.Tensor, segm: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True
    ) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t) - the reverse diffusion process (one step)."""
        out = self.p_mean_variance(x, segm, t, clip_denoised)
        noise = torch.randn_like(x)
        
        nonzero_mask = (t != 0).reshape(-1, 1, 1, 1, 1).float()
        
        return out["mean"] + nonzero_mask * torch.sqrt(out["variance"]) * noise

    def to_new_colorspace(self, x):
        return torch.from_numpy(np.interp(x.detach().cpu().numpy(), self.x_space, self.y_space)).reshape(x.shape).to(x.dtype).to(x.device)
        
    def to_orig_colorspace(self, x):
        return torch.from_numpy(np.interp(x.detach().cpu().numpy(), self.y_space, self.x_space)).reshape(x.shape).to(x.dtype).to(x.device)
    
    @torch.no_grad()
    def sample(
        self,
        size: int = 1,
        segm: torch.Tensor = None,
        device: torch.device = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Sample a new image by running the reverse diffusion process."""
        if device is None:
            device = self.device
        if segm is None:
            segm = np.random.choice(len(self.segm_distribution), size=size)
            segm = [self.segm_distribution[x] for x in segm]
            
        segm = torch.stack(segm).to(device)
        img = torch.randn(segm.shape, device=device)[:, [0]]
        
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((size,), t, device=device, dtype=torch.long)
            img = self.p_sample(img, segm, t_batch, clip_denoised)

        # print()
        # print(img.min(), img.max())
        # img = img.clip(-self.scale, self.scale)
        
        img = img * 994.8743318327591 + 34.54711526953125
        img[segm[:, [0]].to(torch.bool)] += 10305
        img[segm[:, [1]].to(torch.bool)] += 8472
        img[segm[:, [2]].to(torch.bool)] += 7168
        img = (img - 1697) / (58091 - 1679)
        img = img * 2 - 1

        # img = self.to_orig_colorspace(img)
        # if img.min().item() > -2 and img.max().max().item() < 1:
        #     prefix = "segmentation_translator"
        #     dirpath = '/kaggle/working/checkpoints'
        #     ckpt_path = f"{dirpath}/{prefix}_epoch={self.current_epoch}.pt"
        #     torch.save(self.model.state_dict(), ckpt_path)
        return img
    
    def compute_loss(
        self, x_start: torch.Tensor, segm: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the diffusion model loss."""
        x_noisy, noise_target = self.q_sample(x_start, t, noise)
        pred = self.forward(x_noisy, segm, t)
        
        if self.prediction_type == "epsilon":
            target = noise_target
            x0 = self._predict_x0_from_eps(x_noisy, t, pred)
        elif self.prediction_type == "x_start":
            target = x_start
            x0 = pred
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        pred_scaled = (x0 - torch.stack([x.min()[None, None, None, None] for x in x0])) / (torch.stack([x.max()[None, None, None, None] for x in x0]) - torch.stack([x.min()[None, None, None, None] for x in x0]))
        target_scaled = (target - torch.stack([x.min()[None, None, None, None] for x in target])) / (torch.stack([x.max()[None, None, None, None] for x in target]) - torch.stack([x.min()[None, None, None, None] for x in target]))
        sobel_pred = self.sobel(pred_scaled)
        sobel_target = self.sobel(target_scaled)
        if self.loss_type == "l1":
            point_loss = self.loss_alpha * F.l1_loss(pred, target, reduction="none")\
                         + self.sobel_alpha * F.l1_loss(sobel_pred, sobel_target, reduction="none")
        elif self.loss_type == "l2":
            point_loss = self.loss_alpha * F.mse_loss(pred, target, reduction="none")\
                         + self.sobel_alpha * F.mse_loss(sobel_pred, sobel_target, reduction="none")
        elif self.loss_type == "huber":
            point_loss = self.loss_alpha * F.smooth_l1_loss(pred, target, reduction="none")\
                         + self.sobel_alpha * F.smooth_l1_loss(sobel_pred, sobel_target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        loss = 0.
        sigmas = 1 / (self.posterior_variance.gather(-1, t) * self.alphas.gather(-1, t))
        # sigmas = torch.ones_like(sigmas)
        for i in range(x_start.size(0)):
            loss += point_loss[i].mean() * sigmas[i]
        return loss

    def augment_segmentation(self, segmentations: torch.Tensor):
        mask_classes = torch.randint(0, 3, (segmentations.shape[0],))
        for i, mask_class in enumerate(mask_classes):
            if mask_class == 0:
                continue
            segmentations[i, 0] += segmentations[i, mask_class]
            segmentations[i, mask_class] = 0
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        tomo, segm, context = batch
        # tomo = self.to_new_colorspace(tomo)
        self.augment_segmentation(segm)
        self.real_volume = tomo[0].detach().cpu().clone()
        if len(self.segm_distribution) < 200:
            self.segm_distribution.extend(segm.detach().cpu().clone())

        t = torch.randint(1, self.timesteps, (tomo.shape[0],), device=self.device)
        # t = torch.randint(0, self.timesteps, (tomo.shape[0] // 2,), device=self.device)
        # t = torch.cat((t, self.timesteps-t-1))
        
        loss = self.compute_loss(tomo, segm, t)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {"loss": loss}

    def on_train_epoch_end(self):
        if self.log_epochs != -1 and self.current_epoch not in self.log_epochs:
            return
        batch_size = 2
        
        self.model.eval()
        imgs = self.sample(size=batch_size, device="cuda", clip_denoised=False)
        real = self.real_volume
        # real = self.to_orig_colorspace(self.real_volume)

        for i in range(batch_size):
            fig = visualize_3d_slices(imgs[i].cpu(), return_slices=False)
            self.logger.experiment.log({f"{i+1} generated": fig})
            fig.savefig(f"gen_epoch={self.current_epoch}_{i+1}.png")
        fig = visualize_3d_slices(real)
        self.logger.experiment.log({"real": visualize_3d_slices(real, return_slices=False)})
        fig.savefig(f"real_epoch={self.current_epoch}.png")

    def set_training_settings(self, optimizer_cfg, scheduler_cfg):
        self.opt = lambda x: instantiate(optimizer_cfg, params=x)
        self.sch = lambda x: instantiate(scheduler_cfg, optimizer=x)
    
    def configure_optimizers(self):
        """
        Configure optimizers for each scale
        """
        optimizer = self.opt(self.model.parameters())
        scheduler = self.sch(optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
