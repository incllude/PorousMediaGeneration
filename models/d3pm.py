from typing import Dict, List, Optional, Tuple, Union

import wandb
import torch
import numpy as np
from math import ceil
import torch.nn as nn
import lightning as l
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceFocalLoss
from hydra.utils import instantiate
from utils.utils import visualize_3d_slices


class D3PM(l.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        volume_shape: List[int],
        timesteps: int = 1000,
        transition_type: str = "uniform",
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        segmentation_loss = None,
        classification_loss = None,
        segmentation_weight: float = 1.,
        classification_weight: float = 1.,
        log_epochs: List[int] = -1
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        
        self.model = model
        self.volume_shape = volume_shape
        self.num_classes = volume_shape[0]
        self.real_volume =  None
        self.seg_loss = segmentation_loss
        self.seg_weight = segmentation_weight
        self.cls_loss = classification_loss
        self.cls_weight = classification_weight
        self.log_epochs = log_epochs
        
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        
        self._setup_diffusion_schedule(transition_type, beta_schedule, beta_start, beta_end)
        self.params_distribution = []
        
    def _setup_diffusion_schedule(
        self, transition_type:str, beta_schedule: str, beta_start: float, beta_end: float
    ):
        """Set up the noise schedule for the diffusion process."""
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
            self.register_buffer("betas", torch.clip(betas, 0.0001, 0.9999))
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        if transition_type == "uniform":
            q = [torch.eye(self.num_classes)]
            for beta in self.betas:
                mat = torch.ones(self.num_classes, self.num_classes) * beta / self.num_classes
                mat.diagonal().fill_(1 - (self.num_classes - 1) * beta / self.num_classes)
                q.append(mat)            
            
            q_bar = [torch.eye(self.num_classes)]
            for q_t in q:
                q_bar.append(q_bar[-1] @ q_t)
                
            self.register_buffer("q", torch.stack(q, dim=0))
            self.register_buffer("q_bar", torch.stack(q_bar, dim=0))
        elif transition_type == "gaussian":
            transition_bands = getattr(self, 'transition_bands', self.num_classes - 1)
    
            q = [torch.eye(self.num_classes)]
            
            for beta in self.betas:
                mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)
                
                values = torch.linspace(0., 255., self.num_classes, dtype=torch.float64)
                values = values * 2. / (self.num_classes - 1.)
                values = values[:transition_bands+1]
                values = -values * values / beta
                
                values_full = torch.cat([values[1:].flip(0), values])
                values_full = torch.softmax(values_full, dim=0)
                values = values_full[transition_bands:]
                
                for k in range(1, min(transition_bands + 1, self.num_classes)):
                    if self.num_classes - k > 0:
                        off_diag = torch.full((self.num_classes - k,), values[k])
                        mat += torch.diag(off_diag, diagonal=k)
                        mat += torch.diag(off_diag, diagonal=-k)
                
                diag = 1. - mat.sum(dim=1)
                mat += torch.diag(diag)
                
                q.append(mat.float())
            
            q_bar = [torch.eye(self.num_classes)]
            for q_t in q[1:]:
                q_bar.append(q_bar[-1] @ q_t)
                
            self.register_buffer("q", torch.stack(q, dim=0))
            self.register_buffer("q_bar", torch.stack(q_bar, dim=0))
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run the model forward."""
        return self.model(x, c, t)
        
    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None, eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_t | x_0) - the forward diffusion process."""
        probs = torch.einsum("bc...,bcd->bd...", x_start, self.q_bar[t])
        logits = torch.log(probs + eps)
        noise = torch.clip(noise, eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        # return F.softmax(logits + gumbel_noise, dim=1)
        return F.one_hot(torch.argmax(logits + gumbel_noise, dim=1), self.num_classes).permute(0, 4, 1, 2, 3).float()

    def q_posterior(self, x_0, x_t, t):

        step_1 = torch.einsum("bc...,bcd->bd...", x_t, self.q[t].transpose(1, 2))
        step_t = torch.einsum("bc...,bcd->bd...", x_0, self.q_bar[t-1])
        return F.softmax(torch.log(step_1) + torch.log(step_t), dim=1)
    
    def compute_loss(
        self, x_start: torch.Tensor, context: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the diffusion model loss."""
        if noise is None:
            noise = torch.rand_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise, eps=1e-10)
        pred_logits = self.forward(x_noisy, context, t)
        target = x_start

        target_q_posterior = self.q_posterior(x_start, x_noisy, t)
        # log_pred_q_posterior = torch.log(self.q_posterior(F.softmax(pred_logits, dim=1), x_noisy, t))
        pred_q_posterior = self.q_posterior(F.softmax(pred_logits, dim=1), x_noisy, t)
        # vb_loss = 100 * F.kl_div(log_pred_q_posterior, target_q_posterior, reduction="none").sum(dim=1).mean()
        vb_loss = F.smooth_l1_loss(pred_q_posterior, target_q_posterior, beta=0.25, reduction="none").sum(dim=1).mean()
        cls_loss = F.kl_div(F.log_softmax(pred_logits, dim=1), x_start, reduction="none").sum(dim=1).mean()

        loss = vb_loss + cls_loss
        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        batch, context = batch
        assert 0. <= batch.min() and batch.max() <= 1.
        self.params_distribution.extend(context.detach().cpu().tolist())
        self.real_volume = batch[0].cpu().clone()
        
        t = torch.randint(1, self.timesteps+1, (batch.shape[0],), device=self.device)
        
        loss = self.compute_loss(batch, context, t)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {"loss": loss}

    @torch.no_grad()
    def p_sample(
        self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, eps=1e-4
    ) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t) - the reverse diffusion process (one step)."""

        noise = torch.rand_like(x)
        pred_logits = self.forward(x, c, t)
        pred_q_posterior = self.q_posterior(F.softmax(pred_logits, dim=1), x, t)

        noise = torch.clip(noise, eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        not_first_step = (t != 1).reshape(-1, 1, 1, 1, 1).float()

        # sample = F.softmax(torch.log(pred_q_posterior) + gumbel_noise * not_first_step, dim=1)
        sample = F.one_hot(
            torch.argmax(torch.log(pred_q_posterior) + gumbel_noise * not_first_step, dim=1),
            self.num_classes
        ).permute(0, 4, 1, 2, 3).float()
        return sample
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        context: torch.Tensor = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """Sample a new image by running the reverse diffusion process."""
        if device is None:
            device = self.device
        if context is None:
            context = torch.tensor(np.random.choice(self.params_distribution, size=shape[0]))
            
        img = torch.clip(torch.rand(shape, device=device), 1e-10, 1.0)
        img = -torch.log(-torch.log(img))
        img = F.softmax(img, dim=1)
        img = F.one_hot(
            torch.argmax(img, dim=1),
            self.num_classes
        ).permute(0, 4, 1, 2, 3).float()
        context = context.to(device)
        
        for t in reversed(range(1, self.timesteps+1)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            img = self.p_sample(img, context, t_batch, eps=1e-10)
            
        return img

    def on_train_epoch_end(self):
        if self.log_epochs != -1 and self.current_epoch not in self.log_epochs:
            return
        batch_size = 2
        
        self.model.eval()
        context = torch.tensor([
            np.quantile(self.params_distribution, 0.25),
            np.quantile(self.params_distribution, 0.75),
        ]).float()
        imgs = self.sample((batch_size, *self.volume_shape), context=context, device="cuda")
        real = self.real_volume

        for i in range(batch_size):
            fig = visualize_3d_slices(imgs[i].cpu(), return_slices=False)
            self.logger.experiment.log({f"{i+1} generated": fig})
            fig.savefig(f"gen_epoch={self.current_epoch}_porosity={context[i].item()}_{i+1}.png")
        fig = visualize_3d_slices(real)
        self.logger.experiment.log({"real": fig})
        fig.savefig(f"real_epoch={self.current_epoch}.png")
        
        imgs = torch.argmax(imgs, dim=1)
        porosity = torch.tensor([(img == 2).float().mean() for img in imgs])
        # porosity = torch.tensor([img[2].mean() for img in imgs])
        diff = torch.abs(porosity - context).mean()
        self.log("porosity_diff", diff.item())
        self.model.train()

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
