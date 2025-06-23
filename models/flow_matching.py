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
from torchdiffeq import odeint


class ProbabilisticFlowMatching(l.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        volume_shape: List[int],
        log_epochs: List[int] = -1,
        ode_method: str = "euler",
        num_steps: int = 100,
        characteristics: Dict = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        
        self.model = model
        self.volume_shape = volume_shape
        self.num_classes = volume_shape[0]
        self.real_volume = None
        self.log_epochs = log_epochs
        self.characteristics = characteristics
        
        self.ode_method = ode_method
        self.num_steps = num_steps
        
        self.chars_dist = None
        self.chars_dist_size = 0
        
    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run the model forward - predicts velocity field."""
        v = self.model(x=x, c=c, t=t)
        res = F.softmax(2*v, dim=1) - F.softmax(-2*v, dim=1)
        return res
    
    def get_probability_path(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mean and std of probability path at time t.
        We use a simple linear interpolation path.
        """
        alpha_t = t.view(-1, 1, 1, 1, 1)
        return alpha_t
    
    def sample_from_prior(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from prior distribution (random simplex point)."""
        # prior = torch.randn(shape, device=device)
        
        noise = torch.rand(shape, device=device)
        gumbel_noise = -torch.log(-torch.log(noise))
        prior = F.softmax(gumbel_noise, dim=1)
        # prior = torch.rand(shape, device=device)
        # prior = F.one_hot(
        #     torch.argmax(prior, dim=1), 
        #     self.num_classes
        # ).permute(0, 4, 1, 2, 3).float()
        return prior
    
    def forward_process(
        self, 
        x_1: torch.Tensor, 
        t: torch.Tensor, 
        x_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: interpolate between x_0 (noise) and x_1 (data).
        Returns x_t and the velocity field.
        """
        if x_0 is None:
            x_0 = self.sample_from_prior(x_1.shape, x_1.device)
        
        alpha_t = self.get_probability_path(t)
        
        x_t = alpha_t * x_1 + (1 - alpha_t) * x_0
        
        # noise = torch.rand_like(x_t) * (1 - alpha_t) * 0.99 + alpha_t
        # noise = -torch.log(-torch.log(noise))
        # x_t = F.softmax(torch.log(x_t) + noise, dim=1)
        
        velocity = x_1 - x_0
        
        return x_0, x_t, velocity
    
    def compute_loss(
        self, 
        x_1: torch.Tensor, 
        context: torch.Tensor, 
        x_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Flow Matching loss."""
        batch_size = x_1.shape[0]
        
        t = torch.rand(batch_size, device=x_1.device)
        
        x_0, x_t, true_velocity = self.forward_process(x_1, t, x_0)
        
        pred_velocity = self.forward(x_t, context, t)
        
        loss = F.mse_loss(pred_velocity, true_velocity)
        
        return loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        batch, context = batch
        assert 0. <= batch.min() and batch.max() <= 1.
        if self.chars_dist is not None:
            self.chars_dist = torch.cat((self.chars_dist, context.clone()), dim=0)
        else:
            self.chars_dist = context.clone()
        self.chars_dist_size += batch.size(0)
        self.real_volume = batch[0].cpu().clone()

        loss = self.compute_loss(batch, context)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {"loss": loss}
    
    def velocity_function(self, t: float, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Velocity function for ODE integration."""
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)
        return self.forward(x, context, t_tensor)
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        context: torch.Tensor = None,
        device: torch.device = None,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample by solving ODE from t=0 to t=1."""
        if device is None:
            device = self.device
        # if context is None:
        #     context = torch.tensor(
        #         np.random.choice(self.params_distribution, size=shape[0]),
        #         device=device
        #     )
        
        x_0 = self.sample_from_prior(shape, device)
        context = context.to(device)
        
        def ode_func(t, x):
            t_scalar = t.item()
            return self.velocity_function(t_scalar, x, context.clone())
        
        t_span = torch.linspace(0, 1, self.num_steps + 1, device=device)
        
        if self.ode_method == "euler":
            x = x_0
            trajectory = [x] if return_trajectory else None
            dt = 1.0 / self.num_steps
            
            for i in range(self.num_steps):
                t_curr = i * dt
                velocity = self.velocity_function(t_curr, x, context.clone())
                x = x + dt * velocity
                x = torch.clip(x, 0., 1.)
                x = x / x.sum(dim=1, keepdims=True)
                if return_trajectory:
                    trajectory.append(x)
        else:
            solution = odeint(ode_func, x_0, t_span, method=self.ode_method)
            x = solution[-1]
            trajectory = solution if return_trajectory else None

        x = x.detach()
        if return_trajectory:
            return x, trajectory
        return x
    
    def on_train_epoch_end(self):
        if self.log_epochs != -1 and self.current_epoch not in self.log_epochs:
            return
        batch_size = 2
        
        self.model.eval()
        q25 = int(0.25 * (self.chars_dist_size - 1)) + 1
        q75 = int(0.75 * (self.chars_dist_size - 1)) + 1
        q25_k = torch.kthvalue(self.chars_dist["porosity"], q25).indices
        q75_k = torch.kthvalue(self.chars_dist["porosity"], q75).indices
        context = self.chars_dist[[q25_k, q75_k]].clone()
        
        imgs = self.sample((batch_size, *self.volume_shape), context=context, device="cuda").cpu()
        real = self.real_volume

        for i in range(batch_size):
            fig = visualize_3d_slices(imgs[i], return_slices=False)
            self.logger.experiment.log({f"{i+1} generated": fig})
            fig.savefig(f"gen_epoch={self.current_epoch}_{i+1}.png")
        
        fig = visualize_3d_slices(real)
        self.logger.experiment.log({"real": fig})
        fig.savefig(f"real_epoch={self.current_epoch}.png")

        context = context.cpu()
        characteristics = self.characteristics(imgs)
        diff = ((characteristics - context) / context).abs().mean()
        for c in diff.keys():
            self.log(f"{c}_mape", diff[c].item())
        self.model.train()

    def set_training_settings(self, optimizer_cfg, scheduler_cfg):
        self.opt = lambda x: instantiate(optimizer_cfg, params=x)
        self.sch = lambda x: instantiate(scheduler_cfg, optimizer=x)
    
    def configure_optimizers(self):
        """Configure optimizers."""
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
