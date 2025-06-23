import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict import TensorDict
from einops.layers.torch import Rearrange
from tensordict.nn import TensorDictModule

class ScalarEncoding(nn.Module):
    """
    Синусоидальное кодирование числового признака небольшого масштаба (0-0.1)
    Усиливает различия между близкими значениями
    """
    def __init__(self, dim, scale=1.):
        super().__init__()
        self.dim = dim
        self.scale = scale
    
    def forward(self, x):
        device = x.device
        
        scaled_x = x * self.scale
        
        scaled_x = scaled_x.view(-1, 1)
        
        freqs = torch.exp(
            -torch.arange(0, self.dim // 2, device=device) * (math.log(self.scale) / (self.dim // 2 - 1))
        )
        
        args = scaled_x * freqs[None, :]
        
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding

class SelfAttention3D(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out

class DiTBlock3D(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention3D(dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )
        
    def forward(self, x, t, c):
        condition = torch.cat([t, c], dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=1)
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)
        
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(x_norm)
        
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm)
        
        return x

class DiT3D(nn.Module):
    def __init__(
        self,
        depth=12,
        dim=768,
        heads=12,
        patch_size=8,
        in_channels=1,
        resolution=(64, 64, 64),
        use_context=False,
        contexts=["context"]
    ):
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        self.dim = dim
        
        d, h, w = resolution
        assert d % patch_size == 0 and h % patch_size == 0 and w % patch_size == 0, "Patch size must be divisible by resolution"
        self.num_patches = (d // patch_size) * (h // patch_size) * (w // patch_size)
        
        emb_size = 16
        self.init_conv = nn.Sequential(
            nn.Identity()
        )
        self.patchify = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', p1=2, p2=2, p3=2),
            nn.Conv3d(in_channels*2**3, 32, kernel_size=3, padding=1, padding_mode="zeros"),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            Rearrange('b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', p1=2, p2=2, p3=2),
            nn.Conv3d(32*2**3, 128, kernel_size=3, padding=1, padding_mode="zeros"),
            nn.BatchNorm3d(128),
            nn.SiLU(),
            Rearrange('b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', p1=2, p2=2, p3=2),
            nn.Conv3d(128*2**3, dim, kernel_size=1, padding=0, padding_mode="zeros"),
            nn.BatchNorm3d(dim),
            nn.SiLU(),
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', p1=1, p2=1, p3=1)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        divide = 1
        condition_embedding_dim = dim // 4
        if use_context:
            contexts = OmegaConf.to_object(contexts)
            divide = 1 + len(contexts)
            self.ctx_embedding = nn.Sequential(
                *[
                    TensorDictModule(
                        module=nn.Sequential(
                            ScalarEncoding(condition_embedding_dim, scale=100),
                            nn.Linear(condition_embedding_dim, condition_embedding_dim * 4),
                            nn.SiLU(),
                            nn.Linear(condition_embedding_dim * 4, dim//divide),
                        ),
                        in_keys=[ctx],
                        out_keys=[ctx]
                    )
                    for ctx in contexts
                 ],
                TensorDictModule(
                    module=lambda *x: torch.cat(x, dim=-1) if len(x) > 1 else x,
                    in_keys=contexts,
                    out_keys=["context"]
                )
            )
        self.time_embedding = nn.Sequential(
            ScalarEncoding(condition_embedding_dim, scale=100),
            nn.Linear(condition_embedding_dim, condition_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(condition_embedding_dim * 4, dim-dim//divide*(divide-1)),
        )
        
        self.blocks = nn.ModuleList([
            DiTBlock3D(dim, heads=heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        
        self.unpatchify = nn.Sequential(
            nn.Linear(dim, (patch_size//2)**3*32),
            Rearrange('b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)', 
                      p1=patch_size//2, p2=patch_size//2, p3=patch_size//2,
                      d=d//patch_size, h=h//patch_size, w=w//patch_size),
            nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1, padding_mode="zeros"),
            nn.GELU(),
            nn.Conv3d(32, 2**3*in_channels, 1),
            Rearrange('b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)', 
                      p1=2, p2=2, p3=2)
        )
        
    def forward(self, x, t, c=None):
        
        x = self.init_conv(x)
        x = self.patchify(x)  # [B, num_patches, dim]
        
        x = x + self.pos_embedding
        
        time_emb = self.time_embedding(t)
        if getattr(self, "ctx_embedding", False):
            ctx_emb = self.ctx_embedding(c)["context"]
            condition = torch.cat((time_emb, ctx_emb), dim=-1)
        else:
            condition = time_emb
        
        for block in self.blocks:
            x = block(x, time_emb, ctx_emb)
        
        x = self.norm(x)
        
        o = self.unpatchify(x)
        
        return o
