from typing import List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
import torch.nn.functional as F
from tensordict import TensorDict
from einops.layers.torch import Rearrange
from tensordict.nn import TensorDictModule


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight, gain=0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


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
            -torch.arange(0, self.dim // 2, device=device) * (math.log(100.0) / (self.dim // 2 - 1))
        )
        
        args = scaled_x * freqs[None, :]
        
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding


class IBN(nn.Module):

    def __init__(self, planes, ratio):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm3d(self.half, affine=True)
        self.BN = nn.BatchNorm3d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class VolumeContextAttention(nn.Module):
    """
    Attention модуль с:
    - Отдельными токенами для каждого context
    - FiLM модуляцией времени на токены
    - Позиционным кодированием для токенов объема
    - TransformerEncoder вместо простого MultiheadAttention
    """
    
    def __init__(self, dim, context_dim, time_dim, heads=8, num_contexts=2, num_layers=2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.num_contexts = num_contexts
        
        self.context_projections = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(context_dim // num_contexts, dim)
            ) for _ in range(num_contexts)
        ])
        
        self.time_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2)
        )
        self.time_film[1].weight.data.zero_()
        self.time_film[1].bias.data.zero_()
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 16**3, dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.norm_input = nn.LayerNorm(dim)
        
    def forward(self, x, time_emb, context_emb):
        """
        x: [B, C, D, H, W] - пространственные признаки
        time_emb: [B, time_dim] - временное встраивание  
        context_emb: [B, context_dim] - контекстное встраивание (concat всех context)
        """
        batch_size, channels, d, h, w = x.shape

        context_tokens = torch.stack([context_emb[ctx] for ctx in context_emb.keys() if ctx != "context"], dim=1)
        
        volume_tokens = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, DHW, C]
        
        if context_tokens is not None:
            all_tokens = torch.cat([context_tokens, volume_tokens], dim=1)  # [B, num_contexts+DHW, C]
        else:
            all_tokens = volume_tokens

        seq_len = all_tokens.size(1)
        pos_emb = self.pos_encoding.repeat(batch_size, 1, 1)[:, :seq_len, :]  # [1, seq_len, dim]
        
        all_tokens = all_tokens + pos_emb
        
        if time_emb is not None:
            film_params = self.time_film(time_emb)  # [B, dim * 2]
            scale, shift = film_params.chunk(2, dim=-1)  # [B, dim] каждый
            scale = scale.unsqueeze(1) + 1.0  # [B, 1, dim] + bias для стабильности
            shift = shift.unsqueeze(1)        # [B, 1, dim]
            
            all_tokens = all_tokens * scale + shift
        
        all_tokens = self.norm_input(all_tokens)
        
        transformed_tokens = self.transformer(all_tokens)
        
        transformed_tokens = transformed_tokens + all_tokens
        
        num_context_tokens = context_tokens.size(1) if context_tokens is not None else 0
        output_tokens = transformed_tokens[:, num_context_tokens:, :]  # [B, DHW, dim]
        
        output = output_tokens.permute(0, 2, 1).view(batch_size, channels, d, h, w)
        
        return output


class ResBlockWrapper(nn.Module):
    """Wrapper для ResBlock, который работает с TensorDict"""
    
    def __init__(self, res_block):
        super().__init__()
        self.res_block = res_block
        
    def forward(self, *args):
        if len(args) == 3:
            x, time_emb, context_emb = args
        elif len(args) == 2:
            x, time_emb = args
            context_emb = None
        else:
            x = args[0]
            time_emb = None
            context_emb = None
            
        return self.res_block(x, time_emb, context_emb)


class ResBlock(nn.Module):
    """3D Residual block with optional attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        context_channels: int = 0,
        dropout: float = 0.1,
        attention: bool = False,
        heads: int = 4,
        kernel_size: int = 3
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels*3)
        )
        self.time_mlp[1].weight.data.zero_()
        self.time_mlp[1].bias.data.zero_()

        self.block1 = nn.Sequential(
            IBN(in_channels, 0.5),
            nn.SiLU(),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size//2,
                padding_mode="reflect",
                groups=1
            )
        )
        
        self.block2 = nn.Sequential(
            IBN(out_channels, 0.5),
            nn.SiLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                padding=kernel_size//2,
                padding_mode="reflect",
                groups=1
            )
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
        if attention:
            self.self_attention = VolumeContextAttention(
                dim=out_channels, 
                context_dim=context_channels, 
                time_dim=time_channels, 
                heads=heads,
                num_contexts=2,
                num_layers=2
            )
        else:
            self.self_attention = nn.Identity()
            self.cross_attention = None
            self.ffnn = nn.Identity()

    def forward(self, x, time_emb=None, context_emb=None):
        
        h = self.block1(x)
        
        if time_emb is not None:
            scale1, shift1, gate1 = (self.time_mlp(time_emb)[:, :, None, None, None]).chunk(3, dim=1)
            h = h * (1 + scale1) + shift1
            h = h + gate1 * self.block2(h)
        else:
            h = h + self.block2(h)
            
        h = h + self.shortcut(x)
        
        if hasattr(self.self_attention, 'forward') and callable(self.self_attention.forward):
            try:
                h = h + self.self_attention(h, time_emb, context_emb)
            except TypeError:
                h = h + self.self_attention(h)
        else:
            h = h + self.self_attention(h)
        
        h = h + self.ffnn(h)

        return h


class SkipConnection(nn.Module):
    """Модуль объединения skip-соединений в стиле DiffNeXt."""

    def __init__(self):
        super().__init__()

    def forward(self, first, second):
        return torch.cat([first, second], dim=1)


class Residual(nn.Module):
    """Модуль объединения skip-соединений в стиле DiffNeXt."""

    def __init__(self):
        super().__init__()

    def forward(self, first, second):
        return first + second


class Interpolation(nn.Module):

    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Downsample(nn.Module):

    def __init__(self, stride, features):
        super().__init__()

        self.pool = nn.AvgPool3d(stride, stride)
        self.p = nn.Parameter(torch.ones(1) * features)

    def forward(self, x):
        return self.pool(x.clamp(1e-6).pow(self.p)).pow(1.0 / self.p)


class Upsample(nn.Module):

    def __init__(self, context_channels, **kwargs):
        super().__init__()

        self.context_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_channels*2, kwargs["in_channels"]*2)
        )
        self.context_mlp[1].weight.data.zero_()
        self.context_mlp[1].bias.data.zero_()

        self.conv = nn.ConvTranspose3d(**kwargs)
        # self.lin = nn.Conv3d(kwargs["in_channels"], kwargs["out_channels"], 1)
        # self.interpolation = Interpolation(kwargs["stride"], mode="trilinear")

    def forward(self, x, context):
        scale, shift = (self.context_mlp(context["context"])[:, :, None, None, None]).chunk(2, dim=1)
        return self.conv(x * (1 + scale) + shift)
        # return self.interpolation(x) + self.conv(x)
        # return self.lin(self.interpolation(x)) + self.conv(x)


class FiLM3d(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.scale = nn.Conv3d(channels, channels, 1)
        self.shift = nn.Conv3d(channels, channels, 1)
        self.scale.weight.data.zero_()
        self.shift.weight.data.zero_()
        self.scale.bias.data.zero_()
        self.shift.bias.data.zero_()

    def forward(self, prev, lower):
        return (1 + self.scale(lower)) * prev + self.shift(lower)
        # return prev + lower


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=1, 
        out_channels=1,
        channels=[32, 64, 128, 256],
        attention_levels=[False, True, True],
        strides=[2, 2, 2],
        num_res_blocks=2,
        condition_embedding_dim=256,
        dropout=0.1,
        use_context=False,
        contexts=["context"]
    ):
        super().__init__()

        self.init_channels = condition_embedding_dim
        self.condition_dim = condition_embedding_dim
        self.use_context = use_context
        
        self.time_embedding = nn.Sequential(
            ScalarEncoding(condition_embedding_dim, scale=100),
            nn.Linear(condition_embedding_dim, condition_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(condition_embedding_dim * 4, condition_embedding_dim),
        )
        
        if use_context:
            contexts = OmegaConf.to_object(contexts)
            
            context_scales = {
                'porosity': 100,
                'surface_area': 10
            }
            
            self.ctx_embedding = nn.Sequential(
                *[
                    TensorDictModule(
                        module=nn.Sequential(
                            ScalarEncoding(
                                condition_embedding_dim, 
                                scale=context_scales.get(ctx, 100)
                            ),
                            nn.Linear(condition_embedding_dim, condition_embedding_dim * 4),
                            nn.SiLU(),
                            nn.Linear(condition_embedding_dim * 4, condition_embedding_dim),
                        ),
                        in_keys=[ctx],
                        out_keys=[ctx]
                    )
                    for ctx in contexts
                 ],
                TensorDictModule(
                    module=lambda *x: torch.cat(x, dim=-1) if len(x) > 1 else x[0],
                    in_keys=contexts,
                    out_keys=["context"]
                )
            )
            self.context_dim = condition_embedding_dim
        else:
            self.context_dim = 0

        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(nn.Sequential(
            TensorDictModule(
                module=nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1, padding_mode="reflect"),
                in_keys=["image"],
                out_keys=["0_level"]
            ),
            TensorDictModule(
                module=ResBlockWrapper(ResBlock(
                    in_channels=channels[0],
                    out_channels=channels[1],
                    time_channels=self.condition_dim,
                    context_channels=self.context_dim,
                    dropout=dropout,
                    attention=attention_levels[0]
                )),
                in_keys=["0_level", "time_emb", "context_emb"],
                out_keys=["0_level"]
            ),
            nn.Sequential(*[
                TensorDictModule(
                    module=ResBlockWrapper(ResBlock(
                        in_channels=channels[1],
                        out_channels=channels[1],
                        time_channels=self.condition_dim,
                        context_channels=self.context_dim,
                        dropout=dropout,
                        attention=attention_levels[0]
                    )),
                    in_keys=["0_level", "time_emb", "context_emb"],
                    out_keys=["0_level"]
                ) for j in range(num_res_blocks - 1)
            ])
        ))

        for i in range(1, len(channels)-1):
            self.down_blocks.append(nn.Sequential(*[
                TensorDictModule(
                    module=nn.AvgPool3d(strides[i-1], strides[i-1]),
                    in_keys=[f"{i-1}_level"],
                    out_keys=[f"{i}_level"]
                ),
                TensorDictModule(
                    module=ResBlockWrapper(ResBlock(
                        in_channels=channels[i],
                        out_channels=channels[i+1],
                        time_channels=self.condition_dim,
                        context_channels=self.context_dim,
                        dropout=dropout,
                        attention=attention_levels[i]
                    )),
                    in_keys=[f"{i}_level", "time_emb", "context_emb"],
                    out_keys=[f"{i}_level"]
                ),
                nn.Sequential(*[
                    TensorDictModule(
                        module=ResBlockWrapper(ResBlock(
                            in_channels=channels[i+1],
                            out_channels=channels[i+1],
                            time_channels=self.condition_dim,
                            context_channels=self.context_dim,
                            dropout=dropout,
                            attention=attention_levels[i]
                        )),
                        in_keys=[f"{i}_level", "time_emb", "context_emb"],
                        out_keys=[f"{i}_level"]
                    ) for j in range(num_res_blocks - 1)
                ])
            ]))
            
        i = len(channels) - 1
        self.down_blocks.append(
            TensorDictModule(
                module=nn.AvgPool3d(strides[i-1], strides[i-1]),
                in_keys=[f"{i-1}_level"],
                out_keys=[f"{i}_level"]
            )
        )
        self.fusion = TensorDictModule(
            module=VolumeContextAttention(
                dim=channels[i],
                context_dim=self.context_dim,
                time_dim=self.condition_dim,
                heads=4,
                num_contexts=len(contexts) if use_context else 0,
                num_layers=3
            ),
            in_keys=[f"{i}_level", "time_emb", "context_emb"],
            out_keys=[f"{i}_level"]
        )

        # channels[0] *= 2
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels)-1, 0, -1):
            self.up_blocks.append(nn.Sequential(*[
                TensorDictModule(
                    module=Upsample(
                        in_channels=channels[i], 
                        out_channels=channels[i], 
                        kernel_size=strides[i-1]*2,
                        stride=strides[i-1],
                        padding=strides[i-1]//2,
                        context_channels=self.context_dim
                    ),
                    in_keys=[f"{i}_level", "context_emb"],
                    out_keys=[f"{i}_level"]
                ),
                TensorDictModule(
                    module=FiLM3d(channels[i]),
                    in_keys=[f"{i-1}_level", f"{i}_level"],
                    out_keys=[f"{i-1}_level"]
                ),
                TensorDictModule(
                    module=ResBlockWrapper(ResBlock(
                        in_channels=channels[i],
                        out_channels=channels[i],
                        time_channels=self.condition_dim,
                        context_channels=self.context_dim,
                        dropout=dropout,
                        attention=attention_levels[i-1]
                    )),
                    in_keys=[f"{i-1}_level", "time_emb", "context_emb"],
                    out_keys=[f"{i-1}_level"]
                ),
                TensorDictModule(
                    module=ResBlockWrapper(ResBlock(
                        in_channels=channels[i],
                        out_channels=channels[i-1],
                        time_channels=self.condition_dim,
                        context_channels=self.context_dim,
                        dropout=dropout,
                        attention=attention_levels[i-1]
                    )),
                    in_keys=[f"{i-1}_level", "time_emb", "context_emb"],
                    out_keys=[f"{i-1}_level"]
                ),
                nn.Sequential(*[
                    TensorDictModule(
                        module=ResBlockWrapper(ResBlock(
                            in_channels=channels[i-1],
                            out_channels=channels[i-1],
                            time_channels=self.condition_dim,
                            context_channels=self.context_dim,
                            dropout=dropout,
                            attention=attention_levels[i-1]
                        )),
                        in_keys=[f"{i-1}_level", "time_emb", "context_emb"],
                        out_keys=[f"{i-1}_level"]
                    ) for j in range(num_res_blocks - 2)
                ])
            ]))
            
        self.up_blocks.append(nn.Sequential(
            TensorDictModule(
                module=nn.Conv3d(channels[0], out_channels, kernel_size=1),
                in_keys=["0_level"],
                out_keys=["output"]
            )
        ))
        
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        time_emb = self.time_embedding(t)
        
        if self.use_context and c is not None:
            context_emb = self.ctx_embedding(c)
        else:
            context_emb = None
        
        td = TensorDict(image=x, time_emb=time_emb, context_emb=context_emb)
        
        for block in self.down_blocks:
            td = block(td)

        td = self.fusion(td)
        
        for block in self.up_blocks:
            td = block(td)
        
        return td["output"]
