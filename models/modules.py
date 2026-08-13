import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from functools import partial
from typing import Any, Callable, Optional, Union
from torch import Tensor

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=(1, stride),
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=(1, stride), bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
class RotaryPositionalEmbedding(nn.Module):
    """
    Precompute inv_freq buffer for RoPE. Use get_cos_sin(seq_len, device)
    to obtain cos and sin of shape (seq_len, head_dim//2).
    - head_dim must be even.
    """
    def __init__(self, head_dim, base=10000.0, max_seq_len=2048):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        # inv_freq length = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)  # shape (head_dim//2,)
        self.max_seq_len = max_seq_len

    def get_cos_sin(self, seq_len, device):
        """
        Returns cos, sin each shaped (seq_len, head_dim//2)
        """
        # Make sure inv_freq is on target device
        inv_freq = self.inv_freq.to(device)
        pos = torch.arange(seq_len, device=device).float()  # (seq_len,)
        # angles: (seq_len, head_dim//2)
        angles = pos[:, None] * inv_freq[None, :]
        return torch.cos(angles), torch.sin(angles)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply RoPE rotation to q and k.
    q, k: tensors shaped (B, H, L, Dh)  where Dh == head_dim
    cos, sin: (L, Dh/2)

    Returns rotated (q, k) with same shapes.
    """
    # ensure dims
    assert q.ndim == 4 and k.ndim == 4, "expect (B,H,L,Dh)"
    B, H, L, Dh = q.shape
    assert Dh % 2 == 0, "head_dim must be even"
    # reshape to (..., L, Dh//2, 2)
    q_ = q.view(B, H, L, Dh // 2, 2)
    k_ = k.view(B, H, L, Dh // 2, 2)

    # cos,sin: (L, Dh//2) -> (1,1,L,Dh//2) for broadcasting
    cos = cos.view(1, 1, L, Dh // 2)
    sin = sin.view(1, 1, L, Dh // 2)

    q0 = q_[..., 0]  # (B,H,L,Dh//2)
    q1 = q_[..., 1]
    k0 = k_[..., 0]
    k1 = k_[..., 1]

    q_rot0 = q0 * cos - q1 * sin
    q_rot1 = q0 * sin + q1 * cos
    k_rot0 = k0 * cos - k1 * sin
    k_rot1 = k0 * sin + k1 * cos

    q_out = torch.stack((q_rot0, q_rot1), dim=-1).view(B, H, L, Dh)
    k_out = torch.stack((k_rot0, k_rot1), dim=-1).view(B, H, L, Dh)
    return q_out, k_out

class MultiHeadedAttention_Transformer_ROPE(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, max_seq_len=2048):
        super(MultiHeadedAttention_Transformer_ROPE, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # add rotary helper (head-dim = d_k)
        self.rotary = RotaryPositionalEmbedding(self.d_k, max_seq_len=max_seq_len)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Linear projections -> (nbatches, h, L, d_k)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # ===== INSERT RoPE here =====
        # query/key currently (B, H, L, d_k)
        seq_len = query.size(-2)
        cos, sin = self.rotary.get_cos_sin(seq_len, device=query.device)  # (L, d_k//2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        # ============================

        # 2) Apply attention on rotated q/k
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) Concat & final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Encoder_Transformer(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, final_norm=True):
        super(Encoder_Transformer, self).__init__()
        self.layers = clones(layer, N)
        if final_norm: self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return (self.norm(x) if hasattr(self, 'norm') else x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer_Transformer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer_Transformer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print("Scores: ", scores.shape)
    # with autocast(enabled=False):
    if mask is not None:
        scores = scores.float()
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

    
class PositionwiseFeedForward_Transformer(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward_Transformer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding_Transformer(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding_Transformer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
