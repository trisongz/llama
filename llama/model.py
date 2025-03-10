# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
import os
import math
import torch
import contextlib
import logging
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init

from tqdm.auto import tqdm
from torch import nn
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from contextvars import ContextVar
from typing import Optional, Tuple, Type, Union
from dataclasses import dataclass

from llama.utils import (
    get_torch_device,
    get_torch_device_name,
)

logger = logging.getLogger(__name__)

int8_available = False
bnb: object = None

with contextlib.suppress(ImportError):
    os.environ['BITSANDBYTES_NOWELCOME'] = '1'
    import bitsandbytes as bnb
    int8_available = True


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    # Specify device
    device: str = None

    # int8 support
    int8_enabled: bool = False

    @property
    def device_name(self):
        if self.device is None:
            self.device = get_torch_device_name()
        return self.device

    @property
    def torch_device(self):
        return get_torch_device(self.device_name)
    
    @property
    def as_int8(self) -> bool:
        return self.int8_enabled and int8_available



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # sourcery skip: inline-immediately-returned-variable
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # sourcery skip: merge-comparisons
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



class UninitializedLinear(nn.Linear):
    def reset_parameters(self) -> None:
        pass


class InferenceQuantizedLinear(bnb.nn.Linear8bitLt):
    def __init__(self, *args, **kwargs):
        super().__init__(
            has_fp16_weights = False, 
            threshold = 6.0, 
            *args, 
            **kwargs
        )

    def reset_parameters(self) -> None:
        pass


default_quantize: ContextVar[bool] = ContextVar("default_quantize", default=False)


def get_linear_class() -> Type[nn.Linear]:
    if default_quantize.get():
        return InferenceQuantizedLinear
    return UninitializedLinear


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.as_int8:
            self._init_int8_vars(args)
        else:
            self._init_vars(args)
        
        if args.device_name == "cuda":
            self.cache_k = self.cache_k.cuda()
            self.cache_v = self.cache_v.cuda()



    def _init_vars(self, args: ModelArgs):
        """
        Initializes the params based on defaults
        """

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        )
        

    def _init_int8_vars(self, args: ModelArgs):
        """
        Initializes the params based on int8
        """
        self.n_local_heads = (
            args.n_heads // 1
        )  # fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads
        Linear = get_linear_class()
        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        as_int8: bool = False,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        if as_int8:
            Linear = get_linear_class()
            self.w1 = Linear(dim, hidden_dim, bias=False)
            self.w2 = Linear(
                hidden_dim,
                dim,
                bias = False,
            )
            self.w3 = Linear(
                dim,
                hidden_dim,
                bias = False,
            )


        else:
            self.w1 = ColumnParallelLinear(
                dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
            )
            self.w2 = RowParallelLinear(
                hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
            )
            self.w3 = ColumnParallelLinear(
                dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
            )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim = args.dim, 
            hidden_dim = 4 * args.dim, 
            multiple_of = args.multiple_of,
            as_int8 = args.as_int8,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # sourcery skip: inline-immediately-returned-variable
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


def convert_linear_to_bnb(float_linear):
    new_layer = InferenceQuantizedLinear(
        float_linear.in_features,
        float_linear.out_features,
        bias=float_linear.bias is not None,
    )
    new_layer._parameters["weight"] = bnb.nn.Int8Params(
        float_linear.weight.data.cpu(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    if float_linear.bias is not None:
        new_layer._parameters["bias"] = float_linear.bias
    return new_layer



class Transformer(nn.Module):
    def __init__(
        self, 
        params: ModelArgs
    ):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        if self.params.as_int8:
            self._init_int8_vars()
        else:
            self._init_vars()

    def _init_vars(self):  # sourcery skip: class-extract-method
        """
        Initializes standard vars
        """

        self.tok_embeddings = ParallelEmbedding(
            self.params.vocab_size, self.params.dim, init_method=lambda x: x
        )
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.params.n_layers):
            self.layers.append(TransformerBlock(layer_id, self.params))

        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.output = ColumnParallelLinear(
            self.params.dim, self.params.vocab_size, bias=False, init_method=lambda x: x
        )
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )


    def _init_int8_vars(self):
        """
        Initializes int8 vars
        """
        self.tok_embeddings = torch.nn.Embedding(self.params.vocab_size, self.params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.params.n_layers):
            self.layers.append(TransformerBlock(layer_id, self.params))

        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)

        Linear = get_linear_class()
        self.output = Linear(self.params.dim, self.params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )


    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()

    def quantize(self):
        # https://github.com/pytorch/vision/issues/2391#issuecomment-653900218
        def get_layer(model, name):
            layer = model
            for attr in name.split("."):
                layer = getattr(layer, attr)
            return layer

        def set_layer(model, name, layer):
            with contextlib.suppress(ValueError):
                attrs, name = name.rsplit(".", 1)
                model = get_layer(model, attrs)
            setattr(model, name, layer)

        linear_layers = {
            k: v for k, v in self.named_modules() if isinstance(v, nn.Linear)
        }
        logger.info(f"Quantizing: {len(linear_layers)} layers")
        for name, layer in tqdm(linear_layers.items()):
            new_layer = convert_linear_to_bnb(layer)
            set_layer(self, name, new_layer)
        
        # Handle CUDA?
        if self.params.device_name == 'cuda':
            self.cuda()
