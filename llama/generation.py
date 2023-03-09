# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
import os
import sys
import time
import torch
import json
import logging
from typing import List, Optional, Union, Callable
from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Transformer, default_quantize, int8_available
from llama.utils import (
    get_torch_device,
    setup_model_parallel,
    get_torch_default_tensor_type,
)


from pathlib import Path

logger = logging.getLogger(__name__)

class LLaMA:
    def __init__(
        self, 
        model: Transformer, 
        tokenizer: Tokenizer,
        device: Optional[Union[str, torch.device]] = None,
        params: Optional[ModelArgs] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.params = params or model.params
        device = device or self.params.torch_device
        if isinstance(device, str):
            device = get_torch_device(device)
        self.device = device

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int = 1024,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.0, #0.95,
        repetition_penalty: float = (1.0 / 0.85),
        
        repetition_penalty_range: int = 1024,
        repetition_penalty_slope: float = 0.7,
        # repetition_penalty: float = 1.15,

        token_callback: Optional[Callable] = None,
        eos_text: Optional[str] = '\n<|endoftext|>\n',
    ) -> List[str]:  # sourcery skip: comprehension-to-generator, low-code-quality
        """
        Implements params from `shawwn/llama`
        
        """
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).to(self.device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        prev_text = ''
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if temperature > 0 and \
                repetition_penalty_range > 0 and \
                repetition_penalty_slope > 0:

                next_token_scores = apply_top_p(logits, top_p)
                next_token_scores = apply_temperature(next_token_scores, temperature)
                next_token_scores = apply_advanced_repetition_penalty(
                    tokens[:, :cur_pos],
                    next_token_scores,
                    repetition_penalty_range,
                    repetition_penalty_slope,
                    repetition_penalty,
                )
                next_token_scores = torch.nn.functional.softmax(
                    next_token_scores, dim=-1
                )
                next_token = torch.multinomial(
                    next_token_scores, num_samples=1
                ).squeeze(1)

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            elif repetition_penalty != 1.0:
                logits_new = logits.clone()
                batch_size = len(tokens)
                for i in range(batch_size):
                    for token in set(tokens[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if logits[i, token] < 0:
                            logits_new[i, token] = logits[i, token] * repetition_penalty
                        else:
                            logits_new[i, token] = logits[i, token] / repetition_penalty
                logits = logits_new

                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = sample(probs, top_p=top_p, top_k=top_k)
            
            elif temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample(probs, top_p=top_p, top_k=top_k)
            
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            if next_token == self.tokenizer.eos_id:
                break
            tokens[:, cur_pos] = next_token
            if token_callback is not None:
                assert len(prompts) == 1
                text, = self.decode(tokens)
                #assert text.startswith(prev_text)
                if not text.startswith(prev_text):
                    # Some kind of bogus token generation; abort early.
                    break
                next_word = text[len(prev_text):]
                prev_text = text
                token_callback(next_word)
            prev_pos = cur_pos

        return self.decode(tokens, eos_text = eos_text)
    
    def decode(
        self, 
        tokens: torch.Tensor,
        eos_text: Optional[str] = '\n<|endoftext|>\n',
    ):  # sourcery skip: remove-unused-enumerate
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            t = [token for token in t if token != -1]
            # # cut to max gen len
            # t = t[: len(prompt_tokens[i]) + max_gen_len]
            while self.tokenizer.eos_id in t:
                pos = t.index(self.tokenizer.eos_id)
                t[pos:pos+1] = self.tokenizer.encode(eos_text, bos=False, eos=False)
            decoded.append(self.tokenizer.decode(t))
        return decoded
    
    
    @classmethod
    def load_dist(
        cls,
        ckpt_dir: str,
        max_seq_len: int,
        max_batch_size: int,
        local_rank: int,
        world_size: int,

        device: Optional[str] = None,
        tokenizer_path: Optional[str] = None,

    ) -> 'LLaMA':
        """
        Load a model from a checkpoint directory in distributed mode
        """

        start_time = time.time()
        ckpt_path = Path(ckpt_dir)
        checkpoints = sorted(ckpt_path.glob("*.pth"))
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        curr_ckpt_path = checkpoints[local_rank]
        logger.info(f"Loading checkpoint {curr_ckpt_path}")
        
        checkpoint = torch.load(curr_ckpt_path, map_location="cpu")
        params = json.loads(ckpt_path.joinpath('params.json').read_text())
        model_args: ModelArgs = ModelArgs(
            max_seq_len = max_seq_len, 
            max_batch_size = max_batch_size, 
            device = device,
            **params
        )
        if tokenizer_path: tokenizer_path = Path(tokenizer_path)
        elif ckpt_path.joinpath('tokenizer.model').exists():
            tokenizer_path = ckpt_path.joinpath('tokenizer.model')
        elif ckpt_path.parent.joinpath('tokenizer.model').exists():
            tokenizer_path = ckpt_path.parent.joinpath('tokenizer.model')
        else:
            raise ValueError('Tokenizer not found')
        
        tokenizer = Tokenizer(model_path = tokenizer_path.as_posix())
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(get_torch_default_tensor_type())
        model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict = False)
        generator = LLaMA(model, tokenizer, params = model_args)

        if model_args.device_name == 'cuda':
            logger.info(
                f"Loaded in {time.time() - start_time:.2f} seconds with {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB"
            )
        else:
            logger.info(f"Loaded in {time.time() - start_time:.2f} seconds")
        return generator

    @classmethod
    def load_default(
        cls,
        ckpt_dir: str,
        max_seq_len: int,
        max_batch_size: int,

        device: Optional[str] = None,
        tokenizer_path: Optional[str] = None,

    ) -> 'LLaMA':
        """
        Load a model from a checkpoint directory.
        """

        start_time = time.time()
        ckpt_path = Path(ckpt_dir)
        checkpoints = sorted(ckpt_path.glob("*.pth"))
        logger.info(f"Loading checkpoint {checkpoints}")
        
        params = json.loads(ckpt_path.joinpath('params.json').read_text())
        model_args: ModelArgs = ModelArgs(
            max_seq_len = max_seq_len, 
            max_batch_size = max_batch_size, 
            device = device,
            **params
        )
        if tokenizer_path: tokenizer_path = Path(tokenizer_path)
        elif ckpt_path.joinpath('tokenizer.model').exists():
            tokenizer_path = ckpt_path.joinpath('tokenizer.model')
        elif ckpt_path.parent.joinpath('tokenizer.model').exists():
            tokenizer_path = ckpt_path.parent.joinpath('tokenizer.model')
        else:
            raise ValueError('Tokenizer not found')
        
        tokenizer = Tokenizer(model_path = tokenizer_path.as_posix())
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(get_torch_default_tensor_type())
        model = Transformer(model_args)
        key_to_dim = {
            "w1": 0,
            "w2": -1,
            "w3": 0,
            "wo": -1,
            "wq": 0,
            "wk": 0,
            "wv": 0,
            "output": 0,
            "tok_embeddings": -1,
            "ffn_norm": None,
            "attention_norm": None,
            "norm": None,
            "rope": None,
        }

        torch.set_default_tensor_type(torch.FloatTensor)
        # load the state dict incrementally, to avoid memory problems
        for i, ckpt in enumerate(checkpoints):
            logger.info(f"Loading checkpoint {i}/{len(checkpoints)}")
            checkpoint = torch.load(ckpt, map_location="cpu")
            for parameter_name, parameter in model.named_parameters():
                short_name = parameter_name.split(".")[-2]
                if key_to_dim[short_name] is None and i == 0:
                    parameter.data = checkpoint[parameter_name]
                elif key_to_dim[short_name] == 0:
                    size = checkpoint[parameter_name].size(0)
                    parameter.data[size * i : size * (i + 1), :] = checkpoint[
                        parameter_name
                    ]
                elif key_to_dim[short_name] == -1:
                    size = checkpoint[parameter_name].size(-1)
                    parameter.data[:, size * i : size * (i + 1)] = checkpoint[
                        parameter_name
                    ]
                del checkpoint[parameter_name]
            del checkpoint
        
        if model_args.device_name == 'cuda':
            model.cuda()
        
        generator = LLaMA(model, tokenizer, params = model_args)
        if model_args.device_name == 'cuda':
            logger.info(
                f"Loaded in {time.time() - start_time:.2f} seconds with {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB"
            )
        else:
            logger.info(f"Loaded in {time.time() - start_time:.2f} seconds")
        return generator

    @classmethod
    def load_int8(
        cls,
        ckpt_dir: str,
        max_seq_len: int = 512,
        max_batch_size: int = 32,
        device: Optional[str] = None,
        quantize: Optional[bool] = True,
        tokenizer_path: Optional[str] = None,

    ) -> 'LLaMA':
        """
        Load a model from a checkpoint directory using int8.
        """
        start_time = time.time()
        ckpt_path = Path(ckpt_dir)

        checkpoints = sorted(ckpt_path.glob("*.pth"))
        params = json.loads(ckpt_path.joinpath('params.json').read_text())
        model_args: ModelArgs = ModelArgs(
            max_seq_len = max_seq_len, 
            max_batch_size = max_batch_size, 
            int8_enabled = True,
            device = device,
            **params
        )
        if tokenizer_path: tokenizer_path = Path(tokenizer_path)
        elif ckpt_path.joinpath('tokenizer.model').exists():
            tokenizer_path = ckpt_path.joinpath('tokenizer.model')
        elif ckpt_path.parent.joinpath('tokenizer.model').exists():
            tokenizer_path = ckpt_path.parent.joinpath('tokenizer.model')
        else:
            raise ValueError('Tokenizer not found')
    
        logger.info(f"Loading checkpoint {ckpt_path}")

        tokenizer = Tokenizer(model_path = tokenizer_path.as_posix())
        model_args.vocab_size = tokenizer.n_words

        torch.set_default_tensor_type(torch.HalfTensor)
        logger.info("Allocating transformer on host")
        ctx_tok = default_quantize.set(quantize)
        model = Transformer(model_args)
        default_quantize.reset(ctx_tok)
        key_to_dim = {
            "w1": 0,
            "w2": -1,
            "w3": 0,
            "wo": -1,
            "wq": 0,
            "wk": 0,
            "wv": 0,
            "output": 0,
            "tok_embeddings": -1,
            "ffn_norm": None,
            "attention_norm": None,
            "norm": None,
            "rope": None,
        }

        # ?
        torch.set_default_tensor_type(torch.FloatTensor)

        # load the state dict incrementally, to avoid memory problems
        for i, ckpt in enumerate(checkpoints):
            logger.info(f"Loading checkpoint {i}/{len(checkpoints)}")
            checkpoint = torch.load(ckpt, map_location="cpu")
            for parameter_name, parameter in model.named_parameters():
                short_name = parameter_name.split(".")[-2]
                if key_to_dim[short_name] is None and i == 0:
                    parameter.data = checkpoint[parameter_name]
                elif key_to_dim[short_name] == 0:
                    size = checkpoint[parameter_name].size(0)
                    parameter.data[size * i : size * (i + 1), :] = checkpoint[
                        parameter_name
                    ]
                elif key_to_dim[short_name] == -1:
                    size = checkpoint[parameter_name].size(-1)
                    parameter.data[:, size * i : size * (i + 1)] = checkpoint[
                        parameter_name
                    ]
                del checkpoint[parameter_name]
            del checkpoint
        if model_args.device_name == 'cuda':
            model.cuda()

        generator = LLaMA(model, tokenizer, params = model_args)
        if model_args.device_name == 'cuda':
            logger.info(
                f"Loaded in {time.time() - start_time:.2f} seconds with {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB"
            )
        else:
            logger.info(f"Loaded in {time.time() - start_time:.2f} seconds")
        return generator

    @classmethod
    def load(
        cls,
        ckpt_dir: str,
        max_seq_len: int = 512,
        max_batch_size: int = 32,
        seed: Optional[int] = 1,
        device: Optional[str] = None,
        quantize: Optional[bool] = True,
        int8_enabled: Optional[bool] = int8_available,
        tokenizer_path: Optional[str] = None,
        is_dist_mode: Optional[bool] = False,
    ) -> 'LLaMA':
        """
        Load a model from a checkpoint directory.
        """
        if int8_enabled and not int8_available:
            logger.warning('Int8 is not available. Install `bitsnbytes` to enable it.')
            int8_enabled = False
        if int8_enabled:
            return cls.load_int8(
                ckpt_dir = ckpt_dir,
                max_seq_len = max_seq_len,
                max_batch_size = max_batch_size,
                device = device,
                quantize = quantize,
                tokenizer_path = tokenizer_path,
            )
        
        if not is_dist_mode:
            return cls.load_default(
                ckpt_dir = ckpt_dir,
                max_seq_len = max_seq_len,
                max_batch_size = max_batch_size,
                device = device,
                tokenizer_path = tokenizer_path,
            )

        local_rank, world_size = setup_model_parallel(seed)
        if local_rank > 0: sys.stdout = open(os.devnull, "w")
        return cls.load_default(
            ckpt_dir = ckpt_dir,
            max_seq_len = max_seq_len,
            max_batch_size = max_batch_size,
            device = device,
            local_rank = local_rank,
            world_size = world_size,
            tokenizer_path = tokenizer_path,
        )


def apply_temperature(scores, tempt):
    scores = scores / tempt
    return scores

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def sample(probs, top_p: float = 0.0, top_k: int = 40):
    if top_k > 0:
        probs_sort, probs_idx = torch.topk(probs, top_k)
    else:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    if top_p > 0.0:
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def apply_top_p(scores, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1):
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def apply_advanced_repetition_penalty(
    input_ids, scores, penalty_range, penalty_slope, penalty
):
    penalty_range = int(penalty_range)
    clipped_penalty_range = min(input_ids.shape[-1], penalty_range)

    if penalty != 1.0:
        if penalty_range > 0:
            if clipped_penalty_range < input_ids.shape[1]:
                input_ids = input_ids[..., -clipped_penalty_range:]

            if penalty_slope != 0:
                _penalty = (
                    torch.arange(
                        penalty_range, dtype=scores.dtype, device=scores.device
                    )
                    / (penalty_range - 1)
                ) * 2.0 - 1
                _penalty = (penalty_slope * _penalty) / (
                    1 + torch.abs(_penalty) * (penalty_slope - 1)
                )
                _penalty = 1 + ((_penalty + 1) / 2).unsqueeze(0) * (penalty - 1)
                penalty = _penalty[..., -clipped_penalty_range:]

        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score <= 0, score * penalty, score / penalty)
        scores.scatter_(1, input_ids, score)

    return scores