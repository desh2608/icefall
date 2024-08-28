#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang,
#                                                  Mingshuang Luo,)
#                                                  Zengwei Yao)
#              2023  Johns Hopkins University (author: Desh Raj)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:

export CUDA_VISIBLE_DEVICES="0,1,2,3"

cd egs/libricss/SURT
./prepare.sh

./dprnn_zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir dprnn_zipformer/exp \
  --max-duration 300

# For mix precision training:

./dprnn_zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir dprnn_zipformer/exp \
  --max-duration 550
"""

import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import AmiAsrDataModule
from decoder import Decoder
from dprnn import DPRNN
from einops import rearrange
from einops.layers.torch import Rearrange
from joiner import Joiner
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import LOG_EPSILON, fix_random_seed
from model2 import SURT
from optim import Eden, ScaledAdam
from scaling import ScaledLSTM, ScheduledFloat
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer2

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-mask-encoder-layers",
        type=int,
        default=4,
        help="Number of layers in the DPRNN based mask encoder.",
    )

    parser.add_argument(
        "--mask-encoder-dim",
        type=int,
        default=256,
        help="Hidden dimension of the LSTM blocks in DPRNN.",
    )

    parser.add_argument(
        "--mask-encoder-segment-size",
        type=int,
        default=32,
        help="Segment size of the SegLSTM in DPRNN. Ideally, this should be equal to the "
        "decode-chunk-length of the zipformer encoder.",
    )

    parser.add_argument(
        "--chunk-width-randomization",
        type=bool,
        default=True,
        help="Whether to randomize the chunk width in DPRNN.",
    )

    # Zipformer config is based on the "small" Zipformer here:
    # https://github.com/k2-fsa/icefall/pull/1058
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,2,2,2,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,768,768,768,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,256,256,256,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,192,192,192,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--use-joint-encoder-layer",
        type=str,
        default="lstm",
        choices=["linear", "lstm", "none"],
        help="Whether to use a joint layer to combine all branches.",
    )

    # Config for auxiliary encoder
    parser.add_argument(
        "--aux-output-layer",
        type=int,
        default=2,
        help="Output from this layer of the main encoder will be fed into the auxiliary "
        "encoder to use for speaker attribution.",
    )

    parser.add_argument(
        "--use-aux-encoder",
        type=str2bool,
        default=True,
        help="Whether to use an auxiliary encoder for speaker attribution.",
    )

    parser.add_argument(
        "--aux-num-encoder-layers",
        type=str,
        default="2,2,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--aux-downsampling-factor",
        type=str,
        default="2,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--aux-feedforward-dim",
        type=str,
        default="256,256,256",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--aux-num-heads",
        type=str,
        default="4,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--aux-encoder-dim",
        type=str,
        default="256,256,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--aux-query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--aux-value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--aux-pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--aux-pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--aux-encoder-unmasked-dim",
        type=str,
        default="128,128,128",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--aux-cnn-module-kernel",
        type=str,
        default="31,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--use-aux-joint-encoder-layer",
        type=str,
        default="lstm",
        choices=["linear", "lstm", "none", "zipformer"],
        help="Whether to use a joint layer to combine all aux branches.",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--aux-joiner-dim",
        type=int,
        default=256,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        help="Maximum number of speakers in each mixture.",
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=True,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )

    parser.add_argument(
        "--aux-chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--aux-left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )

    parser.add_argument(
        "--aux-pos-emb-skip-rate-schedule",
        type=str,
        default="none",
        choices=["none", "increase", "decrease", "full"],
        help="Whether to skip positional embedding in the auxiliary encoder.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conv_lstm_transducer_stateless_ctc/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--model-init-ckpt",
        type=str,
        default=None,
        help="""The model checkpoint to initialize the model (either full or part).
        If not specified, the model is randomly initialized.
        """,
    )

    parser.add_argument(
        "--aux-model-init-ckpt",
        type=str,
        default=None,
        help="""The model checkpoint to initialize the auxiliary speaker encoder.""",
    )

    parser.add_argument(
        "--freeze-main-model",
        type=str2bool,
        default=False,
        help="""If true, the main model is frozen. Only the auxiliary speaker encoder
        is trained.
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.004, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=6,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network) part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--ctc-loss-scale",
        type=float,
        default=0.2,
        help="Scale for CTC loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=2000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=1,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=100,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    # Options related to speaker prefixing
    parser.add_argument(
        "--use-speaker-prefixing",
        type=str2bool,
        default=False,
        help="Whether to prefix random speakers to train the auxiliary branch.",
    )

    parser.add_argument(
        "--speaker-buffer-frames",
        type=str,
        default="128",
        help="Number of frames to prefix for each speaker.",
    )

    parser.add_argument(
        "--num-prefix-speakers",
        type=str,
        default="0,1,2,3,4",
        help="Number of speakers to prefix.",
    )

    parser.add_argument(
        "--num-prefix-speakers-probs",
        type=str,
        default="0.05,0.05,0.1,0.2,0.6",
        help="Probability of prefixing different number of speakers.",
    )

    parser.add_argument(
        "--fixed-prefix-speakers",
        type=str2bool,
        default=False,
        help="If True, we will always prefix a buffer of size max_speakers x speaker_buffer_frames "
        "padding with zeros if there are fewer speakers than max_speakers.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 2000,
            # parameters for SURT
            "num_channels": 2,
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed
            # parameters for Noam
            "model_warm_step": 5000,  # arg given to model, not for lrate
            # parameters for ctc loss
            "beam_size": 10,
            "use_double_scores": True,
            "env_info": get_env_info(),
        }
    )

    return params


def get_mask_encoder_model(params: AttributeDict) -> nn.Module:
    mask_encoder = DPRNN(
        feature_dim=params.feature_dim,
        input_size=params.mask_encoder_dim,
        hidden_size=params.mask_encoder_dim,
        output_size=params.feature_dim * params.num_channels,
        segment_size=params.mask_encoder_segment_size,
        num_blocks=params.num_mask_encoder_layers,
        chunk_width_randomization=params.chunk_width_randomization,
    )
    return mask_encoder


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def _to_float_tuple(s: str):
    return tuple(map(float, s.split(",")))


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
        aux_output_layer=params.aux_output_layer if params.use_aux_encoder else None,
    )
    return encoder


def get_aux_encoder_model(params: AttributeDict) -> nn.Module:
    if params.aux_pos_emb_skip_rate_schedule == "none":
        pos_emb_skip_rate = 0.0
    elif params.aux_pos_emb_skip_rate_schedule == "full":
        pos_emb_skip_rate = 1.0
    elif params.aux_pos_emb_skip_rate_schedule == "increase":
        pos_emb_skip_rate = ScheduledFloat((0.0, 0.0), (4000.0, 1.0), default=1.0)
    elif params.aux_pos_emb_skip_rate_schedule == "decrease":
        pos_emb_skip_rate = ScheduledFloat((0.0, 1.0), (4000.0, 0.0), default=0.0)
    else:
        raise ValueError(
            f"Unknown aux_pos_emb_skip_rate_schedule: {params.aux_pos_emb_skip_rate_schedule}"
        )
    encoder_dims = _to_int_tuple(params.encoder_dim)
    aux_input_dim = (
        encoder_dims[params.aux_output_layer]
        if params.aux_output_layer >= 0
        else encoder_dims[0]
    )
    aux_encoder_dims = _to_int_tuple(params.aux_encoder_dim)
    assert aux_input_dim == aux_encoder_dims[0], (
        f"aux_input_dim ({aux_input_dim}) != aux_encoder_dims[0] "
        f"({aux_encoder_dims[0]})"
    )
    aux_encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.aux_downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.aux_num_encoder_layers),
        encoder_dim=_to_int_tuple(params.aux_encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.aux_encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.aux_query_head_dim),
        pos_head_dim=_to_int_tuple(params.aux_pos_head_dim),
        value_head_dim=_to_int_tuple(params.aux_value_head_dim),
        pos_dim=params.aux_pos_dim,
        num_heads=_to_int_tuple(params.aux_num_heads),
        feedforward_dim=_to_int_tuple(params.aux_feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.aux_cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.aux_chunk_size),
        left_context_frames=_to_int_tuple(params.aux_left_context_frames),
        pos_emb_skip_rate=pos_emb_skip_rate,
    )
    return aux_encoder


def get_joint_encoder_layer(params: AttributeDict) -> nn.Module:
    class TakeFirst(nn.Module):
        def forward(self, x):
            return x[0]

    if params.use_joint_encoder_layer == "linear":
        encoder_dim = int(params.encoder_dim.split(",")[-1])
        joint_layer = nn.Sequential(
            Rearrange("(c b) t d -> b t (c d)", c=params.num_channels),
            nn.Linear(
                params.num_channels * encoder_dim, params.num_channels * encoder_dim
            ),
            nn.ReLU(),
            Rearrange("b t (c d) -> (c b) t d", c=params.num_channels),
        )
    elif params.use_joint_encoder_layer == "lstm":
        encoder_dim = int(params.encoder_dim.split(",")[-1])
        joint_layer = nn.Sequential(
            Rearrange("(c b) t d -> b t (c d)", c=params.num_channels),
            ScaledLSTM(
                input_size=params.num_channels * encoder_dim,
                hidden_size=params.num_channels * encoder_dim,
                num_layers=1,
                bias=True,
                batch_first=True,
                dropout=0.0,
                bidirectional=False,
            ),
            TakeFirst(),
            nn.ReLU(),
            Rearrange("b t (c d) -> (c b) t d", c=params.num_channels),
        )
    elif params.use_joint_encoder_layer == "none":
        joint_layer = None
    else:
        raise ValueError(
            f"Unknown joint encoder layer type: {params.use_joint_encoder_layer}"
        )
    return joint_layer


def get_aux_joint_encoder_layer(params: AttributeDict) -> nn.Module:
    class AuxJointEncoderLayer(nn.Module):
        def __init__(
            self,
            aux_encoder_dim: int,
            num_channels: int,
            type: str,
        ):
            super().__init__()
            self.type = type
            self.num_channels = num_channels

            if type == "linear":
                self.layer = nn.Linear(
                    num_channels * aux_encoder_dim, num_channels * aux_encoder_dim
                )
            elif type == "lstm":
                self.layer = ScaledLSTM(
                    input_size=num_channels * aux_encoder_dim,
                    hidden_size=num_channels * aux_encoder_dim,
                    num_layers=1,
                    bias=True,
                    batch_first=True,
                    dropout=0.0,
                    bidirectional=False,
                )
            elif type == "zipformer":
                pass

            elif params.use_aux_joint_encoder_layer == "none":
                self.layer = nn.Identity()
            else:
                raise ValueError(f"Unknown joint encoder layer type: {type}")

        def forward(self, x, x_lens):
            x = rearrange(x, "(c b) t d -> b t (c d)", c=self.num_channels)
            if self.type == "linear":
                x = self.layer(x)
                x = nn.functional.relu(x)
            elif self.type == "lstm":
                x, _ = self.layer(x)
                x = nn.functional.relu(x)
            elif self.type == "zipformer":
                x, x_lens = self.layer(x, x_lens)
            x = rearrange(x, "b t (c d) -> (c b) t d", c=self.num_channels)
            return x

    aux_encoder_dim = int(params.aux_encoder_dim.split(",")[-1])
    return AuxJointEncoderLayer(
        aux_encoder_dim, params.num_channels, params.use_aux_joint_encoder_layer
    )


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=int(params.encoder_dim.split(",")[-1]),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_aux_joiner_model(params: AttributeDict) -> nn.Module:
    aux_joiner = Joiner(
        encoder_dim=int(params.aux_encoder_dim.split(",")[-1]),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.aux_joiner_dim,
        vocab_size=params.max_speakers,
    )
    return aux_joiner


def get_surt_model(
    params: AttributeDict,
) -> nn.Module:
    mask_encoder = get_mask_encoder_model(params)
    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)
    joint_layer = get_joint_encoder_layer(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    if params.use_aux_encoder:
        aux_encoder = get_aux_encoder_model(params)
        aux_joiner = get_aux_joiner_model(params)
        aux_joint_layer = get_aux_joint_encoder_layer(params)
    else:
        aux_encoder = None
        aux_joiner = None
        aux_joint_layer = None

    model = SURT(
        mask_encoder=mask_encoder,
        encoder_embed=encoder_embed,
        encoder=encoder,
        joint_encoder_layer=joint_layer,
        decoder=decoder,
        joiner=joiner,
        aux_encoder=aux_encoder,
        aux_joint_encoder_layer=aux_joint_layer,
        aux_joiner=aux_joiner,
        num_channels=params.num_channels,
        encoder_dim=int(params.encoder_dim.split(",")[-1]),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute RNN-T loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Conformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"].to(device)
    feature_lens = batch["input_lens"].to(device)

    # at entry, feature is (N, T, C)
    assert feature.ndim == 3

    # The dataloader returns text as a list of cuts, each of which is a list of channel
    # text. We flatten this to a list where all channels are together, i.e., it looks like
    # [utt1_ch1, utt2_ch1, ..., uttN_ch1, utt1_ch2, ...., uttN,ch2].
    text = [val for tup in zip(*batch["text"]) for val in tup]
    assert len(text) == len(feature) * params.num_channels

    # Convert all channel texts to token IDs and create a ragged tensor.
    if params.use_aux_encoder:
        # In this case, each text would be |-delimited, and we need to encode each separately
        # so that we know how many tokens are in each segment.
        text_list = [t.split("|") for t in text]
        y = [sp.encode(t, out_type=int) for t in text_list]
        num_tokens = [[len(t) for t in utt] for utt in y]
        # Replace "|" with space so that we can encode the whole utterance as a single string.
        text = [utt.replace("|", " ") for utt in text]

        # Now create supervisions for the aux encoder.
        speakers = [val for tup in zip(*batch["speakers"]) for val in tup]
        assert len(speakers) == len(feature) * params.num_channels
        # Repeat each speaker ID for each token in the corresponding text. We also
        # limit the number of speakers to params.max_speakers.
        speakers = [
            [
                min(params.max_speakers, speaker)
                for speaker, num_token in zip(speakers_utt, num_tokens_utt)
                for _ in range(num_token)
            ]
            for speakers_utt, num_tokens_utt in zip(speakers, num_tokens)
        ]
        y_spk = k2.RaggedTensor(speakers).to(device)

    if params.use_speaker_prefixing and batch["num_prefix_speakers"] > 0:
        prefix = batch["speaker_prefix"].to(device)  # B x T x D
        if params.fixed_prefix_speakers:
            # Pad prefix to make T dimension equal to max_speakers x speaker_buffer_frames.
            B, T, D = prefix.shape
            prefix = torch.cat(
                [
                    prefix,
                    torch.zeros(
                        B,
                        params.max_speakers * batch["speaker_buffer_frames"] - T,
                        D,
                        device=device,
                    ),
                ],
                dim=1,
            )
        num_prefix_frames = prefix.shape[1]
        feature = torch.cat([prefix, feature], dim=1)
        feature_lens += num_prefix_frames
    else:
        num_prefix_frames = 0

    y = sp.encode(text, out_type=int)
    y = k2.RaggedTensor(y).to(device)

    batch_idx_train = params.batch_idx_train
    warm_step = params.model_warm_step

    with torch.set_grad_enabled(is_training):
        (simple_loss, pruned_loss, ctc_loss, aux_spk_loss, x_masked) = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            y_spk=y_spk,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            reduction="none",
            subsampling_factor=params.subsampling_factor,
            num_prefix_frames=num_prefix_frames,
        )
        simple_loss_is_finite = torch.isfinite(simple_loss)
        pruned_loss_is_finite = torch.isfinite(pruned_loss)
        ctc_loss_is_finite = torch.isfinite(ctc_loss)
        aux_spk_loss_is_finite = torch.isfinite(aux_spk_loss)

        is_finite = (
            simple_loss_is_finite
            & pruned_loss_is_finite
            & ctc_loss_is_finite
            & aux_spk_loss_is_finite
        )
        if not torch.all(is_finite):
            logging.info(
                "Not all losses are finite!\n"
                f"simple_losses: {simple_loss}\n"
                f"pruned_losses: {pruned_loss}\n"
                f"ctc_losses: {ctc_loss}\n"
                f"aux_spk_losses: {aux_spk_loss}\n"
            )
            # display_and_save_batch(batch, params=params, sp=sp)
            simple_loss = simple_loss[simple_loss_is_finite]
            pruned_loss = pruned_loss[pruned_loss_is_finite]
            ctc_loss = ctc_loss[ctc_loss_is_finite]
            aux_spk_loss = aux_spk_loss[aux_spk_loss_is_finite]

            # If either all simple_loss or pruned_loss is inf or nan,
            # we stop the training process by raising an exception
            if (
                torch.all(~simple_loss_is_finite)
                or torch.all(~pruned_loss_is_finite)
                or torch.all(~ctc_loss_is_finite)
                or torch.all(~aux_spk_loss_is_finite)
            ):
                raise ValueError(
                    "There are too many utterances in this batch "
                    "leading to inf or nan losses."
                )

        simple_loss_sum = simple_loss.sum()
        pruned_loss_sum = pruned_loss.sum()
        ctc_loss_sum = ctc_loss.sum()
        aux_spk_loss_sum = aux_spk_loss.sum()

        s = params.simple_loss_scale
        # take down the scale on the simple loss from 1.0 at the start
        # to params.simple_loss scale by warm_step.
        simple_loss_scale = (
            s
            if batch_idx_train >= warm_step
            else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
        )
        pruned_loss_scale = (
            1.0
            if batch_idx_train >= warm_step
            else 0.1 + 0.9 * (batch_idx_train / warm_step)
        )
        loss = (
            simple_loss_scale * simple_loss_sum
            + pruned_loss_scale * pruned_loss_sum
            + params.ctc_loss_scale * ctc_loss_sum
            + aux_spk_loss_sum
        )

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # info["frames"] is an approximate number for two reasons:
        # (1) The acutal subsampling factor is ((lens - 1) // 2 - 1) // 2
        # (2) If some utterances in the batch lead to inf/nan loss, they
        #     are filtered out.
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # `utt_duration` and `utt_pad_proportion` would be normalized by `utterances`  # noqa
    info["utterances"] = feature.size(0)
    # averaged input duration in frames over utterances
    info["utt_duration"] = feature_lens.sum().item()
    # averaged padding proportion over utterances
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss_sum.detach().cpu().item()
    info["pruned_loss"] = pruned_loss_sum.detach().cpu().item()
    if params.ctc_loss_scale > 0.0:
        info["ctc_loss"] = ctc_loss_sum.detach().cpu().item()
    if params.use_aux_encoder:
        info["aux_spk_loss"] = aux_spk_loss_sum.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    torch.cuda.empty_cache()
    model.train()

    tot_loss = MetricsTracker()

    cur_batch_idx = params.get("cur_batch_idx", 0)

    for batch_idx, batch in enumerate(train_dl):
        if batch_idx < cur_batch_idx:
            continue
        cur_batch_idx = batch_idx

        params.batch_idx_train += 1
        batch_size = batch["inputs"].shape[0]

        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            set_batch_count(model, params.batch_idx_train)
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except:  # noqa
            display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            params.cur_batch_idx = batch_idx
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            del params.cur_batch_idx
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()
            if cur_grad_scale < 1.0 or (cur_grad_scale < 8.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                sp=sp,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_surt_model(params)

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)

    # We first load the state dict for aux model, then load the state dict for
    # the main model. This is because the aux model is a subset of the main model.
    if checkpoints is None and params.aux_model_init_ckpt is not None:
        logging.info(
            f"Initializing aux model with checkpoint from {params.aux_model_init_ckpt}"
        )
        init_ckpt = torch.load(params.aux_model_init_ckpt, map_location=device)
        model.load_state_dict(init_ckpt["model"], strict=False)

    if checkpoints is None and params.model_init_ckpt is not None:
        logging.info(
            f"Initializing model with checkpoint from {params.model_init_ckpt}"
        )
        init_ckpt = torch.load(params.model_init_ckpt, map_location=device)
        model.load_state_dict(init_ckpt["model"], strict=False)

    if params.freeze_main_model:
        logging.info("Freezing main model")
        for name, p in model.named_parameters():
            if "aux" not in name:
                p.requires_grad = False

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")
    num_trainable_param = sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )
    logging.info(f"Number of trainable parameters: {num_trainable_param}")

    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    parameters_names = []
    parameters_names.append(
        [name_param_pair[0] for name_param_pair in model.named_parameters()]
    )
    optimizer = ScaledAdam(
        model.parameters(),
        lr=params.base_lr,
        clipping_scale=2.0,
        parameters_names=parameters_names,
    )

    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        diagnostic = diagnostics.attach_diagnostics(model)

    ami = AmiAsrDataModule(args)

    train_cuts = ami.train_cuts(use_speaker_prefixing=params.use_speaker_prefixing)
    train_cuts = train_cuts.filter(lambda c: 0.5 <= c.duration <= 35.0)
    if params.use_speaker_prefixing:
        train_cuts = train_cuts.filter(lambda c: len(c.speakers) <= params.max_speakers)
    dev_cuts = ami.ami_cuts(
        split="dev",
        type="ihm-mix",
        groups=True,
        use_speaker_prefixing=params.use_speaker_prefixing,
    ).subset(first=200)
    # dev_cuts = dev_cuts.trim_to_supervision_groups(max_pause=0.0).filter(
    #     lambda c: 0.2 <= c.duration <= 60.0
    # )

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    text_delimiter = "|" if params.use_aux_encoder else " "
    train_dl = ami.train_dataloaders(
        train_cuts,
        sampler_state_dict=sampler_state_dict,
        text_delimiter=text_delimiter,
        return_sources=False,
        use_speaker_prefixing=params.use_speaker_prefixing,
        max_speakers=params.max_speakers,
        num_prefix_speakers=_to_int_tuple(params.num_prefix_speakers),
        num_prefix_speakers_probs=_to_float_tuple(params.num_prefix_speakers_probs),
        speaker_buffer_frames=_to_int_tuple(params.speaker_buffer_frames),
    )
    valid_dl = ami.valid_dataloaders(
        dev_cuts,
        text_delimiter=text_delimiter,
        use_speaker_prefixing=params.use_speaker_prefixing,
        max_speakers=params.max_speakers,
        num_prefix_speakers=_to_int_tuple(params.num_prefix_speakers),
        num_prefix_speakers_probs=_to_float_tuple(params.num_prefix_speakers_probs),
        speaker_buffer_frames=_to_int_tuple(params.speaker_buffer_frames),
    )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    y = [sp.encode(text_ch) for text_ch in batch["text"]]
    num_tokens = [sum(len(yi) for yi in y_ch) for y_ch in y]
    logging.info(f"num tokens: {num_tokens}")


def main():
    parser = get_parser()
    AmiAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    main()
