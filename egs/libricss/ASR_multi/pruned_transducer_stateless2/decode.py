#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
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
(1) greedy search
./pruned_transducer_stateless2/decode.py \
        --epoch 28 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --max-duration 600 \
        --decoding-method greedy_search

(2) beam search (not recommended)
./pruned_transducer_stateless2/decode.py \
        --epoch 28 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --max-duration 600 \
        --decoding-method beam_search \
        --beam-size 4

(3) modified beam search
./pruned_transducer_stateless2/decode.py \
        --epoch 28 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --max-duration 600 \
        --decoding-method modified_beam_search \
        --beam-size 4

(4) fast beam search
./pruned_transducer_stateless2/decode.py \
        --epoch 28 \
        --avg 15 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --max-duration 600 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8
"""


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriCSSAsrDataModule
from beam_search import (
    beam_search,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from conformer import Conformer
from decoder import Decoder
from joiner import Joiner
from lhotse import CutSet, SupervisionSegment, SupervisionSet
from lhotse.cut import Cut
from lhotse.utils import fastcopy
from model import Transducer

from icefall.checkpoint import load_checkpoint
from icefall.env import get_env_info
from icefall.utils import AttributeDict, setup_logger


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=25,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 0.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless2/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - fast_beam_search
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=4,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help=("The context size in the decoder. 1 means bigram; 2 means tri-gram"),
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

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

        - encoder_dim: Hidden dim for multi-head attention model.

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
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "encoder_dim": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "num_encoder_layers": 12,
            # parameters for decoder
            "decoder_dim": 512,
            # parameters for joiner
            "joiner_dim": 512,
            # parameters for Noam
            "model_warm_step": 3000,  # arg given to model, not for lrate
            "env_info": get_env_info(),
        }
    )

    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    # TODO: We can add an option to switch between Conformer and Transformer
    encoder = Conformer(
        num_features=params.feature_dim,
        subsampling_factor=params.subsampling_factor,
        d_model=params.encoder_dim,
        nhead=params.nhead,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
    )
    return encoder


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
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return model


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = model.device
    feature = batch["features"].to(device)
    feature_lens = batch["features_lens"].to(device)
    assert feature.ndim == 3

    # at entry, feature is (N, T, C)

    encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)
    hyps = []

    if params.decoding_method == "fast_beam_search":
        hyp_tokens = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    else:
        batch_size = encoder_out.size(0)

        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                )
            elif params.decoding_method == "beam_search":
                hyp = beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            hyps.append(sp.decode(hyp).split())

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    elif params.decoding_method == "fast_beam_search":
        return {
            f"beam_{params.beam}_max_contexts_{params.max_contexts}_max_states_{params.max_states}": hyps
        }
    else:
        return {f"beam_size_{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[str, List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the cut id, and the second is the predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 50
    else:
        log_interval = 10

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
            batch=batch,
        )

        cut_ids = [cut.id for cut in batch["cuts"]]

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(cut_ids)
            for cut_id, hyp_words in zip(cut_ids, hyps):
                hyp = " ".join(hyp_words)
                this_batch.append((cut_id, hyp))

            results[name].extend(this_batch)

        num_cuts += len(cut_ids)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    cuts: CutSet,
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[int]]]],
):
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        out_sups = []
        with recog_path.open("w") as f:
            for cut_id, hyp_words in results:
                f.write(f"{cut_id}\t{hyp_words}\n")
                old_cut = cuts[cut_id]
                new_sup = create_new_supervision(old_cut, hyp_words)
                out_sups.append(new_sup)
        out_sups = SupervisionSet.from_segments(out_sups)
        out_sups.to_file(params.res_dir / f"{test_set_name}-{key}-hyps.jsonl.gz")
        logging.info(f"The transcripts are stored in {recog_path}")


def create_new_supervision(cut: Cut, hyp: str) -> SupervisionSegment:
    """Create a new supervision segment with the given hyp.

    Args:
      cut:
        The original cut.
      hyp:
        The hypothesis.
    Returns:
      Return a new supervision segment with the given hyp.
    """
    if len(cut.supervisions) == 0:
        # We will have to extract speaker, start and duration from the cut id.
        # The cut id is like this: OV10_session0-260-031489_031768-235
        reco_id, spk, start_end, *_ = cut.id.split("-")
        start, end = start_end.split("_")
        start = float(start) / 100
        end = float(end) / 100
        duration = end - start
        supervision = SupervisionSegment(
            id=cut.id,
            recording_id=reco_id,
            start=start,
            duration=duration,
            channel=0,
            language="English",
            speaker=spk,
            text=hyp,
        )
    else:
        supervision = fastcopy(cut.supervisions[0], text=hyp, start=cut.start)
    return supervision


@torch.no_grad()
def main():
    parser = get_parser()
    LibriCSSAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "fast_beam_search",
        "modified_beam_search",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    params.suffix = f"epoch-{params.epoch}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)

    model.to(device)
    model.eval()
    model.device = device

    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    libricss = LibriCSSAsrDataModule(args)

    dev_ihm_cuts = libricss.dev_ihm_cuts()
    test_ihm_cuts = libricss.test_ihm_cuts()
    dev_sdm_cuts = libricss.dev_sdm_cuts()
    test_sdm_cuts = libricss.test_sdm_cuts()
    dev_gss_cuts = libricss.dev_gss_cuts()
    test_gss_cuts = libricss.test_gss_cuts()

    dev_ihm_dl = libricss.test_dataloaders(dev_ihm_cuts)
    test_ihm_dl = libricss.test_dataloaders(test_ihm_cuts)
    dev_sdm_dl = libricss.test_dataloaders(dev_sdm_cuts)
    test_sdm_dl = libricss.test_dataloaders(test_sdm_cuts)
    dev_gss_dl = libricss.test_dataloaders(dev_gss_cuts)
    test_gss_dl = libricss.test_dataloaders(test_gss_cuts)

    test_sets = [
        "dev_ihm",
        "test_ihm",
        "dev_sdm",
        "test_sdm",
        "dev_gss",
        "test_gss",
    ]
    test_dls = [
        dev_ihm_dl,
        test_ihm_dl,
        dev_sdm_dl,
        test_sdm_dl,
        dev_gss_dl,
        test_gss_dl,
    ]
    test_cuts = [
        dev_ihm_cuts,
        test_ihm_cuts,
        dev_sdm_cuts,
        test_sdm_cuts,
        dev_gss_cuts,
        test_gss_cuts,
    ]

    for ts, td, tc in zip(test_sets, test_dls, test_cuts):
        logging.info(f"Decoding {ts}")
        results_dict = decode_dataset(
            dl=td,
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
        )

        save_results(
            cuts=tc,
            params=params,
            test_set_name=ts,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
