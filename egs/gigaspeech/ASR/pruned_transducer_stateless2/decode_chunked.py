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

(2) beam search
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import GigaSpeechAsrDataModule
from beam_search import (
    fast_beam_search_one_best,
    greedy_search_batch,
    modified_beam_search,
)
from train import get_params, get_transducer_model

from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
from icefall.utils import (
    AttributeDict,
    convert_timestamp,
    setup_logger,
)
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.supervision import AlignmentItem
from lhotse.serialization import SequentialJsonlWriter


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--chunk",
        type=float,
        default=30.0,
        help="""Chunk duration (in seconds) for decoding.""",
    )

    parser.add_argument(
        "--extra",
        type=float,
        default=2.0,
        help="""Extra duration (in seconds) at both sides.""",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=29,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 0.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=8,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
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
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[List[str]]]:
    """Decode one batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or LG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return the decoding result, timestamps, and scores.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)

    if params.decoding_method == "fast_beam_search":
        res = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            return_timestamps=True,
        )
    elif params.decoding_method == "greedy_search":
        res = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            return_timestamps=True,
        )
    elif params.decoding_method == "modified_beam_search":
        res = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            return_timestamps=True,
        )
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

    hyps = []
    timestamps = []
    scores = []
    for i in range(feature.shape[0]):
        hyps.append(res.hyps[i])
        timestamps.append(
            convert_timestamp(res.timestamps[i], params.subsampling_factor)
        )
        scores.append(res.scores[i])

    return hyps, timestamps, scores


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    decoding_graph: Optional[k2.Fsa] = None,
    cuts_writer: SequentialJsonlWriter = None,
) -> None:
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
      cuts_writer:
        If not None, it is used to write the cuts with the decoding result.
    """
    #  Background worker to add alignemnt and save cuts to disk.
    def _save_worker(
        cuts: List[Cut],
        hyps: List[List[str]],
        timestamps: List[List[float]],
        scores: List[List[float]],
    ):
        for cut, symbol_list, time_list, score_list in zip(
            cuts, hyps, timestamps, scores
        ):
            symbol_list = sp.id_to_piece(symbol_list)
            ali = [
                AlignmentItem(symbol=symbol, start=start, duration=None, score=score)
                for symbol, start, score in zip(symbol_list, time_list, score_list)
            ]
            assert len(cut.supervisions) == 1, len(cut.supervisions)
            cut.supervisions[0].alignment = {"symbol": ali}
            cuts_writer.write(cut, flush=True)

    num_cuts = 0
    log_interval = 10
    futures = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        # We only want one background worker so that serialization is deterministic.

        for batch_idx, batch in enumerate(dl):
            cuts = batch["supervisions"]["cut"]

            hyps, timestamps, scores = decode_one_batch(
                params=params,
                model=model,
                decoding_graph=decoding_graph,
                batch=batch,
            )

            futures.append(
                executor.submit(_save_worker, cuts, hyps, timestamps, scores)
            )

            num_cuts += len(cuts)
            if batch_idx % log_interval == 0:
                logging.info(f"cuts processed until now is {num_cuts}")


@torch.no_grad()
def main():
    parser = get_parser()
    GigaSpeechAsrDataModule.add_arguments(parser)
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
    params.res_dir = params.exp_dir / f"{params.decoding_method}-chunked"

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam_size}"
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

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))
    elif params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))

    model.to(device)
    model.eval()
    model.device = device

    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    gigaspeech = GigaSpeechAsrDataModule(args)

    affix = f"_chunk{int(params.chunk)}_extra{int(params.extra)}"
    dev_cuts = gigaspeech.dev_cuts(affix=affix)
    # test_cuts = gigaspeech.test_cuts(affix=affix)

    dev_dl = gigaspeech.test_dataloaders(dev_cuts, chunked=True)
    # test_dl = gigaspeech.test_dataloaders(test_cuts, chunked=True)

    test_sets = [f"DEV{affix}"]
    test_dls = [dev_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        cuts_writer = CutSet.open_writer(
            f"{params.res_dir}/cuts_{test_set}.jsonl.gz", overwrite=True
        )
        decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
            cuts_writer=cuts_writer,
        )
        cuts_writer.close()

    logging.info("Done!")


if __name__ == "__main__":
    main()
