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
./zipformer/decode_chunked.py \
        --epoch 50 \
        --avg 22 \
        --exp-dir ./zipformer/exp \
        --max-duration 600 \
        --decoding-method greedy_search \
        --chunk 30 --extra 4

(2) modified beam search
./zipformer/decode_chunked.py \
        --epoch 50 \
        --avg 22 \
        --exp-dir ./zipformer/exp \
        --max-duration 600 \
        --decoding-method modified_beam_search \
        --beam-size 4 \
        --chunk 30 --extra 4
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
from asr_datamodule import TedLiumAsrDataModule
from beam_search import (
    beam_search,
    fast_beam_search_one_best,
    fast_beam_search_nbest,
    fast_beam_search_nbest_LG,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from train import (
    add_model_arguments,
    get_params,
    get_model,
)

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    convert_timestamp,
    make_pad_mask,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)
from lhotse.cut import CutSet
from lhotse.supervision import AlignmentItem
from lhotse.utils import LOG_EPSILON


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
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
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
          - fast_beam_search_nbest
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
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding_method is fast_beam_search_nbest_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--ilme-scale",
        type=float,
        default=0.0,
        help="""
        Used only when --decoding_method is fast_beam_search_LG.
        It specifies the scale for the internal language model estimation.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
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

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    add_model_arguments(parser)

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
    cuts = batch["supervisions"]["cut"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    if params.causal:
        # this seems to cause insertions at the end of the utterance if used with zipformer.
        pad_len = 30
        feature_lens += pad_len
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, pad_len),
            value=LOG_EPSILON,
        )

    x, x_lens = model.encoder_embed(feature, feature_lens)

    src_key_padding_mask = make_pad_mask(x_lens)
    x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

    encoder_out, encoder_out_lens = model.encoder(x, x_lens, src_key_padding_mask)
    encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

    flag = False
    if (
        params.decoding_method == "fast_beam_search"
        or params.decoding_method == "fast_beam_search_1best_HP"
    ):
        res = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            allow_partial=True,
            return_timestamps=True,
        )
    elif params.decoding_method == "fast_beam_search_nbest_LG":
        res = fast_beam_search_nbest_LG(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            nbest_scale=params.nbest_scale,
            ilme_scale=params.ilme_scale,
            allow_partial=True,
            return_timestamps=True,
        )
    elif (
        params.decoding_method == "fast_beam_search_nbest"
        or params.decoding_method == "fast_beam_search_nbest_HP"
    ):
        res = fast_beam_search_nbest(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            nbest_scale=params.nbest_scale,
            allow_partial=True,
            return_timestamps=True,
        )
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
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
        flag = True
        batch_size = encoder_out.size(0)
        res = []

        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                    return_timestamps=True,
                )
            elif params.decoding_method == "beam_search":
                hyp = beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                    return_timestamps=True,
                )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            res.append(hyp)

    hyps = []
    timestamps = []
    scores = []
    for i in range(feature.shape[0]):
        if flag:
            hyps.append(res[i].hyps)
            timestamps.append(
                convert_timestamp(res[i].timestamps, params.subsampling_factor)
            )
        else:
            hyps.append(res.hyps[i])
            timestamps.append(
                convert_timestamp(res.timestamps[i], params.subsampling_factor)
            )
        try:
            scores.append(res.scores[i])
        except TypeError or AttributeError:
            scores.append([0.0] * len(hyps[i]))

    return hyps, timestamps, scores


def decode_dataset(
    cuts_chunked: CutSet,
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      cuts_chunked:
        The cut set with chunks.
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
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 50
    else:
        log_interval = 20

    # First we compute the hyp tokens for all chunks.
    logging.info("Computing encoder representations for all chunks")
    hyp_tokens = {}
    for batch_idx, batch in enumerate(dl):
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        hyps, timestamps, scores = decode_one_batch(
            params=params,
            model=model,
            decoding_graph=decoding_graph,
            batch=batch,
        )
        for cut_id, hyp, timestamp in zip(cut_ids, hyps, timestamps):
            token_alis = [
                AlignmentItem(
                    symbol=symbol,
                    start=start,
                    duration=min(0.2, round(timestamp[i + 1] - start, 2))
                    if i < len(timestamp) - 1
                    else 0.2,
                    score=None,
                )
                for i, (symbol, start) in enumerate(zip(hyp, timestamp))
            ]
            hyp_tokens[cut_id] = token_alis

        num_cuts += len(batch["inputs"])

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    def _sorting_key(k):
        cut_id, chunk_idx = k.rsplit("-", 1)
        return (cut_id, int(chunk_idx))

    # Next, we combine the outputs for all the chunks of the same utterance. We iterate
    # over the sorted keys so that chunks are ordered by time.
    hyp_per_utt = defaultdict(list)
    for k in sorted(hyp_tokens, key=_sorting_key):
        cut_id = k.rsplit("-", 1)[0]
        cut = cuts_chunked[k]
        token_alis = hyp_tokens[k]
        # Remove the parts from both end corresponding to the "extra" padding.
        left = params.extra if cut.start > 0 else 0
        right = (
            cut.duration - params.extra
            if cut.end < cut.recording.duration
            else cut.duration
        )
        token_alis = [ali for ali in token_alis if left <= ali.start < right]
        hyp_per_utt[cut_id] += token_alis

    results = {}
    for cut_id in hyp_per_utt:
        token_alis = hyp_per_utt[cut_id]
        tokens = [ali.symbol for ali in token_alis]
        text = sp.decode(tokens)
        results[cut_id] = (cut_id, text)

    return results


def save_results(
    cuts_full: CutSet,
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, Tuple[str, str]],
):
    test_set_wers = dict()
    results = []
    for key, res in results_dict.items():
        cut = cuts_full[key]
        _, hyp_text = res
        ref_text = " ".join(s.text for s in cut.supervisions)
        results.append((key, ref_text.split(), hyp_text.split()))

    recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
    results = sorted(results, key=lambda x: x[0])
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
    with open(errs_filename, "w") as f:
        wer = write_error_stats(f, f"{test_set_name}", results, enable_log=True)
        test_set_wers[test_set_name] = wer

    logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    TedLiumAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "fast_beam_search",
        "fast_beam_search_1best_HP",
        "fast_beam_search_nbest",
        "fast_beam_search_nbest_HP",
        "fast_beam_search_nbest_LG",
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
        if "nbest" in params.decoding_method:
            params.suffix += f"-nbest-scale-{params.nbest_scale}"
            params.suffix += f"-num-paths-{params.num_paths}"
            if "LG" in params.decoding_method or "HP" in params.decoding_method:
                params.suffix += f"-ngram-lm-scale-{params.ngram_lm_scale}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

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
    model = get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
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
            model.to(device)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if start >= 0:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()
    model.device = device

    if "fast_beam_search" in params.decoding_method:
        if params.decoding_method == "fast_beam_search_nbest_LG":
            lexicon = Lexicon(params.lang_dir)
            word_table = lexicon.word_table
            lg_filename = params.lang_dir / "LG.pt"
            logging.info(f"Loading {lg_filename}")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lg_filename, map_location=device)
            )
            decoding_graph.scores *= params.ngram_lm_scale
        elif (
            params.decoding_method == "fast_beam_search_nbest_HP"
            or params.decoding_method == "fast_beam_search_1best_HP"
        ):
            hp_filename = params.lang_dir / "HP.pt"
            decoding_graph = k2.Fsa.from_dict(
                torch.load(hp_filename, map_location=device)
            )
            decoding_graph.scores *= params.ngram_lm_scale
            word_table = None
        else:
            decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
            word_table = None
    else:
        decoding_graph = None
        word_table = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    tedlium = TedLiumAsrDataModule(args)

    def _chunk_cuts(cuts: CutSet) -> CutSet:
        cuts = cuts.cut_into_windows(duration=params.chunk)
        cuts = cuts.extend_by(
            params.extra, direction="both", pad_silence=False, preserve_id=True
        )
        # Remove existing supervisions and add empty ones.
        cuts = cuts.drop_supervisions()
        cuts = cuts.fill_supervisions()
        return cuts

    dev_full = tedlium.dev_cuts(affix="_lf").to_eager()
    test_full = tedlium.test_cuts(affix="_lf").to_eager()

    dev_chunked = _chunk_cuts(dev_full).to_eager()
    test_chunked = _chunk_cuts(test_full).to_eager()

    dev_dl = tedlium.test_dataloaders(dev_chunked, chunked=True)
    test_dl = tedlium.test_dataloaders(test_chunked, chunked=True)

    test_sets = [f"dev_lf", f"test_lf"]
    test_dls = [dev_dl, test_dl]
    test_cuts_full = [dev_full, test_full]
    test_cuts_chunked = [dev_chunked, test_chunked]

    for name, dl, cuts_full, cuts_chunked in zip(
        test_sets, test_dls, test_cuts_full, test_cuts_chunked
    ):
        results_dict = decode_dataset(
            cuts_chunked=cuts_chunked,
            dl=dl,
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
        )

        save_results(
            cuts_full=cuts_full,
            params=params,
            test_set_name=f"{name}_full",
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
