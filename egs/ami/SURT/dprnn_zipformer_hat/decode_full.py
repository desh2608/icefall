#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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
./dprnn_zipformer/decode.py \
    --epoch 30 \
    --avg 9 \
    --use-averaged-model true \
    --exp-dir ./dprnn_zipformer/exp \
    --max-duration 600 \
    --decoding-method greedy_search

(2) modified beam search
./dprnn_zipformer/decode.py \
    --epoch 30 \
    --avg 9 \
    --use-averaged-model true \
    --exp-dir ./dprnn_zipformer/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4
"""


import argparse
import json
import logging
import random
from collections import defaultdict
from itertools import chain, groupby
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import AmiAsrDataModule
from beam_search2 import (
    beam_search,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
    speaker_guided_beam_search,
)
from kaldialign import align, edit_distance
from lhotse.utils import EPSILON
from meeteval.wer import wer
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from train2 import add_model_arguments, get_params, get_surt_model

from icefall import LmScorer, NgramLm
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_surt_error_stats,
)

OVERLAP_RATIOS = ["0L", "0S", "OV10", "OV20", "OV30", "OV40"]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
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
        default=9,
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
        default="dprnn_zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - speaker_guided_beam_search
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
        "--use-speaker-prefixing",
        type=str2bool,
        default=False,
        help="Whether to prefix speaker frames from previous chunks.",
    )

    parser.add_argument(
        "--speaker-buffer-frames",
        type=int,
        default=32,
        help="Number of frames to prefix for each speaker.",
    )

    parser.add_argument(
        "--oracle-speaker-buffer",
        type=str,
        default=None,
        help="Path to the oracle speaker buffer.",
    )

    parser.add_argument(
        "--save-aux-labels",
        type=str2bool,
        default=False,
        help="""If true, save the auxiliary labels.""",
    )

    parser.add_argument(
        "--compute-orcwer",
        type=str2bool,
        default=False,
        help="""If true, compute ORC-WER""",
    )

    add_model_arguments(parser)

    return parser


def get_subarray_with_largest_sum(logits: torch.Tensor, k: int):
    """
    Subroutine for finding a subarray of size k with maximum sum.
    Returns the start and end indices of the subarray.
    """
    logits_ = logits.tolist()
    N = len(logits_)
    if k > N:
        return 0, N
    # Compute sum of first window of size k
    res = sum(logits_[:k])
    start = 0

    # Compute sums of remaining windows by removing first element of previous
    # window and adding last element of current window.
    curr_sum = res
    for i in range(k, N):
        curr_sum += logits_[i] - logits_[i - k]
        if curr_sum > res:
            res = curr_sum
            start = i - k + 1
    return start, start + k


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    speaker_buffer: Dict[str, torch.Tensor],
    speaker_confidence: Dict[str, List[float]],
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
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    cuts = batch["cuts"]
    assert feature.ndim == 3

    feature = feature.to(device)
    feature_lens = batch["input_lens"].to(device)

    # Apply the mask encoder
    B, T, F = feature.shape
    h, h_lens, _, masks = model.forward_mask_encoder(feature, feature_lens)

    # Apply the encoder
    (
        encoder_out,
        encoder_out_lens,
        aux_encoder_out,
        aux_encoder_out_lens,
    ) = model.forward_encoder(
        h,
        h_lens,
    )

    N = encoder_out.size(0)
    num_channels = N // B

    if params.use_speaker_prefixing:
        # initialize an empty prefix
        prefix = torch.zeros(
            (
                feature.shape[0],
                params.max_speakers * params.speaker_buffer_frames,
                feature.shape[2],
            ),
            device=device,
        )
        # put existing speaker buffer into prefix
        for i, cut in enumerate(cuts):
            session_id = cut.recording_id
            if session_id in speaker_buffer:
                session_prefix = speaker_buffer[session_id]
                if isinstance(session_prefix, list):
                    session_prefix = random.choice(session_prefix)
                prefix[i, : session_prefix.shape[0]] = session_prefix

        # append the prefix in front of the feature.
        feature = torch.cat([prefix, feature], dim=1)
        feature_lens = feature_lens + prefix.shape[1]

        # Apply the mask encoder
        B, T, F = feature.shape
        h, h_lens, _, masks = model.forward_mask_encoder(feature, feature_lens)

        # Apply the encoder
        *_, aux_encoder_out, aux_encoder_out_lens = model.forward_encoder(
            h,
            h_lens,
        )

    if model.joint_encoder_layer is not None:
        encoder_out = model.joint_encoder_layer(encoder_out)

    if model.aux_joint_encoder_layer is not None:
        aux_encoder_out = model.aux_joint_encoder_layer(
            aux_encoder_out, aux_encoder_out_lens
        )

    # Remove the prefix frames
    if params.use_speaker_prefixing:
        N_prefix = ((prefix.shape[1] - 7) // 2 + 1) // 2
        aux_encoder_out = aux_encoder_out[:, N_prefix:]

    hyps = []  # contains speaker-wise hyps
    hyps_nospk = []  # contains text-only without spk
    raw_hyps = []  # contains channel-wise hyps
    if params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        results = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            aux_encoder_out=aux_encoder_out,
            return_aux_probs=params.use_speaker_prefixing
            and params.oracle_speaker_buffer is not None,
        )
        if params.use_speaker_prefixing and params.oracle_speaker_buffer is not None:
            results, speaker_log_probs = results
    elif params.decoding_method == "modified_beam_search":
        results = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            aux_encoder_out=aux_encoder_out,
            beam=params.beam_size,
        )
    elif params.decoding_method == "speaker_guided_beam_search":
        results = speaker_guided_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            aux_encoder_out=aux_encoder_out,
            beam=params.beam_size,
        )
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
            hyps.append(sp.decode(hyp))

    # Keep track of predicted speakers for each batch item. This will be needed for
    # updating the speaker buffer and speaker confidence.
    predicted_speakers = []
    for i in range(B):
        cur_hyps = []
        for j in range(num_channels):
            cur_hyps.append(results[i + j * B])

        # Get hyps by channel
        raw_hyps.append(
            [
                [(token, spk) for token, spk in zip(hyp.hyps, hyp.aux_hyps)]
                for hyp in cur_hyps
            ]
        )

        # Get hyps without speaker
        hyps_nospk.append([sp.decode(hyp.hyps) for hyp in cur_hyps])

        # Get hyps by speaker
        hyps_by_speaker = defaultdict(list)
        for hyp in cur_hyps:
            for token, spk, ts in zip(hyp.hyps, hyp.aux_hyps, hyp.timestamps):
                hyps_by_speaker[spk].append((token, ts))
        hyps_by_speaker = dict(hyps_by_speaker)
        # For each speaker, order the tokens by timestamp. We also remove
        # duplicated tokens at the same timestamp.
        hyps_by_speaker = {
            spk: [token for token, _ in sorted(set(tokens), key=lambda x: x[1])]
            for spk, tokens in hyps_by_speaker.items()
        }
        # For each speaker, convert the tokens to words.
        hyps_by_speaker = {
            spk: sp.decode(tokens) for spk, tokens in hyps_by_speaker.items()
        }
        hyps.append(hyps_by_speaker)
        # Update the predicted speakers
        predicted_speakers.append(list(hyps_by_speaker.keys()))

    # Update speaker buffer and speaker confidence
    if params.use_speaker_prefixing and params.oracle_speaker_buffer is not None:
        assert len(speaker_log_probs) == B * num_channels
        # split speaker log-probs by channel and add all channels to get a log-prob tensor
        # of shape (B, T, max_speakers)
        speaker_log_probs = torch.stack(
            torch.chunk(speaker_log_probs, num_channels, dim=0), dim=0
        ).max(dim=0, keepdim=False)[0]

        for i in range(B):
            session_id = cuts[i].recording_id
            if session_id in speaker_buffer:
                old_buffer = speaker_buffer[session_id]
                old_conf = speaker_confidence[session_id]
            else:
                old_buffer = torch.zeros(
                    (
                        params.max_speakers * params.speaker_buffer_frames,
                        feature.shape[2],
                    ),
                    device=device,
                )
                old_conf = torch.ones(params.max_speakers, device=device) * -torch.inf
            # For each predicted speaker, we get the highest confident frames.
            # NOTE: predicted speakers is 1-indexed, while speaker_logits is 0-indexed.
            for spk in predicted_speakers[i]:
                cur_spk_log_probs = speaker_log_probs[i, :, spk - 1]
                start_idx, end_idx = get_subarray_with_largest_sum(
                    cur_spk_log_probs,
                    params.speaker_buffer_frames // params.subsampling_factor,
                )
                new_conf = cur_spk_log_probs[start_idx:end_idx].sum()
                # topk_values, topk_indices = torch.topk(
                #     cur_spk_log_probs,
                #     params.speaker_buffer_frames // params.subsampling_factor,
                # )
                # new_conf = topk_values.sum()
                if new_conf > 2 * old_conf[spk - 1]:
                    # Update speaker buffer
                    start_frame = start_idx * params.subsampling_factor
                    end_frame = min(
                        end_idx * params.subsampling_factor, feature_lens[i]
                    )
                    N_frames = end_frame - start_frame
                    buffer_start_idx = (spk - 1) * params.speaker_buffer_frames
                    old_buffer[
                        buffer_start_idx : buffer_start_idx + N_frames
                    ] = feature[i, start_frame:end_frame]
                    old_conf[spk - 1] = new_conf
                    # topk_frame_indices = topk_indices * params.subsampling_factor
                    # buffer_start_idx = (spk - 1) * params.speaker_buffer_frames
                    # old_buffer[
                    #     buffer_start_idx : buffer_start_idx + len(topk_frame_indices)
                    # ] = feature[i, topk_frame_indices]
                    # old_conf[spk - 1] = new_conf
            speaker_buffer[session_id] = old_buffer
            speaker_confidence[session_id] = old_conf

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}, hyps_nospk, raw_hyps
    else:
        return ({f"beam_size_{params.beam_size}": hyps}, hyps_nospk, raw_hyps)


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
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
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
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

    results = defaultdict(list)
    results_nospk = defaultdict(list)
    raw_results = defaultdict(list)

    # Initialize empty speaker buffer and speaker confidence.
    # The `speaker_buffer` is a dict whose key is the session ID, and the value is a
    # tensor of shape (max_speakers*speaker_buffer_frames, num_features). It contains
    # the most confident frames of all speakers seen so far in the session.
    # The `speaker_confidence` is a dict whose key is the session ID, and the value
    # is a list of floats of length `max_speakers`. It contains the confidence of each
    # speaker in the session. We will update the speaker buffer and speaker confidence
    # only if we find frames which have higher confidence than the existing ones.
    speaker_buffer = {}
    speaker_confidence = {}

    if params.oracle_speaker_buffer is not None:
        speaker_buffer = torch.load(params.oracle_speaker_buffer)
        for session_id in speaker_buffer:
            speaker_confidence[session_id] = torch.ones(params.max_speakers) * torch.inf

    for batch_idx, batch in enumerate(dl):
        cut_ids = [cut.id for cut in batch["cuts"]]
        cuts_batch = batch["cuts"]

        # The speaker buffer and speaker confidence are updated in `decode_one_batch`.
        hyps_dict, hyps_nospk, raw_hyps = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            speaker_buffer=speaker_buffer,
            speaker_confidence=speaker_confidence,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            for cut_id, hyp_words in zip(cut_ids, hyps):
                # Reference is a list of supervision texts sorted by start time.
                # Group reference supervisions by speaker.
                ref_words = defaultdict(list)
                for s in sorted(cuts_batch[cut_id].supervisions, key=lambda s: s.start):
                    ref_words[s.speaker].append(s.text.strip())
                ref_words = dict(ref_words)
                # Convert reference words to a single string.
                ref_words = {spk: " ".join(words) for spk, words in ref_words.items()}
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

            this_batch = []
            for cut_id, hyp_words in zip(cut_ids, hyps_nospk):
                # Reference is a list of supervision texts sorted by start time.
                ref_words = [
                    s.text.strip()
                    for s in sorted(
                        cuts_batch[cut_id].supervisions, key=lambda s: s.start
                    )
                ]
                this_batch.append((cut_id, ref_words, hyp_words))

            results_nospk[name].extend(this_batch)

            this_batch = []
            for cut_id, hyp_tokens in zip(cut_ids, raw_hyps):
                # Reference is a list of supervision texts sorted by start time.
                ref_tokens = [
                    (sp.encode(s.text.strip()), s.speaker)
                    for s in sorted(
                        cuts_batch[cut_id].supervisions, key=lambda s: s.start
                    )
                ]
                this_batch.append((cut_id, ref_tokens, hyp_tokens))

            raw_results[name].extend(this_batch)

        num_cuts += len(cut_ids)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return results, results_nospk, raw_results


def compute_cpWER(ref_text, hyp_text):
    """
    ref_text and hyp_text are lists of strings.
    """
    M = len(ref_text)
    N = len(hyp_text)
    costs = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            cur_ref = ref_text[i].split()
            cur_hyp = hyp_text[j].split()
            result = edit_distance(cur_ref, cur_hyp)
            wer = result["total"] / len(cur_ref)
            costs[i, j] = wer
    row_ind, col_ind = linear_sum_assignment(costs)
    ref_text_ = [ref_text[i] for i in row_ind]
    hyp_text_ = [hyp_text[i] for i in col_ind]
    # Add other strings that are not matched
    for i in range(M):
        if i not in row_ind:
            ref_text_.append(ref_text[i])
            hyp_text_.append("")
    for j in range(N):
        if j not in col_ind:
            ref_text_.append("")
            hyp_text_.append(hyp_text[j])
    count = num_ins = num_del = num_sub = total = 0
    for ref, hyp in zip(ref_text_, hyp_text_):
        ref = ref.strip().split()
        hyp = hyp.strip().split()
        count += len(ref)
        result = edit_distance(ref, hyp)
        num_ins += result["ins"]
        num_del += result["del"]
        num_sub += result["sub"]
        total += result["total"]
    count = max(1, count)  # avoid division by zero
    return {
        "ref_text": ref_text_,
        "hyp_text": hyp_text_,
        "count": count,
        "num_ins": num_ins,
        "num_del": num_del,
        "num_sub": num_sub,
        "ins": num_ins / count,
        "del": num_del / count,
        "sub": num_sub / count,
        "cpwer": total / count,
    }, col_ind


def compute_wder(ref_tokens, hyp_tokens, sp):
    """
    Compute word diarization error rate. This is inspired by the WDER defined in
    https://arxiv.org/pdf/1907.05337.pdf with some differences. First, we use the
    ref-to-hyp speaker mapping from the cpWER computation. Second, we use ORC-WER
    to get the ref-hyp alignment.
    """
    # First we convert the ref tokens to words
    ref_words = [(sp.decode(tokens), spk) for (tokens, spk) in ref_tokens]

    # Now convert the hyp tokens to words. For this, we cannot directly use sp.decode
    # since each channel contains interleaved speakers. We will split each channel
    # at speaker change, decode the segments, and then concatenate them.
    hyp_words = []
    for hyp_channel in hyp_tokens:
        hyp_channel_words = []
        for cur_spk, hyp_segment in groupby(hyp_channel, key=lambda x: x[1]):
            words = sp.decode([token for (token, spk) in hyp_segment])
            hyp_channel_words += [(word, cur_spk) for word in words.split()]
        hyp_words.append(hyp_channel_words)

    ref = [utt for (utt, spk) in ref_words]
    hyp = [" ".join([word for (word, spk) in hyp_channel]) for hyp_channel in hyp_words]

    # Get the optimal reference assignment based on ORC-WER
    orc_wer = wer.orc_word_error_rate(ref, hyp)
    assignment = orc_wer.assignment

    # Assign references to channels
    hyps = hyp_words
    refs = [[] for _ in range(len(hyps))]
    for channel, (utt, spk) in zip(assignment, ref_words):
        refs[channel] += [(word, spk) for word in utt.split()]

    # Now compute the WDER for each channel
    total_words = 0
    num_word_corr = 0
    num_spk_corr = 0
    for ref_c, hyp_c in zip(refs, hyps):
        ref = [word for (word, spk) in ref_c]
        hyp = [word for (word, spk) in hyp_c]
        ali = align(ref, hyp, "*")
        total_words += len(ref)
        i = 0
        j = 0
        for ref_word, hyp_word in ali:
            if ref_word == hyp_word:
                num_word_corr += 1
                if ref_c[i][1] == hyp_c[j][1]:
                    num_spk_corr += 1
            if ref_word != "*":
                i += 1
            if hyp_word != "*":
                j += 1
    return {
        "total_words": total_words,
        "num_correct_words": num_word_corr,
        "num_correct_spks": num_spk_corr,
        "wder": 1 - num_spk_corr / max(1, num_word_corr),
    }


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
    results_nospk_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
    raw_results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]] = None,
    sp: Optional[spm.SentencePieceProcessor] = None,
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        stats_path = params.res_dir / f"stats-{test_set_name}-key-{params.suffix}.txt"
        spk_mapping_path = (
            params.res_dir / f"spk_mapping-{test_set_name}-{key}-{params.suffix}.json"
        )

        raw_results = raw_results_dict[key]
        results = sorted(results, key=lambda x: x[0])
        results_nospk = sorted(results_nospk_dict[key], key=lambda x: x[0])
        raw_results = sorted(raw_results, key=lambda x: x[0])
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        if params.save_aux_labels:
            aux_labels = {}
            for cut_id, ref_tokens, hyp_tokens in raw_results:
                aux_labels[cut_id] = [
                    [spk for _, spk in channel_hyp] for channel_hyp in hyp_tokens
                ]
            aux_labels_path = (
                params.res_dir
                / f"aux_labels-{test_set_name}-{key}-{params.suffix}.json"
            )
            with aux_labels_path.open("w") as f:
                json.dump(aux_labels, f, indent=2)

        # The following saves the per-cut outputs.
        res_filename = (
            params.res_dir / f"res-nospk-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(res_filename, "w") as f:
            json.dump(results_nospk, f, indent=2)

        logging.info("Wrote ORC-WER results to {}".format(res_filename))

        # The following prints out ORC-WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_surt_error_stats(
                f,
                f"{test_set_name}-{key}",
                results_nospk,
                enable_log=True,
                num_channels=params.num_channels,
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

        # Group the cuts by session
        results = {
            reco_id: list(cuts)
            for reco_id, cuts in groupby(results, key=lambda x: x[0].split("-")[0])
        }
        raw_results = {
            reco_id: list(cuts)
            for reco_id, cuts in groupby(raw_results, key=lambda x: x[0].split("-")[0])
        }

        wer_dict = {}
        num_spk_dict = {}
        # results_dict is a dict whose key is the cut ID, and the value is a list of tuples,
        # of the form (relative_spk_id, absolute_spk_id)
        spk_mapping = defaultdict(list)
        num_spk_to_recos = defaultdict(list)

        for reco_id in results:
            cuts = results[reco_id]
            raw_cuts = raw_results[reco_id]

            # We need to sort the cuts by segment index
            cuts = sorted(cuts, key=lambda x: int(x[0].split("-")[-1]))
            raw_cuts = sorted(raw_cuts, key=lambda x: int(x[0].split("-")[-1]))

            # Combine speaker-wise ref and hyp transcripts into a single string
            ref_text = defaultdict(list)
            hyp_text = defaultdict(list)
            for _, ref_words, hyp_words in cuts:
                for spk, words in ref_words.items():
                    ref_text[spk].append(words)
                for spk, words in hyp_words.items():
                    hyp_text[spk].append(words)
            for spk in ref_text:
                ref_text[spk] = " ".join(ref_text[spk])
            for spk in hyp_text:
                hyp_text[spk] = " ".join(hyp_text[spk])

            ref_spks = list(ref_text.keys())
            hyp_spks = list(hyp_text.keys())

            # Compute cpWER
            stats, ref2hyp_map = compute_cpWER(
                list(ref_text.values()), list(hyp_text.values())
            )
            spk_mapping[reco_id] = [
                (ref_spks[i], hyp_spks[j]) for i, j in enumerate(ref2hyp_map)
            ]
            ref_spk_map = {}
            hyp_spk_map = {}
            for i, j in enumerate(ref2hyp_map):
                ref_spk_map[ref_spks[i]] = i
                hyp_spk_map[hyp_spks[j]] = i

            # Compute Word Diarization Error Rate
            # ref_tokens: [(sup1, spk1), (sup2, spk2), ...]
            # hyp_tokens: [[(ch1_word1, spk1), ...], ...]
            # ref_tokens_full = []
            # hyp_tokens_full = [[], []]  # 2 output channels
            # for _, ref_tokens, hyp_tokens in raw_cuts:
            #     ref_tokens_full += [
            #         (utt, ref_spk_map.get(spk, -1)) for (utt, spk) in ref_tokens
            #     ]
            #     for i, hyp_channel in enumerate(hyp_tokens):
            #         hyp_tokens_full[i] += [
            #             (word, hyp_spk_map.get(spk, -2)) for (word, spk) in hyp_channel
            #         ]
            # stats.update(compute_wder(ref_tokens_full, hyp_tokens_full, sp))

            # Store results
            wer_dict[reco_id] = stats
            num_spk_dict[reco_id] = (len(ref_text), len(hyp_text))
            num_spk_to_recos[len(ref_text)].append(reco_id)

        # Compute results for different number of speakers
        for num_spk in num_spk_to_recos:
            total_num_words = sum(
                wer_dict[reco_id]["count"] for reco_id in num_spk_to_recos[num_spk]
            )
            total_ins = sum(
                wer_dict[reco_id]["num_ins"] for reco_id in num_spk_to_recos[num_spk]
            )
            total_del = sum(
                wer_dict[reco_id]["num_del"] for reco_id in num_spk_to_recos[num_spk]
            )
            total_sub = sum(
                wer_dict[reco_id]["num_sub"] for reco_id in num_spk_to_recos[num_spk]
            )
            avg_ins = total_ins / total_num_words
            avg_del = total_del / total_num_words
            avg_sub = total_sub / total_num_words
            avg_cpwer = (total_ins + total_del + total_sub) / total_num_words

            # total_corr_words = sum(
            #     wer_dict[reco_id]["num_correct_words"]
            #     for reco_id in num_spk_to_recos[num_spk]
            # )
            # total_corr_spks = sum(
            #     wer_dict[reco_id]["num_correct_spks"]
            #     for reco_id in num_spk_to_recos[num_spk]
            # )
            # avg_wder = 1 - total_corr_spks / total_corr_words

            wer_dict[f"TOTAL_{num_spk}"] = {
                "ref_text": [],
                "hyp_text": [],
                "count": total_num_words,
                "num_ins": total_ins,
                "num_del": total_del,
                "num_sub": total_sub,
                "ins": avg_ins,
                "del": avg_del,
                "sub": avg_sub,
                "cpwer": avg_cpwer,
                # "wder": avg_wder,
            }

        # Compute average cpWER overall
        all_recos = [reco_id for reco_id in wer_dict if "TOTAL" not in reco_id]
        total_num_words = sum(wer_dict[reco_id]["count"] for reco_id in all_recos)
        total_ins = sum(wer_dict[reco_id]["num_ins"] for reco_id in all_recos)
        total_del = sum(wer_dict[reco_id]["num_del"] for reco_id in all_recos)
        total_sub = sum(wer_dict[reco_id]["num_sub"] for reco_id in all_recos)
        avg_ins = total_ins / total_num_words
        avg_del = total_del / total_num_words
        avg_sub = total_sub / total_num_words
        avg_cpwer = (total_ins + total_del + total_sub) / total_num_words

        # Compute average WDER overall
        # total_corr_words = sum(
        #     wer_dict[reco_id]["num_correct_words"] for reco_id in all_recos
        # )
        # total_corr_spks = sum(
        #     wer_dict[reco_id]["num_correct_spks"] for reco_id in all_recos
        # )
        # avg_wder = 1 - total_corr_spks / total_corr_words

        wer_dict["TOTAL"] = {
            "ref_text": [],
            "hyp_text": [],
            "count": total_num_words,
            "num_ins": total_ins,
            "num_del": total_del,
            "num_sub": total_sub,
            "ins": avg_ins,
            "del": avg_del,
            "sub": avg_sub,
            "cpwer": avg_cpwer,
            # "wder": avg_wder,
        }

        # Write results to file
        with stats_path.open("w") as f:
            json.dump(wer_dict, f, indent=2)

        # Write speaker mapping to file
        with spk_mapping_path.open("w") as f:
            json.dump(spk_mapping, f, indent=2)

        # Print averages
        print(f"Average insertion rate: {avg_ins:.2%}")
        print(f"Average deletion rate: {avg_del:.2%}")
        print(f"Average substitution rate: {avg_sub:.2%}")
        print(f"Average cpWER: {avg_cpwer:.2%}")
        # print(f"Average WDER: {avg_wder:.2%}")
        print(f"Average ORC-WER: {wer}")

        # Print confusion matrix of number of speakers
        y_true = [x[0] for x in num_spk_dict.values()]
        y_pred = [x[1] for x in num_spk_dict.values()]
        cm = confusion_matrix(y_true, y_pred, labels=range(1, max(y_true) + 1))
        print("Confusion matrix of number of speakers:")
        print(cm)


def save_masks(
    params: AttributeDict,
    test_set_name: str,
    masks: List[torch.Tensor],
):
    masks_path = params.res_dir / f"masks-{test_set_name}.pt"
    torch.save(masks, masks_path)
    logging.info(f"The masks are stored in {masks_path}")


def save_aux_encoder_out(
    params: AttributeDict,
    test_set_name: str,
    aux_encoder_out: Dict[Tuple[str, str], torch.Tensor],
):
    aux_encoder_out_path = params.res_dir / f"aux_encoder_out-{test_set_name}.pt"
    torch.save(aux_encoder_out, aux_encoder_out_path)
    logging.info(f"The aux encoder output is stored in {aux_encoder_out_path}")


@torch.no_grad()
def main():
    parser = get_parser()
    LmScorer.add_arguments(parser)
    AmiAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "modified_beam_search",
    ), f"Decoding method {params.decoding_method} is not supported."
    params.res_dir = params.exp_dir / f"{params.decoding_method}_full"

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    assert "," not in params.chunk_size, "chunk_size should be one value in decoding."
    assert (
        "," not in params.left_context_frames
    ), "left_context_frames should be one value in decoding."
    params.suffix += f"-chunk-{params.chunk_size}"
    params.suffix += f"-left-context-{params.left_context_frames}"

    if params.use_speaker_prefixing:
        params.suffix += f"-buffer-{params.speaker_buffer_frames}"
    if params.oracle_speaker_buffer is not None:
        params.suffix += f"-oracle"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_surt_model(params)

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
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
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

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    ami = AmiAsrDataModule(args)

    # NOTE(@desh2608): we filter segments longer than 120s to avoid OOM errors in decoding.
    # However, 99.9% of the segments are shorter than 120s, so this should not
    # substantially affect the results. In future, we will implement an overlapped
    # inference method to avoid OOM errors.

    test_sets = {}
    for split in ["test"]:
        for type in ["ihm-mix", "sdm", "mdm8-bf"]:
            test_sets[f"ami-{split}_{type}"] = (
                ami.ami_cuts(split=split, type=type)
                .trim_to_supervision_groups(max_pause=0.0)
                .filter(lambda c: 0.1 < c.duration)
                .to_eager()
            )

    for test_set, test_cuts in test_sets.items():
        test_dl = ami.test_dataloaders(
            test_cuts,
            use_speaker_prefixing=params.use_speaker_prefixing
            and (params.oracle_speaker_buffer is None),
        )
        results_dict, results_nospk_dict, raw_results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
            results_nospk_dict=results_nospk_dict,
            raw_results_dict=raw_results_dict,
            sp=sp,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
