#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
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
This file merge overlapped chunks into utterances accroding to recording ids. We also
create STM file for reference and CTM file for hypothesis. These can be used for scoring
with NIST asclite.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from cytoolz.itertoolz import groupby

import sentencepiece as spm
from icefall.utils import store_transcripts, write_error_stats
from lhotse import CutSet, load_manifest, SupervisionSegment, MonoCut
from lhotse.supervision import AlignmentItem
from lhotse.utils import fastcopy


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--res-dir",
        type=Path,
        default=None,
        help="Path to directory containing recognition results.",
    )

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Path to directory with full cuts.",
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

    return parser.parse_args()


def get_word_alignments(
    tokens: List[AlignmentItem], words: List[str]
) -> List[AlignmentItem]:
    """
    Get word-level alignments from token-level alignments.

    Args:
      tokens:
        List of token-level alignments.
      words:
        List of words (obtained by decoding the BPE tokens).

    Returns:
      List of word-level alignments.
    """
    start_token = b"\xe2\x96\x81".decode()  # '_'
    onset = None
    offset = None
    flag = True  # flag to indicate whether next word is the start of a word
    res = []

    # First compute the onset and offset of each word. The logic is as follows:
    # 1. If a token starts with `_`, it means it is the start of a new word.
    # 2. If a token does not start with `_`, it means it may be start of a new word
    #    or continuation of a word. It is start of new word if previous token is `_`.
    for i in range(len(tokens)):
        if tokens[i].symbol.startswith(start_token):
            # This means current word has ended. We will add it to the list.
            if onset is not None and offset is not None:
                res.append((onset, offset))
                # reset onset and offset
                onset = None
                offset = None

            if len(tokens[i].symbol) == 1:
                # This is the `_` token. So next token is the start of a word. We turn on the flag.
                flag = True
            else:
                # This is the start of a word
                onset = tokens[i].start
                offset = tokens[i].end
                flag = False
        else:
            if flag is True:
                # This is the first token of a word
                onset = tokens[i].start
                offset = tokens[i].end
                flag = False
            else:
                # This is continuation of a word
                assert onset is not None and offset is not None
                offset = tokens[i].end
    # Add the last word
    if onset is not None and offset is not None:
        res.append((onset, offset))
    if len(words) > len(res):
        words = words[: len(res)]
    if len(res) > len(words):
        res = res[: len(words)]
    assert len(res) == len(words), (len(res), len(words))

    # Then we create word-level alignments
    word_alignments = []
    for i in range(len(res)):
        word_alignments.append(
            AlignmentItem(
                start=res[i][0],
                duration=res[i][1] - res[i][0],
                symbol=words[i],
            )
        )
    return word_alignments


def words_post_processing(text: List[AlignmentItem]) -> List[AlignmentItem]:
    res = []
    for item in text:
        text = item.symbol
        if len(text) == 0:
            continue
        else:
            res.append(
                AlignmentItem(
                    start=item.start,
                    duration=item.duration,
                    symbol=text,
                )
            )
    return res


def merge_chunks(
    cuts_chunk: CutSet,
    cuts_orig: CutSet,
    sp: spm.SentencePieceProcessor,
    extra: float,
) -> CutSet:
    """Merge chunk-wise cuts accroding to recording ids.
    Args:
      cuts_chunk:
        The chunk-wise cuts.
      sp:
        The BPE model.
      extra:
        Extra duration (in seconds) to drop at both sides of each chunk.
    """
    # Divide into groups according to their recording ids
    cut_groups = groupby(
        lambda cut: cut.recording.id, sorted(cuts_chunk, key=lambda c: c.recording_id)
    )

    # Get original cuts by recording ids
    orig_cuts = {c.recording.id: c for c in cuts_orig}

    utt_cut_list = []
    for recording_id, cuts in cut_groups.items():
        # For each group with a same recording, sort it accroding to the start time
        chunk_cuts = sorted(cuts, key=(lambda cut: cut.start))

        orig_cut = orig_cuts[recording_id]
        old_sup = orig_cut.supervisions[0]
        old_text = " ".join(s.text for s in orig_cut.supervisions)

        rec = chunk_cuts[0].recording
        alignments = []
        for cut in chunk_cuts:
            # Get left and right borders
            left = cut.start + extra if cut.start > 0 else 0
            right = cut.end - extra if cut.end < rec.duration else rec.duration

            assert len(cut.supervisions) == 1, len(cut.supervisions)
            alis = cut.supervisions[0].alignment["symbol"]
            for i, ali in enumerate(alis):
                t = ali.start + cut.start
                if left <= t < right:
                    # We assume that a BPE token can be at most 0.2 seconds long.
                    duration = (
                        min(0.2, round(alis[i + 1].start - ali.start, 2))
                        if i < len(alis) - 1
                        else 0.2
                    )
                    alignments.append(
                        AlignmentItem(start=t, duration=duration, symbol=ali.symbol)
                    )

        # Decode the BPE tokens to text
        hyp = [ali.symbol for ali in alignments]
        hyp_text = sp.decode(hyp)

        # We also want to compute word level alignments so we can get CTM file
        # for scoring with NIST asclite.
        hyp_word_alignments = get_word_alignments(alignments, hyp_text.split(" "))
        hyp_word_alignments = words_post_processing(hyp_word_alignments)
        hyp_text = " ".join(ali.symbol for ali in hyp_word_alignments)

        new_sup = SupervisionSegment(
            id=rec.id,
            recording_id=rec.id,
            start=0,
            duration=rec.duration,
            text=hyp_text,
            alignment={"symbol": alignments, "word": hyp_word_alignments},
            language=old_sup.language,
            speaker=old_sup.speaker,
            custom={"orig_text": old_text},
        )

        utt_cut = MonoCut(
            id=rec.id,
            start=0,
            duration=rec.duration,
            channel=0,
            recording=rec,
            supervisions=[new_sup],
        )
        utt_cut_list.append(utt_cut)

    return CutSet.from_cuts(utt_cut_list)


def save_results(
    res_dir: Path,
    test_set_name: str,
    results: List[Tuple[str, List[str], List[str]]],
):
    """Save the results to files.
    Args:
        res_dir:
            The directory to save the results.
        test_set_name:
            The name of the test set (e.g. greedy-DEV).
        results:
            A list of tuples (recording_id, hyp, ref).
    """
    recog_path = res_dir / f"recogs-{test_set_name}.txt"
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = res_dir / f"errs-{test_set_name}.txt"
    with open(errs_filename, "w") as f:
        wer = write_error_stats(f, f"{test_set_name}", results, enable_log=True)

    logging.info("Wrote detailed error stats to {}".format(errs_filename))

    errs_info = res_dir / f"wer-summary-{test_set_name}.txt"
    with open(errs_info, "w") as f:
        print("WER: {:.2f}".format(wer), file=f)


def main():
    args = get_parser()

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    for part in ["dev", "test"]:
        logging.info(f"Scoring {part}...")
        name = f"{part}_chunk{int(args.chunk)}_extra{int(args.extra)}"
        scoring_dir = args.res_dir / f"{name}_scoring"
        scoring_dir.mkdir(exist_ok=True)

        recog_cuts = load_manifest(args.res_dir / f"cuts_{name}.jsonl.gz")
        orig_cuts = load_manifest(args.manifest_dir / f"cuts_{part}_full.jsonl.gz")
        out_file = args.res_dir / f"cuts_{name}_merged.jsonl.gz"
        stm_file = scoring_dir / f"ref.stm"
        ctm_file = scoring_dir / f"hyp.ctm"

        merged_cuts = merge_chunks(
            recog_cuts,
            orig_cuts,
            sp=sp,
            extra=args.extra,
        )
        merged_cuts.to_file(out_file)
        logging.info(f"Cuts saved to {out_file}")

        # Write CTM file
        out_sups = merged_cuts.decompose()[1]
        out_sups.write_alignment_to_ctm(ctm_file, type="word")

        # Write STM file. Some cuts have overlapping segments from the same speaker,
        # which is not allowed in STM file. We will merge these supervisions.
        with open(stm_file, "w") as f:
            for cut in orig_cuts:
                new_sups = []
                for sup in sorted(cut.supervisions, key=lambda x: x.start):
                    if new_sups and new_sups[-1].end >= sup.start:
                        old_sup = new_sups[-1]
                        new_sups[-1] = fastcopy(
                            old_sup,
                            duration=sup.end - old_sup.start,
                            text=old_sup.text + " " + sup.text,
                        )
                    else:
                        new_sups.append(sup)
                for sup in new_sups:
                    print(
                        f"{sup.recording_id} {sup.channel} {sup.speaker} {sup.start:.2f} {sup.end:.2f} {sup.text}",
                        file=f,
                    )

        results = []
        for cut in merged_cuts:
            ref = cut.supervisions[0].custom["orig_text"]
            hyp = cut.supervisions[0].text
            # convert ref and hyp to list of words
            ref = ref.split()
            hyp = hyp.split()
            results.append((cut.id, ref, hyp))
        save_results(args.res_dir, part, results)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
