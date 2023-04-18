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
This file merge overlapped chunks into utterances accroding to recording ids.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from cytoolz.itertoolz import groupby

import sentencepiece as spm
from icefall.utils import store_transcripts, write_error_stats
from lhotse import CutSet, load_manifest
from lhotse import SupervisionSegment, MonoCut

from gigaspeech_scoring import asr_text_post_processing


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
        "--extra",
        type=float,
        default=2.0,
        help="""Extra duration (in seconds) at both sides.""",
    )

    return parser.parse_args()


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
    cut_groups = groupby(lambda cut: cut.recording.id, cuts_chunk)

    # Get original cuts by recording ids
    orig_cuts = {c.recording.id: c for c in cuts_orig}

    utt_cut_list = []
    for recording_id, cuts in cut_groups.items():
        # For each group with a same recording, sort it accroding to the start time
        chunk_cuts = sorted(cuts, key=(lambda cut: cut.start))

        orig_cut = orig_cuts[recording_id]
        old_sup = orig_cut.supervisions[0]
        old_text = " ".join(s.text for s in orig_cut.supervisions)
        old_text = asr_text_post_processing(old_text)

        rec = chunk_cuts[0].recording
        alignments = []
        cur_end = 0
        for cut in chunk_cuts:
            # Get left and right borders
            left = cut.start + extra if cut.start > 0 else 0
            chunk_end = cut.start + cut.duration
            right = chunk_end - extra if chunk_end < rec.duration else rec.duration

            # Assert the chunks are continuous
            assert left == cur_end, (left, cur_end)
            cur_end = right

            assert len(cut.supervisions) == 1, len(cut.supervisions)
            for ali in cut.supervisions[0].alignment["symbol"]:
                t = ali.start + cut.start
                if left <= t < right:
                    alignments.append(ali.with_offset(cut.start))

        # Decode the BPE tokens to text
        hyp = [ali.symbol for ali in alignments]
        hyp_text = sp.decode(hyp)
        hyp_text = asr_text_post_processing(hyp_text)

        new_sup = SupervisionSegment(
            id=rec.id,
            recording_id=rec.id,
            start=0,
            duration=rec.duration,
            text=hyp_text,
            alignment={"symbol": alignments},
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
        print("WER: {:.2f}".format(wer * 100), file=f)


def main():
    args = get_parser()

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    for part in ["DEV", "TEST"]:
        logging.info(f"Scoring {part}...")
        recog_cuts = load_manifest(args.res_dir / f"cuts_{part}.jsonl.gz")
        orig_cuts = load_manifest(args.manifest_dir / f"cuts_{part}_full.jsonl.gz")
        out_file = args.res_dir / f"cuts_{part}_merged.jsonl.gz"

        merged_cuts = merge_chunks(
            recog_cuts,
            orig_cuts,
            sp=sp,
            extra=args.extra,
        )
        merged_cuts.to_file(out_file)
        logging.info(f"Cuts saved to {out_file}")

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
